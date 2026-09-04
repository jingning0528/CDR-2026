import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class HistoryTransformerDTCDR(CrossDomainRecommender):
    r"""DTCDR with candidate-conditioned cross-domain history attention.

    This is NOT a sequential recommender. Interaction history is treated as an
    unordered set of interacted-item embeddings; no positional encoding is used.

    For target-domain prediction:

        query  = target candidate-item embedding
        key/value = source-domain interacted-item embeddings of the same user

    The cross-domain history representation is then combined with the target
    user and target candidate representations for prediction.

    For the auxiliary source-domain loss, the direction is reversed:

        query  = source candidate-item embedding
        key/value = target-domain interacted-item embeddings of the same user

    This model requires overlapping users to obtain genuine cross-domain history
    transfer. Users without history in the opposite domain receive a zero
    cross-domain context vector.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(HistoryTransformerDTCDR, self).__init__(config, dataset)

        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.alpha = config['alpha']

        # This implementation is intentionally based on NeuMF embeddings.
        # It keeps the ablation clean and makes each historical interaction a
        # learned item-embedding token.
        self.num_heads = self._cfg(config, 'history_transformer_num_heads', 4)
        self.ffn_size = self._cfg(
            config, 'history_transformer_ffn_size', 4 * self.embedding_size
        )
        self.attn_dropout = self._cfg(
            config, 'history_transformer_dropout', self.dropout_prob
        )
        self.max_history_len = self._cfg(
            config, 'history_transformer_max_history_len', 50
        )

        if (self.max_history_len is not None
                and int(self.max_history_len) <= 0):
            raise ValueError('history_transformer_max_history_len must be positive')

        if self.embedding_size % self.num_heads != 0:
            raise ValueError(
                'embedding_size ({}) must be divisible by '
                'history_transformer_num_heads ({})'.format(
                    self.embedding_size, self.num_heads
                )
            )

        # ------------------------------------------------------------------
        # Domain-specific user/item embeddings
        # ------------------------------------------------------------------
        self.source_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.source_item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size
        )
        self.target_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.target_item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size
        )

        # ------------------------------------------------------------------
        # Interaction histories from RecBole-CDR.
        # history_item_matrix(domain=...) returns, for every global user ID,
        # the interacted item IDs and corresponding interaction values.
        # ------------------------------------------------------------------
        (
            self.source_history_item_id,
            self.source_history_item_value,
            _,
        ) = dataset.history_item_matrix(domain='source')

        # Cap the source tensors before constructing target histories so the
        # two full padded domain matrices are never retained simultaneously.
        self.source_history_item_id = self._cap_history_storage(
            self.source_history_item_id
        )
        self.source_history_item_value = self._cap_history_storage(
            self.source_history_item_value
        )

        (
            self.target_history_item_id,
            self.target_history_item_value,
            _,
        ) = dataset.history_item_matrix(domain='target')

        # Materialize capped tensors once. The clone is important: a plain
        # slice would remain a view backed by the full padded history storage.
        # With one candidate query, each batch then attends over at most H
        # keys, making attention cost linear in the configured history length.
        self.target_history_item_id = self._cap_history_storage(
            self.target_history_item_id
        )
        self.target_history_item_value = self._cap_history_storage(
            self.target_history_item_value
        )

        self.source_history_item_id = self.source_history_item_id.to(self.device)
        self.source_history_item_value = self.source_history_item_value.to(self.device)
        self.target_history_item_id = self.target_history_item_id.to(self.device)
        self.target_history_item_value = self.target_history_item_value.to(self.device)

        # ------------------------------------------------------------------
        # Candidate-conditioned Transformer-style cross-attention block.
        # MultiheadAttention performs Q/K/V attention; residual + LayerNorm +
        # FFN make this a Transformer-style block rather than simple weighting.
        # ------------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_size,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
        )

        self.attn_norm = nn.LayerNorm(self.embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_size, self.ffn_size),
            nn.ReLU(),
            nn.Dropout(self.attn_dropout),
            nn.Linear(self.ffn_size, self.embedding_size),
        )
        self.ffn_norm = nn.LayerNorm(self.embedding_size)
        self.dropout = nn.Dropout(self.attn_dropout)

        # Domain/type embeddings distinguish query and history semantics while
        # preserving permutation-invariance over history positions.
        self.domain_embedding = nn.Embedding(2, self.embedding_size)
        # 0 = candidate item, 1 = history item
        self.role_embedding = nn.Embedding(2, self.embedding_size)

        # Prediction heads receive:
        #   [domain user, candidate item, cross-domain history context]
        self.source_mlp_layers = MLPLayers(
            [3 * self.embedding_size] + self.mlp_hidden_size,
            self.dropout_prob,
        )
        self.source_mlp_layers.logger = None
        self.source_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.target_mlp_layers = MLPLayers(
            [3 * self.embedding_size] + self.mlp_hidden_size,
            self.dropout_prob,
        )
        self.target_mlp_layers.logger = None
        self.target_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.source_sigmoid = nn.Sigmoid()
        self.target_sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(xavier_normal_initialization)

    @staticmethod
    def _cfg(config, key, default):
        try:
            value = config[key]
        except Exception:
            return default
        return default if value is None else value

    def _cap_history_storage(self, history):
        """Copy at most H columns so full padded storage can be released."""
        if self.max_history_len is None:
            return history
        return history[:, :int(self.max_history_len)].clone()

    def _select_history(self, history_ids, history_values):
        """Optionally limit history length without introducing fake chronology.

        RecBole history matrices are padded. Histories are normally capped
        once during initialization; this remains as a defensive bound for
        externally supplied tensors. The stored order is not assumed temporal.
        """
        if self.max_history_len is None:
            return history_ids, history_values

        max_len = int(self.max_history_len)
        return history_ids[:, :max_len], history_values[:, :max_len]

    def _cross_domain_history_context(
        self,
        user,
        candidate_e,
        history_item_ids,
        history_item_values,
        history_embedding_layer,
        query_domain,
        history_domain,
        return_attention=False,
    ):
        """Candidate queries the opposite-domain interaction-history set.

        Args:
            user: [B]
            candidate_e: [B, D]
            history_item_ids: [num_users, H]
            history_item_values: [num_users, H]
            history_embedding_layer: embedding table for opposite-domain items
            query_domain: 0 for source, 1 for target
            history_domain: 0 for source, 1 for target

        Returns:
            context: [B, D]
            attention_weights (optional): [B, H]
        """
        ids = history_item_ids[user]
        values = history_item_values[user]
        ids, values = self._select_history(ids, values)

        if ids.shape[1] == 0:
            raise ValueError('history matrices must contain at least one column')

        # A position is valid when RecBole reports a non-zero interaction value.
        # This is safer than relying only on item_id == 0 because padding rules
        # can vary with preprocessing.
        valid = values != 0

        # [B, H, D]
        history_e = history_embedding_layer(ids)

        query_domain_vec = self.domain_embedding(
            torch.full(
                (candidate_e.shape[0],),
                query_domain,
                dtype=torch.long,
                device=candidate_e.device,
            )
        )
        history_domain_vec = self.domain_embedding(
            torch.full(
                (candidate_e.shape[0],),
                history_domain,
                dtype=torch.long,
                device=candidate_e.device,
            )
        )

        candidate_role_vec = self.role_embedding(
            torch.zeros(
                candidate_e.shape[0], dtype=torch.long, device=candidate_e.device
            )
        )
        history_role_vec = self.role_embedding(
            torch.ones(
                candidate_e.shape[0], dtype=torch.long, device=candidate_e.device
            )
        )

        # Q: [1, B, D]
        query = (
            candidate_e + query_domain_vec + candidate_role_vec
        ).unsqueeze(0)

        # K/V: [H, B, D]
        history_tokens = (
            history_e
            + history_domain_vec.unsqueeze(1)
            + history_role_vec.unsqueeze(1)
        )
        history_tokens = history_tokens.transpose(0, 1)

        # MultiheadAttention cannot safely process a sample whose entire K/V
        # sequence is masked. Temporarily expose one zero token for such users,
        # then zero their final context afterward.
        has_history = valid.any(dim=1)
        safe_valid = valid.clone()
        no_history = ~has_history

        if no_history.any():
            safe_valid[no_history, 0] = True
            history_tokens[0, no_history, :] = 0.0

        key_padding_mask = ~safe_valid  # True = ignored

        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=history_tokens,
            value=history_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        # Transformer-style residual + normalization + FFN.
        x = self.attn_norm(query + self.dropout(attn_output))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        context = x.squeeze(0)

        # No opposite-domain history -> no transferred context.
        context = torch.where(
            has_history.unsqueeze(-1), context, torch.zeros_like(context)
        )

        if return_attention:
            # PyTorch returns [B, 1, H] for a single query token in versions
            # that keep target length; squeeze robustly.
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.squeeze(1)
            attn_weights = torch.where(
                valid, attn_weights, torch.zeros_like(attn_weights)
            )
            return context, attn_weights

        return context

    def forward(self, user, item, domain='target'):
        return self.neumf_forward(user, item, domain)

    def neumf_forward(self, user, item, domain='source'):
        if domain == 'target':
            # Target prediction:
            # target candidate queries SAME USER'S source-domain history.
            user_e = self.target_user_embedding(user)
            candidate_e = self.target_item_embedding(item)

            source_context = self._cross_domain_history_context(
                user=user,
                candidate_e=candidate_e,
                history_item_ids=self.source_history_item_id,
                history_item_values=self.source_history_item_value,
                history_embedding_layer=self.source_item_embedding,
                query_domain=1,
                history_domain=0,
            )

            features = torch.cat(
                [user_e, candidate_e, source_context], dim=-1
            )
            output = self.target_sigmoid(
                self.target_predict_layer(self.target_mlp_layers(features))
            )

        else:
            # Auxiliary source prediction in the opposite direction.
            user_e = self.source_user_embedding(user)
            candidate_e = self.source_item_embedding(item)

            target_context = self._cross_domain_history_context(
                user=user,
                candidate_e=candidate_e,
                history_item_ids=self.target_history_item_id,
                history_item_values=self.target_history_item_value,
                history_embedding_layer=self.target_item_embedding,
                query_domain=0,
                history_domain=1,
            )

            features = torch.cat(
                [user_e, candidate_e, target_context], dim=-1
            )
            output = self.source_sigmoid(
                self.source_predict_layer(self.source_mlp_layers(features))
            )

        return output.squeeze(-1)

    def get_history_attention(self, user, item, domain='target'):
        """Return attention weights for interpretation/analysis.

        For domain='target', weights correspond to source-history items.
        For domain='source', weights correspond to target-history items.
        """
        if domain == 'target':
            candidate_e = self.target_item_embedding(item)
            _, weights = self._cross_domain_history_context(
                user=user,
                candidate_e=candidate_e,
                history_item_ids=self.source_history_item_id,
                history_item_values=self.source_history_item_value,
                history_embedding_layer=self.source_item_embedding,
                query_domain=1,
                history_domain=0,
                return_attention=True,
            )
            history_ids = self.source_history_item_id[user]
            history_values = self.source_history_item_value[user]
        else:
            candidate_e = self.source_item_embedding(item)
            _, weights = self._cross_domain_history_context(
                user=user,
                candidate_e=candidate_e,
                history_item_ids=self.target_history_item_id,
                history_item_values=self.target_history_item_value,
                history_embedding_layer=self.target_item_embedding,
                query_domain=0,
                history_domain=1,
                return_attention=True,
            )
            history_ids = self.target_history_item_id[user]
            history_values = self.target_history_item_value[user]

        history_ids, history_values = self._select_history(
            history_ids, history_values
        )
        return history_ids, history_values, weights

    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        source_output = self.neumf_forward(source_user, source_item, 'source')
        target_output = self.neumf_forward(target_user, target_item, 'target')

        loss_s = self.loss(source_output, source_label)
        loss_t = self.loss(target_output, target_label)

        return self.alpha * loss_s + (1 - self.alpha) * loss_t

    def predict(self, interaction):
        """Target-domain prediction, matching your current RecBole-CDR setup."""
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.neumf_forward(user, item, 'target')
