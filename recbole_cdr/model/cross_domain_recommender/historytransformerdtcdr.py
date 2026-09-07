import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class SmallHistoryEncoder(nn.Module):
    """One lightweight Transformer-style self-attention block for a history set.

    The input history is treated as an unordered set, so no positional encoding
    is used. Padding positions are ignored through key_padding_mask.
    """

    def __init__(self, embedding_size, num_heads, ffn_size, dropout):
        super(SmallHistoryEncoder, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, embedding_size),
        )
        self.ffn_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history_tokens, valid_mask):
        """Encode history tokens.

        Args:
            history_tokens: [H, B, D]
            valid_mask: [B, H], True for real history positions

        Returns:
            encoded_history: [H, B, D]
            self_attention_weights: [B, H, H] (or version-dependent equivalent)
        """
        has_history = valid_mask.any(dim=1)
        safe_valid = valid_mask.clone()
        no_history = ~has_history

        # MultiheadAttention cannot safely handle a sample for which every key
        # is masked. Expose one zero token temporarily, then zero the complete
        # output for those users afterward.
        if no_history.any():
            safe_valid[no_history, 0] = True
            history_tokens = history_tokens.clone()
            history_tokens[0, no_history, :] = 0.0

        key_padding_mask = ~safe_valid

        attn_output, attn_weights = self.self_attention(
            query=history_tokens,
            key=history_tokens,
            value=history_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        x = self.attn_norm(history_tokens + self.dropout(attn_output))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))

        # Make padded tokens exactly zero so they cannot accidentally contribute
        # to later operations if masks are modified or inspected.
        x_bhd = x.transpose(0, 1)  # [B, H, D]
        x_bhd = torch.where(
            valid_mask.unsqueeze(-1), x_bhd, torch.zeros_like(x_bhd)
        )
        x = x_bhd.transpose(0, 1)

        if no_history.any():
            x[:, no_history, :] = 0.0

        return x, attn_weights


class HistoryTransformerDTCDR(CrossDomainRecommender):
    r"""Small Cross-Transformer DTCDR for overlapping-user CDR (e.g., Amazon/Douban Book-Movie).

    This keeps the existing RecBole-CDR interface while replacing direct
    candidate-to-raw-history attention with a two-stage cross-domain block.

    Target-domain prediction (e.g., Book -> Movie):

        source history item embeddings
            -> source-domain projection into a shared latent space
            -> one small self-attention history encoder
            -> target candidate cross-attends to encoded source history
            -> [target user, target candidate, cross-domain context]
            -> target MLP

    Source-domain auxiliary prediction reverses the direction.

    Interaction histories are treated as unordered sets. No positional encoding
    is used. Users without opposite-domain history receive a zero cross-domain
    context vector, so the model remains compatible with partially overlapping
    datasets, although meaningful cross-domain transfer requires user overlap.
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

        # Keep old config keys working so the existing YAML / training script
        # does not need to change.
        self.num_heads = self._cfg(config, 'history_transformer_num_heads', 2)
        self.ffn_size = self._cfg(
            config, 'history_transformer_ffn_size', 2 * self.embedding_size
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
        # Domain-specific user/item embeddings (same public interface/design).
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
        # RecBole-CDR interaction-history matrices.
        # ------------------------------------------------------------------
        (
            self.source_history_item_id,
            self.source_history_item_value,
            _,
        ) = dataset.history_item_matrix(domain='source')

        # Clone the capped source tensors before constructing target histories.
        # A plain slice would retain the full padded backing storage.
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
        # NEW 1: domain-specific projections into one shared preference space.
        # Books and Movies (or any two domains) start from different item
        # embedding tables. These projections make cross-domain comparison
        # explicit instead of forcing attention to align the spaces by itself.
        # ------------------------------------------------------------------
        self.source_projection = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_size),
        )
        self.target_projection = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_size),
        )

        # 0 = candidate token, 1 = history token. This tells the attention block
        # what role a vector plays without introducing sequence positions.
        self.role_embedding = nn.Embedding(2, self.embedding_size)

        # ------------------------------------------------------------------
        # NEW 2: one small self-attention encoder per history domain.
        # It allows related items inside a user's history to contextualize one
        # another before cross-domain transfer.
        # ------------------------------------------------------------------
        self.source_history_encoder = SmallHistoryEncoder(
            self.embedding_size,
            self.num_heads,
            self.ffn_size,
            self.attn_dropout,
        )
        self.target_history_encoder = SmallHistoryEncoder(
            self.embedding_size,
            self.num_heads,
            self.ffn_size,
            self.attn_dropout,
        )

        # ------------------------------------------------------------------
        # NEW 3: candidate-to-contextualized-history cross-attention.
        # A single shared block keeps the model small. Domain-specific projection
        # layers already handle source/target representation differences.
        # ------------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_size,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
        )
        self.cross_attn_norm = nn.LayerNorm(self.embedding_size)
        self.cross_ffn = nn.Sequential(
            nn.Linear(self.embedding_size, self.ffn_size),
            nn.ReLU(),
            nn.Dropout(self.attn_dropout),
            nn.Linear(self.ffn_size, self.embedding_size),
        )
        self.cross_ffn_norm = nn.LayerNorm(self.embedding_size)
        self.dropout = nn.Dropout(self.attn_dropout)

        # Prediction heads remain unchanged in shape:
        # [domain user, raw candidate item, cross-domain context].
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
        """Materialize at most H columns so full history storage is released."""
        if self.max_history_len is None:
            return history
        return history[:, :int(self.max_history_len)].clone()

    def _select_history(self, history_ids, history_values):
        """Optionally cap stored history length without claiming chronology."""
        if self.max_history_len is None:
            return history_ids, history_values

        max_len = int(self.max_history_len)
        return history_ids[:, :max_len], history_values[:, :max_len]

    def _project_source(self, x):
        return self.source_projection(x)

    def _project_target(self, x):
        return self.target_projection(x)

    def _cross_domain_history_context(
        self,
        user,
        candidate_e,
        history_item_ids,
        history_item_values,
        history_embedding_layer,
        candidate_projection,
        history_projection,
        history_encoder,
        return_attention=False,
    ):
        """Encode opposite-domain history, then cross-attend from candidate.

        Args:
            user: [B]
            candidate_e: [B, D]
            history_item_ids: [num_users, H]
            history_item_values: [num_users, H]
            history_embedding_layer: opposite-domain item embedding table
            candidate_projection: projection for candidate's domain
            history_projection: projection for history's domain
            history_encoder: SmallHistoryEncoder for the history domain

        Returns:
            context: [B, D]
            cross_attention_weights (optional): [B, H]
        """
        ids = history_item_ids[user]
        values = history_item_values[user]
        ids, values = self._select_history(ids, values)

        if ids.shape[1] == 0:
            raise ValueError('history matrices must contain at least one column')

        valid = values != 0  # [B, H]
        has_history = valid.any(dim=1)
        no_history = ~has_history

        # Raw history embeddings -> shared preference space.
        history_e = history_embedding_layer(ids)        # [B, H, D]
        history_e = history_projection(history_e)        # [B, H, D]

        history_role = self.role_embedding(
            torch.ones(
                candidate_e.shape[0], dtype=torch.long, device=candidate_e.device
            )
        )
        history_tokens = history_e + history_role.unsqueeze(1)

        # Remove padding contributions before self-attention.
        history_tokens = torch.where(
            valid.unsqueeze(-1), history_tokens, torch.zeros_like(history_tokens)
        )
        history_tokens = history_tokens.transpose(0, 1)  # [H, B, D]

        # Stage 1: contextualize items within the opposite-domain history.
        encoded_history, _ = history_encoder(history_tokens, valid)

        # Candidate -> same shared space.
        candidate_z = candidate_projection(candidate_e)
        candidate_role = self.role_embedding(
            torch.zeros(
                candidate_e.shape[0], dtype=torch.long, device=candidate_e.device
            )
        )
        query = (candidate_z + candidate_role).unsqueeze(0)  # [1, B, D]

        # Safe mask for users with no opposite-domain history.
        safe_valid = valid.clone()
        if no_history.any():
            safe_valid[no_history, 0] = True
            encoded_history = encoded_history.clone()
            encoded_history[0, no_history, :] = 0.0

        key_padding_mask = ~safe_valid

        # Stage 2: candidate-conditioned cross-domain transfer.
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=encoded_history,
            value=encoded_history,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        x = self.cross_attn_norm(query + self.dropout(attn_output))
        x = self.cross_ffn_norm(x + self.dropout(self.cross_ffn(x)))
        context = x.squeeze(0)  # [B, D]

        # Preserve old fallback behavior: no opposite-domain history means no
        # transferred signal. Prediction can still use user + candidate features.
        context = torch.where(
            has_history.unsqueeze(-1), context, torch.zeros_like(context)
        )

        if return_attention:
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
            # Source history -> target candidate (e.g., Books -> Movies).
            user_e = self.target_user_embedding(user)
            candidate_e = self.target_item_embedding(item)

            source_context = self._cross_domain_history_context(
                user=user,
                candidate_e=candidate_e,
                history_item_ids=self.source_history_item_id,
                history_item_values=self.source_history_item_value,
                history_embedding_layer=self.source_item_embedding,
                candidate_projection=self._project_target,
                history_projection=self._project_source,
                history_encoder=self.source_history_encoder,
            )

            features = torch.cat(
                [user_e, candidate_e, source_context], dim=-1
            )
            output = self.target_sigmoid(
                self.target_predict_layer(self.target_mlp_layers(features))
            )

        else:
            # Target history -> source candidate for auxiliary source loss.
            user_e = self.source_user_embedding(user)
            candidate_e = self.source_item_embedding(item)

            target_context = self._cross_domain_history_context(
                user=user,
                candidate_e=candidate_e,
                history_item_ids=self.target_history_item_id,
                history_item_values=self.target_history_item_value,
                history_embedding_layer=self.target_item_embedding,
                candidate_projection=self._project_source,
                history_projection=self._project_target,
                history_encoder=self.target_history_encoder,
            )

            features = torch.cat(
                [user_e, candidate_e, target_context], dim=-1
            )
            output = self.source_sigmoid(
                self.source_predict_layer(self.source_mlp_layers(features))
            )

        return output.squeeze(-1)

    def get_history_attention(self, user, item, domain='target'):
        """Return final cross-attention weights for interpretation/analysis.

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
                candidate_projection=self._project_target,
                history_projection=self._project_source,
                history_encoder=self.source_history_encoder,
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
                candidate_projection=self._project_source,
                history_projection=self._project_target,
                history_encoder=self.target_history_encoder,
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
        """Target-domain prediction, matching the existing RecBole-CDR setup."""
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.neumf_forward(user, item, 'target')
