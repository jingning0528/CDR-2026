import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class HistoryTransformerDTCDR(CrossDomainRecommender):
    r"""Lightweight candidate-to-history Cross-Transformer DTCDR.

    Designed for overlapping-user CDR datasets such as Amazon Books -> Movies
    and Douban Books -> Movies.

    Target-domain prediction:
        source-domain history
            -> source projection
            -> candidate-to-history multi-head cross-attention
            -> residual + FFN
            -> [target user, target candidate, cross-domain context]
            -> target MLP

    The source auxiliary task reverses the direction.

    Histories are treated as unordered preference sets. No positional encoding
    and no history self-attention are used. Users without opposite-domain
    history receive a zero cross-domain context vector.
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

        # Domain-specific embeddings.
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

        # RecBole-CDR history matrices.
        (
            source_history_ids,
            source_history_values,
            _,
        ) = dataset.history_item_matrix(domain='source')

        # Materialize the source cap before constructing target histories so
        # both full padded matrices are not retained simultaneously.
        source_history_ids = self._cap_history_storage(source_history_ids)
        source_history_values = self._cap_history_storage(source_history_values)

        (
            target_history_ids,
            target_history_values,
            _,
        ) = dataset.history_item_matrix(domain='target')

        target_history_ids = self._cap_history_storage(target_history_ids)
        target_history_values = self._cap_history_storage(target_history_values)

        # Register as buffers so they follow model.to(device) and are saved in
        # checkpoints without being treated as trainable parameters.
        self.register_buffer('source_history_item_id', source_history_ids.long())
        self.register_buffer('source_history_item_value', source_history_values)
        self.register_buffer('target_history_item_id', target_history_ids.long())
        self.register_buffer('target_history_item_value', target_history_values)

        # Domain-specific projections align Books/Movies (or other domains)
        # before cross-attention.
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

        # 0 = candidate token, 1 = history token.
        self.role_embedding = nn.Embedding(2, self.embedding_size)

        # One lightweight candidate-to-history cross-attention block.
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

        # Prediction heads: [user, candidate, opposite-domain context].
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
        """Clone at most H columns so full padded storage can be released."""
        if self.max_history_len is None:
            return history
        return history[:, :int(self.max_history_len)].clone()

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
        return_attention=False,
    ):
        """Candidate attends directly to the same user's opposite-domain history."""

        ids = history_item_ids[user]       # [B, H]
        values = history_item_values[user] # [B, H]

        if ids.shape[1] == 0:
            raise ValueError('history matrices must contain at least one column')

        valid = values != 0
        has_history = valid.any(dim=1)
        no_history = ~has_history

        # Opposite-domain history -> shared latent space.
        history_e = history_embedding_layer(ids)   # [B, H, D]
        history_z = history_projection(history_e)  # [B, H, D]

        history_role = self.role_embedding(
            torch.ones(
                candidate_e.shape[0], dtype=torch.long, device=candidate_e.device
            )
        )
        history_tokens = history_z + history_role.unsqueeze(1)
        history_tokens = torch.where(
            valid.unsqueeze(-1), history_tokens, torch.zeros_like(history_tokens)
        )

        # Target/source candidate -> same latent space.
        candidate_z = candidate_projection(candidate_e)
        candidate_role = self.role_embedding(
            torch.zeros(
                candidate_e.shape[0], dtype=torch.long, device=candidate_e.device
            )
        )
        query = (candidate_z + candidate_role).unsqueeze(0)  # [1, B, D]

        # nn.MultiheadAttention uses [L, B, D] with batch_first=False.
        key_value = history_tokens.transpose(0, 1)            # [H, B, D]

        # MultiheadAttention cannot handle a row with all keys masked. Temporarily
        # expose one zero key, then force the final context back to zero.
        safe_valid = valid.clone()
        if no_history.any():
            safe_valid[no_history, 0] = True
            key_value = key_value.clone()
            key_value[0, no_history, :] = 0.0

        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=~safe_valid,
            need_weights=True,
        )

        # Small Transformer-style residual + FFN around cross-attention only.
        x = self.cross_attn_norm(query + self.dropout(attn_output))
        x = self.cross_ffn_norm(x + self.dropout(self.cross_ffn(x)))
        context = x.squeeze(0)  # [B, D]

        context = torch.where(
            has_history.unsqueeze(-1), context, torch.zeros_like(context)
        )

        if return_attention:
            # Typical shape is [B, 1, H].
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
            # Books -> Movies: target movie candidate queries source book history.
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
            )

            features = torch.cat([user_e, candidate_e, source_context], dim=-1)
            output = self.target_sigmoid(
                self.target_predict_layer(self.target_mlp_layers(features))
            )
        else:
            # Auxiliary reverse direction: movie history -> book candidate.
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
            )

            features = torch.cat([user_e, candidate_e, target_context], dim=-1)
            output = self.source_sigmoid(
                self.source_predict_layer(self.source_mlp_layers(features))
            )

        return output.squeeze(-1)

    def get_history_attention(self, user, item, domain='target'):
        """Return history ids/values and final candidate-to-history attention."""
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
                return_attention=True,
            )
            history_ids = self.target_history_item_id[user]
            history_values = self.target_history_item_value[user]

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
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.neumf_forward(user, item, 'target')
