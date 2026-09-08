import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class HistoryTransformerDTCDR(CrossDomainRecommender):
    r"""User-conditioned cross-attention DTCDR.

    Designed for overlapping-user CDR such as Amazon Books -> Movies and
    Douban Books -> Movies.

    For target-domain prediction, the target-domain USER representation is the
    query and the same user's source-domain item history is the key/value set:

        q_u^T = W_T e_u^T
        K_u^S = V_u^S = W_S H_u^S
        c_u^{S->T} = MultiHeadAttention(q_u^T, K_u^S, V_u^S)

    The resulting cross-domain user context is computed once per unique user
    in the current batch and reused for every candidate item of that user.
    This is important for full-ranking evaluation.

    The source auxiliary task reverses the direction.

    Histories are treated as unordered preference sets. No positional encoding
    and no history self-attention are used. Users with no opposite-domain
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
            config, 'history_transformer_dropout', 0.1
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

        # Domain-specific user/item embeddings.
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

        # Per-user histories supplied by RecBole-CDR.
        source_history_ids, source_history_values, _ = \
            dataset.history_item_matrix(domain='source')
        source_history_ids = self._cap_history_storage(source_history_ids)
        source_history_values = self._cap_history_storage(source_history_values)

        # Cap source storage before constructing the target matrices so two
        # full padded domain histories are not retained simultaneously.
        target_history_ids, target_history_values, _ = \
            dataset.history_item_matrix(domain='target')
        target_history_ids = self._cap_history_storage(target_history_ids)
        target_history_values = self._cap_history_storage(target_history_values)

        self.register_buffer(
            'source_history_item_id', source_history_ids.long()
        )
        self.register_buffer(
            'source_history_item_value', source_history_values
        )
        self.register_buffer(
            'target_history_item_id', target_history_ids.long()
        )
        self.register_buffer(
            'target_history_item_value', target_history_values
        )

        # Bring user queries and opposite-domain histories into a comparable
        # latent space before cross-attention.
        self.source_user_projection = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_size),
        )
        self.target_user_projection = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_size),
        )
        self.source_item_projection = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_size),
        )
        self.target_item_projection = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_size),
        )

        # One small Transformer-style cross-attention block.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_size,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.embedding_size)
        self.cross_ffn = nn.Sequential(
            nn.Linear(self.embedding_size, self.ffn_size),
            nn.ReLU(),
            nn.Dropout(self.attn_dropout),
            nn.Linear(self.ffn_size, self.embedding_size),
        )
        self.cross_ffn_norm = nn.LayerNorm(self.embedding_size)
        self.attn_dropout_layer = nn.Dropout(self.attn_dropout)

        # Prediction head keeps the familiar DTCDR-style ingredients:
        # [domain user, candidate item, opposite-domain user context].
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

    def _user_history_context_unique(
        self,
        users,
        query_user_embedding,
        history_item_ids,
        history_item_values,
        history_embedding_layer,
        query_projection,
        history_projection,
        return_attention=False,
    ):
        """Compute opposite-domain context once per UNIQUE user in a batch."""

        unique_users, inverse = torch.unique(users, sorted=False, return_inverse=True)

        # Query comes from the prediction-domain user representation.
        unique_user_e = query_user_embedding(unique_users)          # [U, D]
        query = query_projection(unique_user_e).unsqueeze(1)        # [U, 1, D]

        # Keys/values come from the same users' opposite-domain histories.
        ids = history_item_ids[unique_users]                        # [U, H]
        values = history_item_values[unique_users]                  # [U, H]

        if ids.shape[1] == 0:
            raise ValueError('history matrices must contain at least one column')
        valid = values != 0                                         # [U, H]
        has_history = valid.any(dim=1)                              # [U]
        no_history = ~has_history

        history_e = history_embedding_layer(ids)                    # [U, H, D]
        history_z = history_projection(history_e)                   # [U, H, D]
        history_z = torch.where(
            valid.unsqueeze(-1), history_z, torch.zeros_like(history_z)
        )

        # MultiheadAttention cannot accept a sample with every key masked.
        # Temporarily expose one zero vector, then zero its final context.
        safe_valid = valid.clone()
        if no_history.any():
            safe_valid[no_history, 0] = True
            history_z = history_z.clone()
            history_z[no_history, 0, :] = 0.0

        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=history_z,
            value=history_z,
            key_padding_mask=~safe_valid,
            need_weights=return_attention,
            average_attn_weights=True,
        )

        # Residual + FFN on the single user-query token.
        x = self.cross_attn_norm(
            query + self.attn_dropout_layer(attn_output)
        )
        x = self.cross_ffn_norm(
            x + self.attn_dropout_layer(self.cross_ffn(x))
        )
        unique_context = x.squeeze(1)                               # [U, D]

        unique_context = torch.where(
            has_history.unsqueeze(-1),
            unique_context,
            torch.zeros_like(unique_context),
        )

        # Reuse each user's context for all candidate rows belonging to them.
        context = unique_context[inverse]                           # [B, D]

        if not return_attention:
            return context

        # [U, 1, H] -> [U, H], then expand back to rows in the batch.
        if attn_weights is None:
            weights = torch.zeros_like(values)
        else:
            weights = attn_weights.squeeze(1)
            weights = torch.where(valid, weights, torch.zeros_like(weights))

        return context, ids[inverse], values[inverse], weights[inverse]

    def forward(self, user, item, domain='target'):
        return self.neumf_forward(user, item, domain)

    def neumf_forward(self, user, item, domain='source'):
        if domain == 'target':
            # Books -> Movies:
            # target Movie user queries the same user's Book history ONCE.
            user_e = self.target_user_embedding(user)
            item_e = self.target_item_embedding(item)

            source_context = self._user_history_context_unique(
                users=user,
                query_user_embedding=self.target_user_embedding,
                history_item_ids=self.source_history_item_id,
                history_item_values=self.source_history_item_value,
                history_embedding_layer=self.source_item_embedding,
                query_projection=self.target_user_projection,
                history_projection=self.source_item_projection,
            )

            features = torch.cat([user_e, item_e, source_context], dim=-1)
            output = self.target_sigmoid(
                self.target_predict_layer(self.target_mlp_layers(features))
            )

        else:
            # Auxiliary reverse direction:
            # source Book user queries the same user's Movie history ONCE.
            user_e = self.source_user_embedding(user)
            item_e = self.source_item_embedding(item)

            target_context = self._user_history_context_unique(
                users=user,
                query_user_embedding=self.source_user_embedding,
                history_item_ids=self.target_history_item_id,
                history_item_values=self.target_history_item_value,
                history_embedding_layer=self.target_item_embedding,
                query_projection=self.source_user_projection,
                history_projection=self.target_item_projection,
            )

            features = torch.cat([user_e, item_e, target_context], dim=-1)
            output = self.source_sigmoid(
                self.source_predict_layer(self.source_mlp_layers(features))
            )

        return output.squeeze(-1)

    def get_history_attention(self, user, item=None, domain='target'):
        """Return opposite-domain history and USER-to-history attention.

        `item` is accepted for backward compatibility but is intentionally not
        used: attention now depends on the user, not on the candidate item.
        """
        if domain == 'target':
            _, ids, values, weights = self._user_history_context_unique(
                users=user,
                query_user_embedding=self.target_user_embedding,
                history_item_ids=self.source_history_item_id,
                history_item_values=self.source_history_item_value,
                history_embedding_layer=self.source_item_embedding,
                query_projection=self.target_user_projection,
                history_projection=self.source_item_projection,
                return_attention=True,
            )
        else:
            _, ids, values, weights = self._user_history_context_unique(
                users=user,
                query_user_embedding=self.source_user_embedding,
                history_item_ids=self.target_history_item_id,
                history_item_values=self.target_history_item_value,
                history_embedding_layer=self.target_item_embedding,
                query_projection=self.source_user_projection,
                history_projection=self.target_item_projection,
                return_attention=True,
            )

        return ids, values, weights

    def calculate_loss(self, interaction):
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        target_output = self.neumf_forward(target_user, target_item, 'target')
        return self.loss(target_output, target_label)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.neumf_forward(user, item, 'target')
