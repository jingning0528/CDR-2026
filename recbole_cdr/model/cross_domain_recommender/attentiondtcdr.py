import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class AttentionDTCDR(CrossDomainRecommender):
    r"""DTCDR with adaptive cross-domain attention fusion.

    Compared with the original DTCDR max fusion:

        e_u = max(e_u^S, e_u^T)
        e_i = max(e_i^S, e_i^T)

    this model learns source/target attention weights separately for the user
    and item representations:

        e_u = alpha_S * e_u^S + alpha_T * e_u^T
        e_i = beta_S  * e_i^S + beta_T  * e_i^T

    Missing domain-specific representations are masked before softmax. If only
    one domain representation exists, its attention weight becomes 1.

    Training keeps the original DTCDR joint objective:

        L = alpha * L_source + (1 - alpha) * L_target

    Prediction remains target-domain by default, matching the current
    RecBole-CDR evaluation setup.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(AttentionDTCDR, self).__init__(config, dataset)

        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.base_model = config['base_model']
        self.alpha = config['alpha']

        assert self.base_model in ['NeuMF', 'DMF'], (
            'base model {} is not supported'.format(self.base_model)
        )

        # --------------------------------------------------------------
        # Original DTCDR backbone
        # --------------------------------------------------------------
        if self.base_model == 'NeuMF':
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

            self.source_mlp_layers = MLPLayers(
                [2 * self.embedding_size] + self.mlp_hidden_size,
                self.dropout_prob,
            )
            self.source_mlp_layers.logger = None
            self.source_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

            self.target_mlp_layers = MLPLayers(
                [2 * self.embedding_size] + self.mlp_hidden_size,
                self.dropout_prob,
            )
            self.target_mlp_layers.logger = None
            self.target_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        else:  # DMF
            self.source_history_user_id, self.source_history_user_value, _ = (
                dataset.history_user_matrix(domain='source')
            )
            self.source_history_item_id, self.source_history_item_value, _ = (
                dataset.history_item_matrix(domain='source')
            )
            self.source_interaction_matrix = dataset.inter_matrix(
                form='csr', domain='source'
            ).astype(np.float32)

            self.source_history_user_id = self.source_history_user_id.to(self.device)
            self.source_history_user_value = self.source_history_user_value.to(self.device)
            self.source_history_item_id = self.source_history_item_id.to(self.device)
            self.source_history_item_value = self.source_history_item_value.to(self.device)

            self.target_history_user_id, self.target_history_user_value, _ = (
                dataset.history_user_matrix(domain='target')
            )
            self.target_history_item_id, self.target_history_item_value, _ = (
                dataset.history_item_matrix(domain='target')
            )
            self.target_interaction_matrix = dataset.inter_matrix(
                form='csr', domain='target'
            ).astype(np.float32)

            self.target_history_user_id = self.target_history_user_id.to(self.device)
            self.target_history_user_value = self.target_history_user_value.to(self.device)
            self.target_history_item_id = self.target_history_item_id.to(self.device)
            self.target_history_item_value = self.target_history_item_value.to(self.device)

            self.source_user_linear = nn.Linear(
                self.source_num_items, self.embedding_size, bias=False
            )
            self.source_item_linear = nn.Linear(
                self.source_num_users, self.embedding_size, bias=False
            )
            self.target_user_linear = nn.Linear(
                self.target_num_items, self.embedding_size, bias=False
            )
            self.target_item_linear = nn.Linear(
                self.target_num_users, self.embedding_size, bias=False
            )

            self.source_user_fc_layers = MLPLayers(
                [self.embedding_size] + self.mlp_hidden_size
            )
            self.source_item_fc_layers = MLPLayers(
                [self.embedding_size] + self.mlp_hidden_size
            )
            self.target_user_fc_layers = MLPLayers(
                [self.embedding_size] + self.mlp_hidden_size
            )
            self.target_item_fc_layers = MLPLayers(
                [self.embedding_size] + self.mlp_hidden_size
            )

        # --------------------------------------------------------------
        # Cross-domain attention fusion
        # --------------------------------------------------------------
        # The same scorer is applied to source and target representations,
        # while learned domain embeddings tell it which domain each vector
        # comes from. User and item attention are kept separate so the model
        # can learn different transfer behavior for users and items.
        self.domain_embedding = nn.Embedding(2, self.embedding_size)  # 0=S, 1=T

        self.user_attention = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1, bias=False),
        )

        self.item_attention = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1, bias=False),
        )

        self.source_sigmoid = nn.Sigmoid()
        self.target_sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # Initialize first, then restore unavailable-domain markers.
        self.apply(xavier_normal_initialization)

        if self.base_model == 'NeuMF':
            self._apply_domain_embedding_masks()

    # ------------------------------------------------------------------
    # Availability helpers
    # ------------------------------------------------------------------
    def _apply_domain_embedding_masks(self):
        """Mark IDs that do not exist in each domain with -inf.

        These -inf values are never sent directly into the attention block;
        they are replaced by zero and excluded by the attention mask.
        """
        with torch.no_grad():
            self.target_user_embedding.weight[self.target_num_users:].fill_(-np.inf)
            self.target_item_embedding.weight[self.target_num_items:].fill_(-np.inf)

            self.source_user_embedding.weight[
                self.overlapped_num_users:self.target_num_users
            ].fill_(-np.inf)
            self.source_item_embedding.weight[
                self.overlapped_num_items:self.target_num_items
            ].fill_(-np.inf)

    def _domain_validity(self, user, item):
        """Return validity masks for source/target user and item vectors."""
        user_source_valid = (
            (user < self.overlapped_num_users) | (user >= self.target_num_users)
        )
        user_target_valid = user < self.target_num_users

        item_source_valid = (
            (item < self.overlapped_num_items) | (item >= self.target_num_items)
        )
        item_target_valid = item < self.target_num_items

        user_valid = torch.stack(
            [user_source_valid, user_target_valid], dim=1
        )  # [B, 2]
        item_valid = torch.stack(
            [item_source_valid, item_target_valid], dim=1
        )  # [B, 2]

        return user_valid, item_valid

    # ------------------------------------------------------------------
    # Attention fusion
    # ------------------------------------------------------------------
    def _domain_attention(self, source_e, target_e, valid_mask, scorer):
        """Fuse source/target representations with learned attention.

        Args:
            source_e:  [B, D]
            target_e:  [B, D]
            valid_mask:[B, 2], True if that domain representation exists
            scorer:    user_attention or item_attention

        Returns:
            fused:   [B, D]
            weights: [B, 2] in [source_weight, target_weight] order
        """
        pair = torch.stack([source_e, target_e], dim=1)  # [B, 2, D]

        # Never pass -inf values into the attention network.
        pair = torch.where(
            valid_mask.unsqueeze(-1), pair, torch.zeros_like(pair)
        )

        # Give the scoring network explicit source/target identity.
        domain_ids = torch.tensor([0, 1], device=pair.device)
        score_input = pair + self.domain_embedding(domain_ids).unsqueeze(0)

        # One scalar relevance score for each domain representation.
        scores = scorer(score_input).squeeze(-1)  # [B, 2]

        # Invalid representations receive zero probability after softmax.
        scores = scores.masked_fill(~valid_mask, -1e9)
        weights = torch.softmax(scores, dim=1)

        fused = torch.sum(weights.unsqueeze(-1) * pair, dim=1)
        return fused, weights

    def _cross_domain_fusion(
        self,
        user_source_e,
        user_target_e,
        item_source_e,
        item_target_e,
        user_valid,
        item_valid,
    ):
        """Adaptive source-target fusion for the user and item separately."""
        user_e, user_weights = self._domain_attention(
            user_source_e,
            user_target_e,
            user_valid,
            self.user_attention,
        )
        item_e, item_weights = self._domain_attention(
            item_source_e,
            item_target_e,
            item_valid,
            self.item_attention,
        )
        return user_e, item_e, user_weights, item_weights

    def forward(self, user, item, domain='target'):
        if self.base_model == 'NeuMF':
            return self.neumf_forward(user, item, domain)
        return self.dmf_forward(user, item, domain)

    # ------------------------------------------------------------------
    # NeuMF
    # ------------------------------------------------------------------
    def neumf_forward(self, user, item, domain='source'):
        user_source_e = self.source_user_embedding(user)
        user_target_e = self.target_user_embedding(user)
        item_source_e = self.source_item_embedding(item)
        item_target_e = self.target_item_embedding(item)

        user_valid, item_valid = self._domain_validity(user, item)

        user_e, item_e, _, _ = self._cross_domain_fusion(
            user_source_e,
            user_target_e,
            item_source_e,
            item_target_e,
            user_valid,
            item_valid,
        )

        if domain == 'source':
            output = self.source_sigmoid(
                self.source_predict_layer(
                    self.source_mlp_layers(torch.cat((user_e, item_e), dim=-1))
                )
            )
        else:
            output = self.target_sigmoid(
                self.target_predict_layer(
                    self.target_mlp_layers(torch.cat((user_e, item_e), dim=-1))
                )
            )

        return output.squeeze(-1)

    # ------------------------------------------------------------------
    # DMF
    # ------------------------------------------------------------------
    def construct_matrix(
        self, input_tensor, history_id_matrix, history_value_matrix, length
    ):
        col_indices = history_id_matrix[input_tensor].flatten()
        row_indices = torch.arange(input_tensor.shape[0], device=self.device)
        row_indices = row_indices.repeat_interleave(history_id_matrix.shape[1], dim=0)

        matrix_01 = torch.zeros(
            input_tensor.shape[0], length, device=self.device
        )
        matrix_01.index_put_(
            (row_indices, col_indices), history_value_matrix[input_tensor].flatten()
        )
        return matrix_01

    def dmf_forward(self, user, item, domain='source'):
        # Source-domain user representation
        col_indices = self.source_history_item_id[user].flatten().clone()
        col_indices[col_indices > self.target_num_items] -= (
            self.target_num_items - self.overlapped_num_items
        )
        row_indices = torch.arange(user.shape[0], device=self.device)
        row_indices = row_indices.repeat_interleave(
            self.source_history_item_id.shape[1], dim=0
        )
        source_user_matrix = torch.zeros(
            user.shape[0], self.source_num_items, device=self.device
        )
        source_user_matrix.index_put_(
            (row_indices, col_indices), self.source_history_item_value[user].flatten()
        )
        source_user_e = self.source_user_linear(source_user_matrix)

        # Target-domain user representation
        target_user_matrix = self.construct_matrix(
            user,
            self.target_history_item_id,
            self.target_history_item_value,
            self.target_num_items,
        )
        target_user_e = self.target_user_linear(target_user_matrix)

        # Source-domain item representation
        col_indices = self.source_history_user_id[item].flatten().clone()
        col_indices[col_indices > self.target_num_users] -= (
            self.target_num_users - self.overlapped_num_users
        )
        row_indices = torch.arange(item.shape[0], device=self.device)
        row_indices = row_indices.repeat_interleave(
            self.source_history_user_id.shape[1], dim=0
        )
        source_item_matrix = torch.zeros(
            item.shape[0], self.source_num_users, device=self.device
        )
        source_item_matrix.index_put_(
            (row_indices, col_indices), self.source_history_user_value[item].flatten()
        )
        source_item_e = self.source_item_linear(source_item_matrix)

        # Target-domain item representation
        target_item_matrix = self.construct_matrix(
            item,
            self.target_history_user_id,
            self.target_history_user_value,
            self.target_num_users,
        )
        target_item_e = self.target_item_linear(target_item_matrix)

        user_valid, item_valid = self._domain_validity(user, item)

        user_e, item_e, _, _ = self._cross_domain_fusion(
            source_user_e,
            target_user_e,
            source_item_e,
            target_item_e,
            user_valid,
            item_valid,
        )

        if domain == 'source':
            user_h = self.source_user_fc_layers(user_e)
            item_h = self.source_item_fc_layers(item_e)
            output = self.source_sigmoid(torch.mul(user_h, item_h).sum(dim=1))
        else:
            user_h = self.target_user_fc_layers(user_e)
            item_h = self.target_item_fc_layers(item_e)
            output = self.target_sigmoid(torch.mul(user_h, item_h).sum(dim=1))

        return output

    # ------------------------------------------------------------------
    # Optional interpretability helper
    # ------------------------------------------------------------------
    def get_attention_weights(self, user, item):
        """Return source/target attention weights for NeuMF inputs.

        This is useful for analysis/visualization in the thesis. The returned
        order is [source, target] for both users and items.
        """
        if self.base_model != 'NeuMF':
            raise NotImplementedError(
                'get_attention_weights is currently implemented for NeuMF only.'
            )

        user_source_e = self.source_user_embedding(user)
        user_target_e = self.target_user_embedding(user)
        item_source_e = self.source_item_embedding(item)
        item_target_e = self.target_item_embedding(item)

        user_valid, item_valid = self._domain_validity(user, item)

        _, _, user_weights, item_weights = self._cross_domain_fusion(
            user_source_e,
            user_target_e,
            item_source_e,
            item_target_e,
            user_valid,
            item_valid,
        )

        return user_weights, item_weights

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        if self.base_model == 'NeuMF':
            target_output = self.neumf_forward(target_user, target_item, 'target')
        else:
            target_output = self.dmf_forward(target_user, target_item, 'target')

        return self.loss(target_output, target_label)

    def predict(self, interaction):
        """Target-domain prediction, matching the current RecBole-CDR setup."""
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]

        if self.base_model == 'NeuMF':
            return self.neumf_forward(user, item, 'target')
        return self.dmf_forward(user, item, 'target')
