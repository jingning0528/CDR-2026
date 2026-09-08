import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class TransformerDTCDR(CrossDomainRecommender):
    r"""DTCDR with Transformer-based cross-domain representation interaction.

    The Transformer does NOT model a temporal sequence.  Instead, each
    user-item pair is represented by four semantic tokens:

        [source-user, target-user, source-item, target-item]

    Multi-head self-attention therefore learns interactions among the available
    source/target and user/item representations.  Tokens that do not exist in a
    domain are masked out and never passed to the Transformer as -inf values.

    Training keeps the original DTCDR multi-task objective:

        L = alpha * L_source + (1 - alpha) * L_target

    Prediction remains target-domain by default, matching RecBole-CDR's usual
    evaluation setup.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(TransformerDTCDR, self).__init__(config, dataset)

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

        # Optional Transformer settings.  Defaults keep the model small because
        # the token set has length 4, not hundreds of sequential positions.
        self.num_heads = self._cfg(config, 'transformer_num_heads', 4)
        self.num_layers = self._cfg(config, 'transformer_num_layers', 1)
        self.dim_feedforward = self._cfg(
            config, 'transformer_dim_feedforward', 4 * self.embedding_size
        )
        self.transformer_dropout = self._cfg(
            config, 'transformer_dropout', self.dropout_prob
        )

        if self.embedding_size % self.num_heads != 0:
            raise ValueError(
                'embedding_size ({}) must be divisible by transformer_num_heads ({})'.format(
                    self.embedding_size, self.num_heads
                )
            )

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

        # ------------------------------------------------------------------
        # Transformer cross-domain interaction block
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation='relu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Learned semantic type information is more suitable than sinusoidal
        # positional encoding because these are NOT temporal positions.
        self.domain_embedding = nn.Embedding(2, self.embedding_size)  # source/target
        self.entity_embedding = nn.Embedding(2, self.embedding_size)  # user/item

        self.source_sigmoid = nn.Sigmoid()
        self.target_sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # Initialize first, then restore unavailable-domain markers.  Doing the
        # masking before self.apply(...) could let initialization overwrite it.
        self.apply(xavier_normal_initialization)

        if self.base_model == 'NeuMF':
            self._apply_domain_embedding_masks()

    @staticmethod
    def _cfg(config, key, default):
        """Read an optional RecBole config value without requiring it."""
        try:
            value = config[key]
        except Exception:
            return default
        return default if value is None else value

    def _apply_domain_embedding_masks(self):
        """Mark IDs that do not exist in each domain with -inf.

        The -inf values are used only to identify invalid domain-specific
        embeddings.  They are replaced by zeros before entering the Transformer.
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

    def _id_validity(self, user, item):
        """Return validity masks for [uS, uT, iS, iT]."""
        user_source_valid = (
            (user < self.overlapped_num_users) | (user >= self.target_num_users)
        )
        user_target_valid = user < self.target_num_users

        item_source_valid = (
            (item < self.overlapped_num_items) | (item >= self.target_num_items)
        )
        item_target_valid = item < self.target_num_items

        return torch.stack(
            [
                user_source_valid,
                user_target_valid,
                item_source_valid,
                item_target_valid,
            ],
            dim=1,
        )

    def _cross_domain_transformer(
        self,
        user_source_e,
        user_target_e,
        item_source_e,
        item_target_e,
        valid_tokens,
    ):
        """Run self-attention over four semantic cross-domain tokens.

        Args:
            *_e: [batch_size, embedding_size]
            valid_tokens: [batch_size, 4], True where the token exists.

        Returns:
            Four contextualized representations in the same token order.
        """
        tokens = torch.stack(
            [user_source_e, user_target_e, item_source_e, item_target_e], dim=1
        )  # [B, 4, D]

        # Never feed -inf into Transformer attention.
        tokens = torch.where(
            valid_tokens.unsqueeze(-1), tokens, torch.zeros_like(tokens)
        )

        # Token semantics:
        #   0 = source user, 1 = target user,
        #   2 = source item, 3 = target item.
        domain_ids = torch.tensor([0, 1, 0, 1], device=tokens.device)
        entity_ids = torch.tensor([0, 0, 1, 1], device=tokens.device)

        tokens = (
            tokens
            + self.domain_embedding(domain_ids).unsqueeze(0)
            + self.entity_embedding(entity_ids).unsqueeze(0)
        )

        # True means "ignore this token" for src_key_padding_mask.
        padding_mask = ~valid_tokens

        # PyTorch 1.7-compatible TransformerEncoder expects [S, B, D].
        contextualized = self.transformer(
            tokens.transpose(0, 1), src_key_padding_mask=padding_mask
        ).transpose(0, 1)

        return (
            contextualized[:, 0, :],
            contextualized[:, 1, :],
            contextualized[:, 2, :],
            contextualized[:, 3, :],
        )

    def forward(self, user, item, domain='target'):
        if self.base_model == 'NeuMF':
            return self.neumf_forward(user, item, domain)
        return self.dmf_forward(user, item, domain)

    # ----------------------------------------------------------------------
    # NeuMF
    # ----------------------------------------------------------------------
    def neumf_forward(self, user, item, domain='source'):
        user_source_e = self.source_user_embedding(user)
        user_target_e = self.target_user_embedding(user)
        item_source_e = self.source_item_embedding(item)
        item_target_e = self.target_item_embedding(item)

        # Determine which domain-specific representations actually exist.
        # Using ID ranges avoids relying on -inf after subsequent updates.
        valid_tokens = self._id_validity(user, item)

        user_source_h, user_target_h, item_source_h, item_target_h = (
            self._cross_domain_transformer(
                user_source_e,
                user_target_e,
                item_source_e,
                item_target_e,
                valid_tokens,
            )
        )

        # Keep domain-specific prediction heads.  The selected representation
        # has already interacted with every valid cross-domain token through
        # Transformer self-attention.
        if domain == 'source':
            user_h = user_source_h
            item_h = item_source_h
            output = self.source_sigmoid(
                self.source_predict_layer(
                    self.source_mlp_layers(torch.cat((user_h, item_h), dim=-1))
                )
            )
        else:
            user_h = user_target_h
            item_h = item_target_h
            output = self.target_sigmoid(
                self.target_predict_layer(
                    self.target_mlp_layers(torch.cat((user_h, item_h), dim=-1))
                )
            )

        return output.squeeze(-1)

    # ----------------------------------------------------------------------
    # DMF
    # ----------------------------------------------------------------------
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

        valid_tokens = self._id_validity(user, item)

        user_source_h, user_target_h, item_source_h, item_target_h = (
            self._cross_domain_transformer(
                source_user_e,
                target_user_e,
                source_item_e,
                target_item_e,
                valid_tokens,
            )
        )

        if domain == 'source':
            user_h = self.source_user_fc_layers(user_source_h)
            item_h = self.source_item_fc_layers(item_source_h)
            output = self.source_sigmoid(torch.mul(user_h, item_h).sum(dim=1))
        else:
            user_h = self.target_user_fc_layers(user_target_h)
            item_h = self.target_item_fc_layers(item_target_h)
            output = self.target_sigmoid(torch.mul(user_h, item_h).sum(dim=1))

        return output

    # ----------------------------------------------------------------------
    # Training / evaluation
    # ----------------------------------------------------------------------
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
