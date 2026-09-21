"""Target-domain NeuMF baseline for cross-domain experiments."""

import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.layers import MLPLayers
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class NeuMF(CrossDomainRecommender):
    """Train a standard NeuMF model using target-domain interactions only."""

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        self.user_mf_embedding = nn.Embedding(
            self.total_num_users, self.mf_embedding_size
        )
        self.item_mf_embedding = nn.Embedding(
            self.total_num_items, self.mf_embedding_size
        )
        self.user_mlp_embedding = nn.Embedding(
            self.total_num_users, self.mlp_embedding_size
        )
        self.item_mlp_embedding = nn.Embedding(
            self.total_num_items, self.mlp_embedding_size
        )
        self.mlp_layers = MLPLayers(
            [2 * self.mlp_embedding_size] + self.mlp_hidden_size,
            self.dropout_prob,
        )
        self.predict_layer = nn.Linear(
            self.mf_embedding_size + self.mlp_hidden_size[-1], 1
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        mf_output = self.user_mf_embedding(user) * self.item_mf_embedding(item)
        mlp_input = torch.cat(
            (self.user_mlp_embedding(user), self.item_mlp_embedding(item)), dim=-1
        )
        mlp_output = self.mlp_layers(mlp_input)
        return self.sigmoid(
            self.predict_layer(torch.cat((mf_output, mlp_output), dim=-1))
        ).squeeze(-1)

    def calculate_loss(self, interaction):
        prediction = self.forward(
            interaction[self.TARGET_USER_ID], interaction[self.TARGET_ITEM_ID]
        )
        return self.loss(prediction, interaction[self.TARGET_LABEL])

    def predict(self, interaction):
        return self.forward(
            interaction[self.TARGET_USER_ID], interaction[self.TARGET_ITEM_ID]
        )
