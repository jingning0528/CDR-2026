import numpy as np
import torch
import torch.nn as nn
import math

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, nhead, num_layers=1, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBlock, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(embedding_size)

    def forward(self, src, mask=None):
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src, mask)
        return output



class AttentionLayer(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)

    def forward(self, user_embedding, item_embedding):
        query = self.query(user_embedding)
        key = self.key(item_embedding)
        attention_scores = torch.sum(query * key, dim=-1)
        # One gate per example. A softmax over this one-dimensional batch
        # vector would mix unrelated users and make predictions batch-dependent.
        attention_weights = torch.sigmoid(attention_scores)
        weighted_value = self.value(item_embedding) * attention_weights.unsqueeze(-1)
        return weighted_value


class TransformerGPU(CrossDomainRecommender):
    r"""DTCDR with reciprocal user-item attention and a Transformer block.

    The name identifies the supplied experiment, but the model contains no
    device-specific code and runs on either CPU or GPU through RecBole's device
    configuration. Differential-privacy gradient clipping and noise injection
    from the original draft have intentionally been removed.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(TransformerGPU, self).__init__(config, dataset)


        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.base_model = config['base_model']
        self.alpha = config['alpha']
        assert self.base_model in ['NeuMF', 'DMF'], "based model {} is not supported! ".format(self.base_model)

        # Transformer related configuration
        self.num_heads = self._cfg(config, 'transformer_gpu_num_heads', 8)
        self.num_layers = self._cfg(config, 'transformer_gpu_num_layers', 1)
        self.dim_feedforward = self._cfg(
            config, 'transformer_gpu_dim_feedforward', 4 * self.embedding_size
        )
        self.transformer_dropout = self._cfg(
            config, 'transformer_gpu_dropout', self.dropout_prob
        )
        self.transformer_activation = self._cfg(
            config, 'transformer_gpu_activation', 'relu'
        )

        assert self.embedding_size % self.num_heads == 0, "Embedding size should be divisible by number of heads"

        # Transformer block
        self.transformer_block = TransformerBlock(
            self.embedding_size,
            self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation=self.transformer_activation
        )

        self.attention_layer = AttentionLayer(self.embedding_size)

        # define layers and loss
        if self.base_model == 'NeuMF':
            self.source_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
            self.source_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)

            self.target_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
            self.target_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
            self.source_mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
            self.source_mlp_layers.logger = None  # remove logger to use torch.save()
            self.source_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

            self.target_mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
            self.target_mlp_layers.logger = None  # remove logger to use torch.save()
            self.target_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        else:
            self.source_history_user_id, self.source_history_user_value, _ = dataset.history_user_matrix(
                domain='source')
            self.source_history_item_id, self.source_history_item_value, _ = dataset.history_item_matrix(
                domain='source')
            self.source_interaction_matrix = dataset.inter_matrix(form='csr', domain='source').astype(np.float32)
            self.source_history_user_id = self.source_history_user_id.to(self.device)
            self.source_history_user_value = self.source_history_user_value.to(self.device)
            self.source_history_item_id = self.source_history_item_id.to(self.device)
            self.source_history_item_value = self.source_history_item_value.to(self.device)

            self.target_history_user_id, self.target_history_user_value, _ = dataset.history_user_matrix(
                domain='target')
            self.target_history_item_id, self.target_history_item_value, _ = dataset.history_item_matrix(
                domain='target')
            self.target_interaction_matrix = dataset.inter_matrix(form='csr', domain='target').astype(np.float32)
            self.target_history_user_id = self.target_history_user_id.to(self.device)
            self.target_history_user_value = self.target_history_user_value.to(self.device)
            self.target_history_item_id = self.target_history_item_id.to(self.device)
            self.target_history_item_value = self.target_history_item_value.to(self.device)

            self.source_user_linear = nn.Linear(in_features=self.source_num_items, out_features=self.embedding_size,
                                                bias=False)
            self.source_item_linear = nn.Linear(in_features=self.source_num_users, out_features=self.embedding_size,
                                                bias=False)
            self.source_user_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)
            self.source_item_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)

            self.target_user_linear = nn.Linear(in_features=self.target_num_items, out_features=self.embedding_size,
                                                bias=False)
            self.target_item_linear = nn.Linear(in_features=self.target_num_users, out_features=self.embedding_size,
                                                bias=False)
            self.target_user_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)
            self.target_item_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)

        self.source_sigmoid = nn.Sigmoid()
        self.target_sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()
        self.apply(xavier_normal_initialization)

        # Initialization must happen before unavailable-domain entries are
        # marked, otherwise Xavier initialization overwrites the masks.
        if self.base_model == 'NeuMF':
            with torch.no_grad():
                self.target_user_embedding.weight[self.target_num_users:].fill_(-np.inf)
                self.target_item_embedding.weight[self.target_num_items:].fill_(-np.inf)
                self.source_user_embedding.weight[
                    self.overlapped_num_users:self.target_num_users
                ].fill_(-np.inf)
                self.source_item_embedding.weight[
                    self.overlapped_num_items:self.target_num_items
                ].fill_(-np.inf)

    @staticmethod
    def _cfg(config, key, default):
        try:
            value = config[key]
        except Exception:
            return default
        return default if value is None else value

    def forward(self, user, item, domain='target'):
        if self.base_model == 'NeuMF':
            return self.neumf_forward(user, item, domain)
        return self.dmf_forward(user, item, domain)

    def neumf_forward(self, user, item, domain='source'):
        user_source_e = self.source_user_embedding(user)
        user_target_e = self.target_user_embedding(user)
        user_e = torch.maximum(user_source_e, user_target_e)

        item_source_e = self.source_item_embedding(item)
        item_target_e = self.target_item_embedding(item)
        item_e = torch.maximum(item_source_e, item_target_e)

        # Reciprocal attention is computed from the same original pair.
        attended_user_e = self.attention_layer(user_e, item_e)
        attended_item_e = self.attention_layer(item_e, user_e)

        # [sequence=2, batch, embedding]. Unlike transforming each single
        # token independently, this lets user and item interact in self-attention.
        pair = self.transformer_block(
            torch.stack([attended_user_e, attended_item_e], dim=0)
        )
        user_e, item_e = pair[0], pair[1]

        if domain == 'source':
            output = self.source_sigmoid(
                self.source_predict_layer(self.source_mlp_layers(torch.cat((user_e, item_e), -1))))
        else:
            output = self.target_sigmoid(
                self.target_predict_layer(self.target_mlp_layers(torch.cat((user_e, item_e), -1))))
        return output.squeeze(-1)

    def construct_matrix(self, input_tensor, history_id_matrix, history_value_matrix, length):
        col_indices = history_id_matrix[input_tensor].flatten()
        row_indices = torch.arange(input_tensor.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(history_id_matrix.shape[1], dim=0)
        matrix_01 = torch.zeros(1).to(self.device).repeat(input_tensor.shape[0], length)
        matrix_01.index_put_((row_indices, col_indices), history_value_matrix[input_tensor].flatten())
        return matrix_01

    def dmf_forward(self, user, item, domain='source'):
        col_indices = self.source_history_item_id[user].flatten()
        col_indices[col_indices > self.target_num_items] = col_indices[col_indices > self.target_num_items] - (
                    self.target_num_items - self.overlapped_num_items)
        row_indices = torch.arange(user.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(self.source_history_item_id.shape[1], dim=0)
        source_user_matrix = torch.zeros(1).to(self.device).repeat(user.shape[0], self.source_num_items)
        source_user_matrix.index_put_((row_indices, col_indices), self.source_history_item_value[user].flatten())
        source_user_e = self.source_user_linear(source_user_matrix)

        target_user_matrix = self.construct_matrix(user, self.target_history_item_id, self.target_history_item_value,
                                                   self.target_num_items)
        target_user_e = self.target_user_linear(target_user_matrix)

        user_e = torch.maximum(source_user_e, target_user_e)

        col_indices = self.source_history_user_id[item].flatten()
        col_indices[col_indices > self.target_num_users] = col_indices[col_indices > self.target_num_users] - (
                self.target_num_users - self.overlapped_num_users)
        row_indices = torch.arange(user.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(self.source_history_user_id.shape[1], dim=0)
        source_item_matrix = torch.zeros(1).to(self.device).repeat(item.shape[0], self.source_num_users)
        source_item_matrix.index_put_((row_indices, col_indices), self.source_history_user_value[user].flatten())
        source_item_e = self.source_item_linear(source_item_matrix)

        target_item_matrix = self.construct_matrix(item, self.target_history_user_id, self.target_history_user_value,
                                                   self.target_num_users)
        target_item_e = self.target_item_linear(target_item_matrix)

        item_e = torch.maximum(source_item_e, target_item_e)

        attended_user_e = self.attention_layer(user_e, item_e)
        attended_item_e = self.attention_layer(item_e, user_e)
        pair = self.transformer_block(
            torch.stack([attended_user_e, attended_item_e], dim=0)
        )
        user_e, item_e = pair[0], pair[1]

        if domain == 'source':
            user_e = self.source_user_fc_layers(user_e)
            item_e = self.source_item_fc_layers(item_e)
            output = torch.mul(user_e, item_e).sum(dim=1)
            output = self.source_sigmoid(output)
        else:
            user_e = self.target_user_fc_layers(user_e)
            item_e = self.target_item_fc_layers(item_e)
            output = torch.mul(user_e, item_e).sum(dim=1)
            output = self.target_sigmoid(output)

        return output

    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        if self.base_model == 'NeuMF':
            source_output = self.neumf_forward(source_user, source_item, 'source')
            target_output = self.neumf_forward(target_user, target_item, 'target')

            loss_s = self.loss(source_output, source_label)
            loss_t = self.loss(target_output, target_label)

            return loss_s * self.alpha + loss_t * (1 - self.alpha)
        else:
            source_output = self.dmf_forward(source_user, source_item, 'source')
            target_output = self.dmf_forward(target_user, target_item, 'target')

            loss_s = self.loss(source_output, source_label)
            loss_t = self.loss(target_output, target_label)

            return loss_s * self.alpha + loss_t * (1 - self.alpha)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        if self.base_model == 'NeuMF':
            output = self.neumf_forward(user, item, 'target')
            return output
        else:
            output = self.dmf_forward(user, item, 'target')
            return output
