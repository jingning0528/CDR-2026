"""Target-only source-token ablation for TransformerDTCDR."""

import torch

from recbole_cdr.model.cross_domain_recommender.transformerdtcdr import (
    TransformerDTCDR,
)


class TransformerDTCDRNoSource(TransformerDTCDR):
    """TransformerDTCDR ablation with no source tokens or source loss.

    The target user and target item still pass through the same Transformer and
    target NeuMF prediction head as TransformerDTCDR. Source user/item tokens
    are always masked from self-attention, and training consumes target-domain
    batches only. This isolates source-information transfer while retaining the
    target-side architecture.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.base_model != 'NeuMF':
            raise ValueError(
                'TransformerDTCDRNoSource supports base_model=NeuMF only. '
                'The DMF representation is constructed from interaction-history '
                'matrices and is not a clean no-source-information ablation.'
            )

    def _id_validity(self, user, item):
        """Mask uS and iS; expose only valid target user/item tokens."""
        target_user_valid = user < self.target_num_users
        target_item_valid = item < self.target_num_items
        masked = torch.zeros_like(target_user_valid, dtype=torch.bool)
        return torch.stack(
            [masked, target_user_valid, masked, target_item_valid], dim=1
        )

    def calculate_loss(self, interaction):
        """Optimize only the target objective from a target-only batch."""
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]
        target_output = self.neumf_forward(target_user, target_item, 'target')
        return self.loss(target_output, target_label)

