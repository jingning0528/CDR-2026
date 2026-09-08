import torch

from recbole.data.interaction import Interaction
from recbole.utils import EvaluatorType

from recbole_cdr.data.dataloader import _cat_interactions_recbole_1_0
from recbole_cdr.trainer.trainer import CrossDomainTrainer, DCDCSRTrainer


def test_sampled_eval_concat_uses_raw_interaction_tensors():
    # A non-string column key triggers RecBole 1.0's row-selection branch.
    field = Ellipsis
    interactions = [
        Interaction({field: torch.tensor([1, 2])}),
        Interaction({field: torch.tensor([3, 4])}),
    ]

    # RecBole 1.0 interprets this string-like field as a row selector.
    assert isinstance(interactions[0][field], Interaction)

    result = _cat_interactions_recbole_1_0(interactions)
    assert torch.equal(
        result.interaction[field], torch.tensor([1, 2, 3, 4])
    )


def test_sampled_eval_uses_target_domain_item_field():
    class Model:
        TARGET_ITEM_ID = 'target_item_id'

        @staticmethod
        def predict(interaction):
            return interaction['prediction']

    trainer = CrossDomainTrainer.__new__(CrossDomainTrainer)
    trainer.model = Model()
    trainer.device = torch.device('cpu')
    trainer.test_batch_size = 1024
    trainer.tot_item_num = 8
    trainer.config = {
        'eval_type': EvaluatorType.RANKING,
        # This is the value that breaks RecBole 1.0.1's generic trainer.
        'ITEM_ID_FIELD': None,
    }

    interaction = Interaction({
        'target_item_id': torch.tensor([2, 4, 3]),
        'prediction': torch.tensor([0.8, 0.2, 0.6]),
    })
    batched_data = (
        interaction,
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([2, 3]),
    )

    _, scores, _, _ = trainer._neg_sample_batch_eval(batched_data)

    assert scores.shape == (2, 8)
    assert scores[0, 2] == torch.tensor(0.8)
    assert scores[0, 4] == torch.tensor(0.2)
    assert scores[1, 3] == torch.tensor(0.6)


def test_all_cross_domain_trainers_support_sampled_evaluation():
    assert CrossDomainTrainer._neg_sample_batch_eval is \
        DCDCSRTrainer._neg_sample_batch_eval
