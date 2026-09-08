import torch

from recbole.data.interaction import Interaction

from recbole_cdr.data.dataloader import _cat_interactions_recbole_1_0


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
