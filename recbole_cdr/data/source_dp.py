"""Differentially private preprocessing for source-domain interactions.

The mechanism uses interaction-level add/remove adjacency.  Each item row of
the projection matrix is clipped to ``clip_norm``. Adding or removing one
binary ``(user, item)`` entry therefore changes the projected matrix by one
such row, so its L2 sensitivity is at most ``clip_norm``. Everything after the
Gaussian mechanism is post-processing.
"""

import math

import numpy as np
import torch
from scipy import sparse
from scipy.special import ndtr

from recbole.data.interaction import Interaction


def _gaussian_delta(epsilon, sigma_over_sensitivity):
    """Exact delta of the Gaussian mechanism at a given epsilon."""
    mu = 1.0 / sigma_over_sensitivity
    return ndtr(mu / 2.0 - epsilon / mu) - math.exp(epsilon) * ndtr(
        -mu / 2.0 - epsilon / mu
    )


def calibrate_gaussian_sigma(epsilon, delta, sensitivity):
    """Return the smallest sigma satisfying the exact Gaussian DP bound.

    This numerically inverts the privacy-loss expression for two Gaussians
    whose means are separated by ``sensitivity``.  Unlike the common
    ``sqrt(2 log(1.25/delta))/epsilon`` shortcut, it is valid for epsilon > 1.
    """
    if epsilon <= 0:
        raise ValueError('dp_epsilon must be positive.')
    if not 0 < delta < 1:
        raise ValueError('dp_delta must be strictly between 0 and 1.')
    if sensitivity < 0:
        raise ValueError('sensitivity must be non-negative.')
    if sensitivity == 0:
        return 0.0

    low, high = 0.0, 1.0
    while _gaussian_delta(epsilon, high) > delta:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _gaussian_delta(epsilon, middle) > delta:
            low = middle
        else:
            high = middle
    return high * sensitivity


def _sparse_projection(
    item_num, projection_dim, density, clip_norm, rng, eligible_item_ids=None
):
    """Construct a seeded sparse Achlioptas-style projection matrix."""
    if projection_dim <= 0:
        raise ValueError('dp_projection_dim must be positive.')
    if not 0 < density <= 1:
        raise ValueError('dp_projection_density must be in (0, 1].')
    if clip_norm <= 0:
        raise ValueError('dp_clip_norm must be positive.')

    # Sampling the number of entries per item avoids allocating an
    # item_num-by-projection_dim Bernoulli mask.
    rows, cols, values = [], [], []
    scale = 1.0 / math.sqrt(density * projection_dim)
    if eligible_item_ids is None:
        eligible_item_ids = range(1, item_num)
    for item_id in eligible_item_ids:
        if item_id == 0:  # ID zero is RecBole padding.
            continue
        count = rng.binomial(projection_dim, density)
        if count == 0:
            continue
        columns = rng.choice(projection_dim, size=count, replace=False)
        row_norm = scale * math.sqrt(count)
        row_scale = min(1.0, clip_norm / row_norm)
        rows.extend([item_id] * count)
        cols.extend(columns.tolist())
        signs = rng.choice(np.asarray([-scale, scale]), size=count) * row_scale
        values.extend(signs.tolist())
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(item_num, projection_dim), dtype=np.float64
    )


def privatize_source_dataset(dataset, user_num, item_num, config, logger):
    """Return a copy of ``dataset`` whose interactions are DP post-processing.

    The decoder retains at most ``dp_output_topk`` positive reconstructed
    item scores per user.  This public cap prevents a dense decoded dataset;
    it is not selected from the confidential interaction counts.
    """
    epsilon = float(config['dp_epsilon'])
    delta = float(config['dp_delta'])
    projection_dim = int(config['dp_projection_dim'])
    density = float(config['dp_projection_density'])
    clip_norm = float(config['dp_clip_norm'])
    output_topk = int(config['dp_output_topk'])
    seed = int(config['dp_seed'])
    if output_topk < 0:
        raise ValueError('dp_output_topk must be non-negative.')

    projection_rng = np.random.RandomState(seed)
    # A configured/public seed may safely define the projection, but must not
    # make DP noise predictable. default_rng() obtains fresh OS entropy.
    noise_rng = np.random.default_rng()
    users = dataset.inter_feat[dataset.uid_field].numpy().astype(np.int64, copy=False)
    items = dataset.inter_feat[dataset.iid_field].numpy().astype(np.int64, copy=False)
    before = len(users)

    # The joint remapping has target-only ID ranges.  Limit both mechanism and
    # decoder to the source vocabulary recorded in this dataset's metadata.
    source_user_ids = sorted(set(dataset.field2token_id[dataset.uid_field].values()) - {0})
    source_item_ids = sorted(set(dataset.field2token_id[dataset.iid_field].values()) - {0})

    # Duplicate events intentionally collapse to a binary implicit-feedback X.
    x = sparse.csr_matrix(
        (np.ones(before, dtype=np.float64), (users, items)),
        shape=(user_num, item_num),
    )
    x.data[:] = 1.0
    projection = _sparse_projection(
        item_num, projection_dim, density, clip_norm, projection_rng, source_item_ids
    )
    projected = (x @ projection).tocsr()
    # Under interaction-level add/remove adjacency, one X entry changes and
    # hence one projection row is added to/removed from one user's Z row.
    row_norms = np.sqrt(projection.multiply(projection).sum(axis=1)).A1
    sensitivity = float(row_norms.max())
    sigma = calibrate_gaussian_sigma(epsilon, delta, sensitivity)

    output_users, output_items = [], []
    max_items_per_user = min(output_topk, len(source_item_ids))
    source_item_ids = np.asarray(source_item_ids, dtype=np.int64)
    for user_id in source_user_ids:
        vector = projected.getrow(user_id).toarray().ravel()
        private_vector = vector + noise_rng.normal(0.0, sigma, size=projection_dim)

        # Decode one row at a time: memory is O(item_num + projection nnz), not
        # O(user_num * item_num).  Top-k and positivity depend only on DP output.
        scores = projection.dot(private_vector)
        candidates = source_item_ids[scores[source_item_ids] > 0.0]
        if max_items_per_user == 0:
            candidates = candidates[:0]
        elif len(candidates) > max_items_per_user:
            keep = np.argpartition(scores[candidates], -max_items_per_user)[-max_items_per_user:]
            candidates = candidates[keep]
        output_users.extend([user_id] * len(candidates))
        output_items.extend(candidates.tolist())

    private_interactions = Interaction({
        dataset.uid_field: torch.as_tensor(output_users, dtype=torch.int64),
        dataset.iid_field: torch.as_tensor(output_items, dtype=torch.int64),
    })
    private_dataset = dataset.copy(private_interactions)

    logger.info('Source DP enabled')
    logger.info('epsilon: %s', epsilon)
    logger.info('delta: %s', delta)
    logger.info('projection dimension: %s', projection_dim)
    logger.info('output top-k: %s', output_topk)
    logger.info('interaction-level L2 sensitivity: %.12g', sensitivity)
    logger.info('calculated noise sigma: %.12g', sigma)
    logger.info('number of source interactions before and after DP: %d -> %d', before, len(output_users))
    return private_dataset
