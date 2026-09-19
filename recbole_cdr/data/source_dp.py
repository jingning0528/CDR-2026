"""Differentially private preprocessing for source-domain interactions.

The mechanism uses interaction-level add/remove adjacency.  Each item row of
the projection matrix is clipped to ``clip_norm``. Adding or removing one
binary ``(user, item)`` entry therefore changes the projected matrix by one
such row, so its L2 sensitivity is at most ``clip_norm``. Everything after the
Gaussian mechanism is post-processing.
"""

import json
import math
from pathlib import Path

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


def _set_retention(reference, comparison):
    """Return the fraction of a nonempty reference set retained."""
    if not reference:
        return None
    return len(reference & comparison) / len(reference)


def _jaccard_similarity(left, right):
    """Return Jaccard similarity, or None when both sets are empty."""
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


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
    noise_multiplier = float(config['dp_noise_multiplier'])
    seed = int(config['dp_seed'])
    noise_seed = int(config['dp_noise_seed'])
    topk_output_dir = config['dp_topk_output_dir']
    if output_topk < 0:
        raise ValueError('dp_output_topk must be non-negative.')
    if noise_multiplier < 0:
        raise ValueError('dp_noise_multiplier must be non-negative.')

    projection_rng = np.random.RandomState(seed)
    # Keep projection and noise randomness independent so repeated overlap
    # experiments can vary either one without changing the other.
    noise_rng = np.random.default_rng(noise_seed)
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
    calibrated_sigma = calibrate_gaussian_sigma(epsilon, delta, sensitivity)
    sigma = calibrated_sigma * noise_multiplier
    if noise_multiplier < 1.0:
        logger.warning(
            'dp_noise_multiplier is %.12g; this preprocessing does not satisfy '
            'the configured (epsilon, delta) guarantee. Use 1.0 or greater for DP.',
            noise_multiplier,
        )

    output_users, output_items = [], []
    max_items_per_user = min(output_topk, len(source_item_ids))
    source_item_ids = np.asarray(source_item_ids, dtype=np.int64)
    original_by_user = {
        user_id: set(x.getrow(user_id).indices.tolist())
        for user_id in source_user_ids
    }
    baseline_topk_by_user = {}
    private_topk_by_user = {}
    dp_overlaps = []
    dp_jaccards = []
    reconstruction_overlaps = []
    for user_id in source_user_ids:
        vector = projected.getrow(user_id).toarray().ravel()
        private_vector = vector + noise_rng.normal(0.0, sigma, size=projection_dim)

        # Decode one row at a time: memory is O(item_num + projection nnz), not
        # O(user_num * item_num).  Top-k and positivity depend only on DP output.
        baseline_scores = projection.dot(vector)
        baseline_candidates = source_item_ids[baseline_scores[source_item_ids] > 0.0]
        if max_items_per_user == 0:
            baseline_candidates = baseline_candidates[:0]
        elif len(baseline_candidates) > max_items_per_user:
            keep = np.argpartition(
                baseline_scores[baseline_candidates], -max_items_per_user
            )[-max_items_per_user:]
            baseline_candidates = baseline_candidates[keep]

        scores = projection.dot(private_vector)
        candidates = source_item_ids[scores[source_item_ids] > 0.0]
        if max_items_per_user == 0:
            candidates = candidates[:0]
        elif len(candidates) > max_items_per_user:
            keep = np.argpartition(scores[candidates], -max_items_per_user)[-max_items_per_user:]
            candidates = candidates[keep]

        baseline_set = set(baseline_candidates.tolist())
        private_set = set(candidates.tolist())
        baseline_topk_by_user[user_id] = baseline_set
        private_topk_by_user[user_id] = private_set
        dp_overlap = _set_retention(baseline_set, private_set)
        if dp_overlap is not None:
            dp_overlaps.append(dp_overlap)
        dp_jaccard = _jaccard_similarity(baseline_set, private_set)
        if dp_jaccard is not None:
            dp_jaccards.append(dp_jaccard)
        reconstruction_overlap = _set_retention(
            original_by_user[user_id], baseline_set
        )
        if reconstruction_overlap is not None:
            reconstruction_overlaps.append(reconstruction_overlap)
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
    logger.info('calibrated noise sigma: %.12g', calibrated_sigma)
    logger.info('noise multiplier: %.12g', noise_multiplier)
    logger.info('noise seed: %d', noise_seed)
    logger.info('applied noise sigma: %.12g', sigma)
    if dp_overlaps:
        logger.info(
            'mean Top-%d retention (sigma=0 retained after DP): %.6f',
            max_items_per_user,
            float(np.mean(dp_overlaps)),
        )
    else:
        logger.info(
            'Top-K retention is undefined because all sigma=0 sets are empty.'
        )
    if dp_jaccards:
        logger.info(
            'mean Top-%d Jaccard similarity (sigma=0 vs DP): %.6f',
            max_items_per_user,
            float(np.mean(dp_jaccards)),
        )
    else:
        logger.info('Top-K Jaccard similarity is undefined for empty unions.')
    if reconstruction_overlaps:
        logger.info(
            'mean original-interaction retention in Top-%d sigma=0 reconstruction: %.6f',
            max_items_per_user,
            float(np.mean(reconstruction_overlaps)),
        )
    else:
        logger.info(
            'Original-interaction retention is undefined because all original sets are empty.'
        )
    logger.info('number of source interactions before and after DP: %d -> %d', before, len(output_users))

    if topk_output_dir:
        output_dir = Path(topk_output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            'topk-projection_seed_{}-noise_seed_{}-epsilon_{:g}-multiplier_{:g}.json'.format(
                seed, noise_seed, epsilon, noise_multiplier
            )
        )
        payload = {
            'projection_seed': seed,
            'noise_seed': noise_seed,
            'epsilon': epsilon,
            'delta': delta,
            'noise_multiplier': noise_multiplier,
            'calibrated_sigma': calibrated_sigma,
            'applied_sigma': sigma,
            'k': max_items_per_user,
            'mean_overlap_sigma0_vs_dp': (
                float(np.mean(dp_overlaps)) if dp_overlaps else None
            ),
            'mean_jaccard_sigma0_vs_dp': (
                float(np.mean(dp_jaccards)) if dp_jaccards else None
            ),
            'mean_overlap_original_vs_sigma0': (
                float(np.mean(reconstruction_overlaps))
                if reconstruction_overlaps else None
            ),
            'users': {
                str(user_id): {
                    'original': sorted(original_by_user[user_id]),
                    'sigma0_topk': sorted(baseline_topk_by_user[user_id]),
                    'dp_topk': sorted(private_topk_by_user[user_id]),
                    'overlap_sigma0_vs_dp': _set_retention(
                        baseline_topk_by_user[user_id],
                        private_topk_by_user[user_id],
                    ),
                    'jaccard_sigma0_vs_dp': _jaccard_similarity(
                        baseline_topk_by_user[user_id],
                        private_topk_by_user[user_id],
                    ),
                    'overlap_original_vs_sigma0': _set_retention(
                        original_by_user[user_id],
                        baseline_topk_by_user[user_id],
                    ),
                }
                for user_id in source_user_ids
            },
        }
        with output_path.open('w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
        logger.info('saved Top-K overlap data: %s', output_path)
    return private_dataset
