"""Shapley value computation for span-level aleatoric uncertainty attribution.

Given a cache of H(Y|X, C_S) for all subsets S ⊆ N, computes the Shapley value
φ_k for each span k, where:

    φ_k = Σ_{S ⊆ N\{k}} w(|S|) · [H(Y|X, C_S) - H(Y|X, C_{S∪{k}})]

    w(s) = s! (n-s-1)! / n!

Key property (efficiency):  Σ_k φ_k = H(Y|X) - H(Y|X, C_N) = I(Y; C|X)
"""

import logging
from itertools import combinations
from math import factorial
from typing import Dict, FrozenSet, Hashable, Set, TypeVar

logger = logging.getLogger(__name__)
Player = TypeVar("Player", bound=Hashable)


def all_subsets(s: Set[Player]) -> list[FrozenSet[Player]]:
    """Return all subsets of *s*, including the empty set.

    Args:
        s: A set of player identifiers.

    Returns:
        List of frozensets, ordered by size (empty set first).
    """
    items = sorted(s, key=str)
    result: list[FrozenSet[Player]] = []
    for r in range(len(items) + 1):
        'combinations(iterable, r) returns all r-length combinations from iterable'
        for combo in combinations(items, r):
            'frozenset is an immutable set'
            result.append(frozenset(combo))
    return result


def compute_shapley_values(
    H_cache: Dict[FrozenSet[Player], float],
    N: Set[Player],
) -> Dict[Player, float]:
    """Compute Shapley values from cached entropy measurements.

    Each H_cache[S] stores H(Y|X, C_S) — the expected semantic entropy
    when the spans in S are clarified and all others are kept as-is.

    The marginal contribution of span k given coalition S is:

        Δ_k(S) = H(Y|X, C_S) - H(Y|X, C_{S∪{k}})
               = I(Y; C_k | X, C_S)

    which is the conditional MI of C_k given that S is already clarified.

    Args:
        H_cache: Mapping from frozenset(S) → H(Y|X, C_S).
                 Must contain entries for ALL 2^n subsets of N.
        N: Full set of span indices {0, 1, ..., n-1}.

    Returns:
        Mapping from span index k → φ_k.
    """
    n = len(N)
    phi: Dict[Player, float] = {}

    for k in sorted(N, key=str):
        others = N - {k}
        phi_k = 0.0

        for S in all_subsets(others):
            s = len(S)
            weight = factorial(s) * factorial(n - s - 1) / factorial(n)
            # Δ_k(S) = H_cache[S] - H_cache[S ∪ {k}]
            S_with_k = S | {k}
            marginal = H_cache[S] - H_cache[S_with_k]
            phi_k += weight * marginal

        phi[k] = phi_k
        logger.info("Shapley: φ_%s = %.4f nats", k, phi_k)

    # Sanity check: efficiency
    total_aleatoric = H_cache[frozenset()] - H_cache[frozenset(N)]
    phi_sum = sum(phi.values())
    residual = abs(phi_sum - total_aleatoric)

    logger.info(
        "Shapley: Σφ_k=%.4f | I(Y;C|X)=%.4f | residual=%.6f",
        phi_sum, total_aleatoric, residual,
    )
    if residual > 1e-4:
        logger.warning(
            "Shapley: efficiency check failed — residual=%.6f > 1e-4", residual,
        )

    return phi
