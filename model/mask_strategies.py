"""Training-task samplers. True in a mask means the column is hidden."""

from numbers import Integral

import torch


MASK_STRATEGIES = frozenset(
    {
        "bernoulli_all",
        "uniform_count_all",
        "prefix_next",
        "ordered_completion",
        "canonical_subset",
        "random_permutation_next",
    }
)
ORDERED_STRATEGIES = frozenset(
    {"prefix_next", "ordered_completion", "canonical_subset"}
)


def validate_order(order, num_vars):
    """Return a complete permutation of column indices, or reject it."""
    if order is None or isinstance(order, (str, bytes)):
        raise ValueError("A complete column-index order is required")
    try:
        result = list(order)
    except TypeError as exc:
        raise ValueError("Order must be a sequence of column indices") from exc
    if (
        len(result) != num_vars
        or any(isinstance(i, bool) or not isinstance(i, Integral) for i in result)
        or set(result) != set(range(num_vars))
    ):
        raise ValueError("Order must contain each column index exactly once")
    return [int(i) for i in result]


def _random_subset(batch_size, num_vars, counts, device, generator):
    """Uniformly choose a subset conditional on its row-wise cardinality."""
    permutations = torch.rand(
        batch_size, num_vars, device=device, generator=generator
    ).argsort(dim=1)
    selected = torch.zeros(batch_size, num_vars, dtype=torch.bool, device=device)
    source = (
        torch.arange(num_vars, device=device).expand(batch_size, -1)
        < counts[:, None]
    )
    selected.scatter_(1, permutations, source)
    return selected


def sample_training_masks(
    batch_size,
    num_vars,
    strategy="bernoulli_all",
    mask_prob=0.15,
    order=None,
    device="cpu",
    generator=None,
):
    """Return (input_mask, target_mask), each shaped [batch, num_vars].

    ``canonical_subset`` is a simple canonical-edge sampler, not MAC's
    query-frequency-weighted training algorithm. ``ordered_completion``
    samples a hidden set and targets its first unknown column in ``order``.
    ``mask_prob`` applies only to ``bernoulli_all``.
    """
    if strategy not in MASK_STRATEGIES:
        raise ValueError(f"Unknown mask strategy: {strategy}")
    if batch_size < 1 or num_vars < 1:
        raise ValueError("batch_size and num_vars must be positive")
    if strategy in ORDERED_STRATEGIES:
        order = validate_order(order, num_vars)
    elif order is not None:
        raise ValueError(f"{strategy} does not use a fixed training order")

    if strategy == "bernoulli_all":
        if not 0 <= mask_prob <= 1:
            raise ValueError("mask_prob must be in [0, 1]")
        # Preserve the legacy RNG call and its possible empty target rows.
        hidden = torch.rand(
            batch_size, num_vars, device=device, generator=generator
        ) < mask_prob
        return hidden, hidden

    if strategy == "random_permutation_next":
        permutation = torch.rand(
            batch_size, num_vars, device=device, generator=generator
        ).argsort(dim=1)
        positions = torch.empty_like(permutation)
        positions.scatter_(
            1, permutation,
            torch.arange(num_vars, device=device).expand(batch_size, -1),
        )
        step = torch.randint(num_vars, (batch_size,), device=device, generator=generator)
        return positions >= step[:, None], positions == step[:, None]

    if strategy == "prefix_next":
        indices = torch.tensor(order, device=device)
        ranks = torch.empty(num_vars, dtype=torch.long, device=device)
        ranks[indices] = torch.arange(num_vars, device=device)
        step = torch.randint(num_vars, (batch_size,), device=device, generator=generator)
        return ranks[None, :] >= step[:, None], ranks[None, :] == step[:, None]

    counts = torch.randint(
        1, num_vars + 1, (batch_size,), device=device, generator=generator
    )
    subset = _random_subset(batch_size, num_vars, counts, device, generator)
    if strategy == "uniform_count_all":
        return subset, subset

    indices = torch.tensor(order, device=device)
    ranks = torch.empty(num_vars, dtype=torch.long, device=device)
    ranks[indices] = torch.arange(num_vars, device=device)
    if strategy == "ordered_completion":
        target_index = ranks[None, :].expand(batch_size, -1).masked_fill(
            ~subset, num_vars
        ).argmin(dim=1)
        target = torch.zeros_like(subset)
        target.scatter_(1, target_index[:, None], True)
        return subset, target

    # The subset is the marginal query E. Its last canonical column is the
    # target; only the earlier members of E are visible in this task.
    target_index = ranks[None, :].expand(batch_size, -1).masked_fill(
        ~subset, -1
    ).argmax(dim=1)
    target = torch.zeros_like(subset)
    target.scatter_(1, target_index[:, None], True)
    return ~subset | target, target
