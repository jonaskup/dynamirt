"""Shared posterior replay for finite-category response models."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
import xarray as xr


def _replay_category_log_probs(
    model: Callable, posterior: xr.Dataset, rng_key: jax.Array,
) -> tuple[jax.Array, list[str]]:
    """Evaluate category log probabilities using joint posterior draws.

    ``model`` is a zero-argument callable configured to sample an unobserved
    finite-category response site ``Y``. Only its stochastic parameter sites
    are substituted; deterministic quantities are recomputed. Extra posterior
    variables (including parameters absent from this model) are ignored.

    Returns an array shaped ``(*sample_shape, row, item, category)`` and the
    corresponding sample dimension names, ordered as chain/draw or sample.
    The caller owns the row design, coordinate labels, and category counts.
    """
    trace = handlers.trace(handlers.seed(model, rng_key)).get_trace()
    sites = [name for name, site in trace.items()
             if site["type"] == "sample" and name != "Y"]
    missing = set(sites) - set(posterior.data_vars)
    if missing:
        raise ValueError(f"Posterior is missing model parameters: {sorted(missing)}")

    sample_dims = [d for d in ("chain", "draw", "sample") if d in posterior.dims]
    sample_shape = tuple(posterior.sizes[d] for d in sample_dims)
    n_draws = int(np.prod(sample_shape))
    draws = {}
    for name in sites:
        values = posterior[name].transpose(*sample_dims, ...).values
        draws[name] = jnp.asarray(values.reshape((n_draws,) + values.shape[len(sample_dims):]))

    def log_probs_for_draw(draw, key):
        replay = handlers.substitute(model, draw)
        response_dist = handlers.trace(handlers.seed(replay, key)).get_trace()["Y"]["fn"]
        support = response_dist.enumerate_support(expand=False)
        return jnp.moveaxis(response_dist.log_prob(support), 0, -1)

    log_probs = jax.vmap(log_probs_for_draw)(
        draws, jax.random.split(rng_key, n_draws)
    )
    return log_probs.reshape(*sample_shape, *log_probs.shape[1:]), sample_dims
