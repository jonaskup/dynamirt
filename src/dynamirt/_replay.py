"""Shared posterior replay for response-distribution statistics."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions import Distribution
import xarray as xr


def _replay_response_statistic(
    model: Callable, posterior: xr.Dataset, rng_key: jax.Array,
    extract: Callable[[Distribution], jax.Array],
) -> tuple[jax.Array, list[str]]:
    """Extract a response statistic using joint posterior draws.

    ``model`` is a zero-argument callable configured to sample an unobserved
    response site ``Y``. Only its stochastic parameter sites
    are substituted; deterministic quantities are recomputed. Extra posterior
    variables (including parameters absent from this model) are ignored.

    ``extract`` maps the replayed response distribution to an array with a
    fixed shape across draws. Returns ``(*sample_shape, *statistic_shape)`` and the
    corresponding sample dimension names, ordered as chain/draw or sample.
    The caller owns the row design and coordinate labels.
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

    def statistic_for_draw(draw, key):
        replay = handlers.substitute(model, draw)
        response_dist = handlers.trace(handlers.seed(replay, key)).get_trace()["Y"]["fn"]
        return extract(response_dist)

    values = jax.vmap(statistic_for_draw)(
        draws, jax.random.split(rng_key, n_draws)
    )
    return values.reshape(*sample_shape, *values.shape[1:]), sample_dims


def _replay_category_log_probs(
    model: Callable, posterior: xr.Dataset, rng_key: jax.Array,
) -> tuple[jax.Array, list[str]]:
    """Evaluate finite-category log probabilities using joint posterior draws.

    The zero-argument model must expose an unobserved response site ``Y``
    whose distribution supports enumeration. Returns an array shaped
    ``(*sample_shape, row, item, category)`` and sample dimension names.
    The caller owns the row design, coordinate labels, and category counts.
    """
    def extract(response_dist):
        support = response_dist.enumerate_support(expand=False)
        return jnp.moveaxis(response_dist.log_prob(support), 0, -1)

    return _replay_response_statistic(model, posterior, rng_key, extract)


def _replay_conditional_means(
    model: Callable, posterior: xr.Dataset, rng_key: jax.Array,
) -> tuple[jax.Array, list[str]]:
    """Evaluate conditional response means using joint posterior draws.

    The zero-argument model must expose an unobserved response site ``Y``
    whose distribution implements ``mean``. Returns an array shaped
    ``(*sample_shape, row, item)`` and sample dimension names for a response
    matrix. These are distribution means, not sampled future responses;
    undefined means and errors from the distribution are propagated.
    """
    return _replay_response_statistic(
        model, posterior, rng_key, lambda response_dist: response_dist.mean,
    )
