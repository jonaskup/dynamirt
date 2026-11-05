"""Response scores on the fitted observation design."""

from numbers import Integral

import jax
from jax.scipy.special import logsumexp
import numpy as np
import xarray as xr

from ._replay import _replay_category_log_probs
from .fit import FitResult
from .gllvm.model import _validate_responses


def log_loss(
    result: FitResult,
    responses=None,
    *,
    posterior: xr.Dataset | xr.DataTree | None = None,
    num_samples: int | None = None,
    batch_size: int = 5,
) -> xr.DataArray:
    """Negative log posterior-predictive probability of each supplied answer.

    Probabilities are averaged over posterior draws before taking the negative
    logarithm. Lower is better.

    Args:
        result: Fitted dynamirt model.
        responses: Numeric array in the same shape and positional row/item
            order as result.responses. NaN cells are not scored. Defaults to
            result.responses, giving training scores. For held-out scoring,
            mask answers before fitting and supply those answers here, with
            NaN elsewhere.
        posterior: Optional posterior Dataset or DataTree node retaining all
            stochastic parameters in model order. Defaults to
            result.to_idata()["posterior"]. Reuse this argument to score
            multiple response masks using the same draws.
        num_samples: SVI draw count when posterior is omitted (default 500).
            Leave None for MCMC or an explicit posterior.
        batch_size: Maximum total posterior draws replayed together, across
            all chains (default 5). All selected draws are used. Reduce to 1
            if memory is limited; only batch probabilities and their running
            sum are retained. This does not batch observation rows.

    Returns:
        DataArray named log_loss with dimensions (obs, item), observation
        covariates as auxiliary coordinates, and NaN at unscored cells.
        Use .mean() for an overall cell-weighted score or .mean("obs") for
        per-item scores.
    """
    values, _, log_mean_p, coords = _scoring_inputs(
        result, responses, posterior, num_samples, batch_size
    )
    observed = ~np.isnan(values)
    categories = np.where(observed, values, 0).astype(int)
    losses = -np.take_along_axis(log_mean_p, categories[..., None], axis=-1)[..., 0]
    losses = np.where(observed, losses, np.nan)
    return xr.DataArray(losses, dims=("obs", "item"), coords=coords, name="log_loss")


def ranked_probability_score(
    result: FitResult,
    responses=None,
    *,
    posterior: xr.Dataset | xr.DataTree | None = None,
    num_samples: int | None = None,
    batch_size: int = 5,
) -> xr.DataArray:
    """Normalized ranked probability score (RPS) of each supplied answer.

    Average category probabilities over posterior draws, then compare their
    cumulative probabilities with the observed category. For an item with K
    categories, the score is the sum of squared differences at thresholds
    0,...,K-2, divided by K-1. Padded categories are excluded. Lower is better;
    scores range from zero to one. For binary items this is the Brier score.

    Args:
        result: Fitted dynamirt model.
        responses: Numeric array in the same shape and positional row/item
            order as result.responses. NaN cells are not scored. Defaults to
            result.responses, giving training scores. For held-out scoring,
            mask answers before fitting and supply those answers here, with
            NaN elsewhere.
        posterior: Optional posterior Dataset or DataTree node retaining all
            stochastic parameters in model order. Defaults to
            result.to_idata()["posterior"]. Reuse this argument to score
            multiple response masks using the same draws.
        num_samples: SVI draw count when posterior is omitted (default 500).
            Leave None for MCMC or an explicit posterior.
        batch_size: Maximum total posterior draws replayed together, across
            all chains (default 5). All selected draws are used. Reduce to 1
            if memory is limited; only batch probabilities and their running
            sum are retained. This does not batch observation rows.

    Returns:
        DataArray named rps with dimensions (obs, item), observation covariates
        as auxiliary coordinates, and NaN at unscored cells. Use .mean() for
        an overall cell-weighted score or .mean("obs") for per-item scores.
    """
    values, counts, log_mean_p, coords = _scoring_inputs(
        result, responses, posterior, num_samples, batch_size
    )
    cumulative = np.cumsum(np.exp(log_mean_p), axis=-1)[..., :-1]
    thresholds = np.arange(cumulative.shape[-1])
    valid = thresholds < counts[:, None] - 1
    observed_cumulative = values[..., None] <= thresholds
    squared_error = (cumulative - observed_cumulative) ** 2
    scores = np.where(valid, squared_error, 0).sum(axis=-1) / (counts - 1)
    scores = np.where(~np.isnan(values), scores, np.nan)
    return xr.DataArray(scores, dims=("obs", "item"), coords=coords, name="rps")


def _scoring_inputs(result, responses, posterior, num_samples, batch_size):
    """Validate answers and replay posterior-averaged probabilities and labels."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    values = np.asarray(result.responses if responses is None else responses)
    if values.shape != np.shape(result.responses):
        raise ValueError("responses must have the same shape as the fitted responses")
    n_obs, n_items = values.shape
    counts = np.broadcast_to(result.model._dynamirt_n_cat, (n_items,))
    _validate_responses(values, discrete=True, n_cat=counts)

    if posterior is None:
        posterior = result.to_idata(num_samples=num_samples)["posterior"]
    elif num_samples is not None:
        raise ValueError("num_samples applies only when posterior is omitted")
    if isinstance(posterior, xr.DataTree):
        posterior = posterior.to_dataset()

    def model():
        result.model(
            responses=None, covariates=result.covariates,
            n_obs=n_obs, n_var=n_items,
        )

    sample_dims = [d for d in ("chain", "draw", "sample") if d in posterior.dims]
    sample_shape = tuple(posterior.sizes[d] for d in sample_dims)
    n_draws = int(np.prod(sample_shape))
    if not sample_dims or n_draws == 0:
        raise ValueError("posterior must contain nonempty sample or chain/draw dimensions")

    # Select on the host before replay, so only one batch's parameters are
    # transferred to the device. Vectorized indexing combines chain/draw into
    # a single sample dimension without stacking the entire posterior.
    log_total = None
    for start in range(0, n_draws, batch_size):
        indices = np.unravel_index(np.arange(start, min(start + batch_size, n_draws)), sample_shape)
        batch = posterior.isel({
            dim: xr.DataArray(index, dims="sample")
            for dim, index in zip(sample_dims, indices)
        })
        log_p, batch_dims = _replay_category_log_probs(model, batch, jax.random.key(0))
        # Transfer only the reduced sum. np.asarray waits for device work to
        # finish before the next batch, preventing queued batches from piling up.
        batch_total = np.asarray(logsumexp(log_p, axis=tuple(range(len(batch_dims))))).astype(np.float64)
        log_total = batch_total if log_total is None else np.logaddexp(log_total, batch_total)
        del log_p, batch_total, batch
    # Weight draws equally, including a shorter final batch. Averaging scores
    # or equally averaging batch means would give a different answer.
    log_mean_p = log_total - np.log(n_draws)
    coords = {
        dim: np.asarray(posterior.coords.get(dim, result.coords.get(dim, np.arange(size))))
        for dim, size in (("obs", n_obs), ("item", n_items))
    }
    for name, covariate in result.covariates.items():
        if np.shape(covariate) == (n_obs,):
            if name in {"obs", "item", "log_loss", "rps"}:
                raise ValueError(f"Covariate name conflicts with score coordinates: {name!r}")
            coords[name] = ("obs", np.asarray(covariate))
    return values, counts, log_mean_p, coords
