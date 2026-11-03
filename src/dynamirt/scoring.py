"""Response scores on the fitted observation design."""

import jax
import jax.numpy as jnp
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

    Returns:
        DataArray named log_loss with dimensions (obs, item), observation
        covariates as auxiliary coordinates, and NaN at unscored cells.
        Use .mean() for an overall cell-weighted score or .mean("obs") for
        per-item scores.
    """
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

    log_p, sample_dims = _replay_category_log_probs(model, posterior, jax.random.key(0))
    sample_shape = log_p.shape[:len(sample_dims)]
    if not sample_dims or not np.prod(sample_shape):
        raise ValueError("posterior must contain nonempty sample or chain/draw dimensions")
    log_mean_p = logsumexp(log_p, axis=tuple(range(len(sample_dims)))) - jnp.log(
        np.prod(sample_shape)
    )
    observed = ~np.isnan(values)
    categories = np.where(observed, values, 0).astype(int)
    losses = -np.asarray(jnp.take_along_axis(log_mean_p, categories[..., None], axis=-1))[..., 0]
    losses = np.where(observed, losses, np.nan)

    coords = {
        dim: np.asarray(posterior.coords.get(dim, result.coords.get(dim, np.arange(size))))
        for dim, size in (("obs", n_obs), ("item", n_items))
    }
    for name, covariate in result.covariates.items():
        if np.shape(covariate) == (n_obs,):
            if name in {"obs", "item", "log_loss"}:
                raise ValueError(f"Covariate name conflicts with score coordinates: {name!r}")
            coords[name] = ("obs", np.asarray(covariate))
    return xr.DataArray(losses, dims=("obs", "item"), coords=coords, name="log_loss")
