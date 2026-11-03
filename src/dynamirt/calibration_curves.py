"""Draw-based category-specific calibration on the rows used for fitting."""

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import xarray as xr

from .fit import FitResult
from ._replay import _replay_category_log_probs


def calibration(
    result: FitResult,
    *,
    posterior: xr.Dataset | xr.DataTree | None = None,
    num_samples: int | None = None,
    n_bins: int = 10,
) -> xr.Dataset:
    """
    WIP - should operate on held out data eventually. As of now might lead to
    misleading results when calibrated on train data.
    
    Calculate per-item, per-category calibration on the original training rows.

    Replays the full model, including latent terms and DIF, using joint
    posterior draws. Only stochastic parameters are reused; deterministic
    quantities are reconstructed. No new-row or held-out prediction is done.

    Args:
        result: FitResult from a dynamirt model.
        posterior: Optional posterior Dataset or idata["posterior"] node,
            retaining all stochastic parameters and the original model order.
            Defaults to result.to_idata()["posterior"].
        num_samples: SVI draw count when posterior is omitted (default 500).
            Leave None for MCMC or an explicit posterior.
        n_bins: Number of equal-width bins spanning [0, 1].

    Returns:
        Dataset with predicted_rate ``(*sample_dims, item, category, bin)``,
        observed_rate and count (item, category, bin), and valid_category
        (item, category). Each category compares P(Y=k) against Y==k.
        Bin membership is fixed separately for each item and category using
        posterior mean probabilities. Interior edges belong to the bin on
        their right; probability one belongs to the final bin. Missing
        responses are excluded. Empty bins and invalid categories have count
        zero and NaN rates.
        Sampling dimensions and item labels are retained; bin_left and
        bin_right record the edges. Inputs are not modified.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    responses = np.asarray(result.responses)
    n_obs, n_items = responses.shape

    def model():
        result.model(
            responses=None, covariates=result.covariates,
            n_obs=n_obs, n_var=n_items,
        )

    if posterior is None:
        posterior = result.to_idata(num_samples=num_samples)["posterior"]
    elif num_samples is not None:
        raise ValueError("num_samples applies only when posterior is omitted")
    if isinstance(posterior, xr.DataTree):
        posterior = posterior.to_dataset()

    log_probs, sample_dims = _replay_category_log_probs(
        model, posterior, jax.random.key(0)
    )
    sample_shape = log_probs.shape[:-3]
    n_draws = int(np.prod(sample_shape))
    probabilities = np.asarray(jnp.exp(log_probs)).reshape(
        n_draws, *log_probs.shape[-3:]
    )
    n_categories = probabilities.shape[-1]
    category_counts = np.broadcast_to(result.model._dynamirt_n_cat, (n_items,))
    valid_category = np.arange(n_categories) < category_counts[:, None]
    edges = np.linspace(0, 1, n_bins + 1)
    membership = np.searchsorted(edges[1:-1], probabilities.mean(axis=0), side="right")
    observed = ~np.isnan(responses)
    shape = (n_items, n_categories, n_bins)
    count = np.zeros(shape, dtype=int)
    observed_rate = np.full(shape, np.nan)
    predicted_rate = np.full((n_draws, *shape), np.nan)
    for item in range(n_items):
        for category in range(category_counts[item]):
            for bin_index in range(n_bins):
                selected = observed[:, item] & (membership[:, item, category] == bin_index)
                count[item, category, bin_index] = selected.sum()
                if count[item, category, bin_index]:
                    observed_rate[item, category, bin_index] = (responses[selected, item] == category).mean()
                    predicted_rate[:, item, category, bin_index] = probabilities[:, selected, item, category].mean(axis=1)

    return xr.Dataset(
        {
            "predicted_rate": ((*sample_dims, "item", "category", "bin"),
                               predicted_rate.reshape(*sample_shape, *shape)),
            "observed_rate": (("item", "category", "bin"), observed_rate),
            "valid_category": (("item", "category"), valid_category),
            "count": (("item", "category", "bin"), count),
        },
        coords={
            **{d: np.asarray(posterior[d]) for d in sample_dims},
            "item": np.asarray(posterior.coords.get(
                "item", result.coords.get("item", np.arange(n_items))
            )),
            "category": np.arange(n_categories),
            "bin": np.arange(n_bins),
            "bin_left": ("bin", edges[:-1]),
            "bin_right": ("bin", edges[1:]),
        },
    )


def plot_calibration(
    calibration: xr.Dataset,
    *,
    item: str | int = 0,
    category: int | None = None,
    prob: float = 0.95,
    ax: Axes | None = None,
) -> Axes:
    """Plot one item/category's observed rates against predicted rates.

    Horizontal bars are pointwise highest-density intervals (HDIs) with mass
    prob, conditional on fixed bin membership. They describe uncertainty in
    predicted rates, not observed-rate uncertainty or predictive intervals.
    item and category are coordinate labels. category defaults to 1 for
    two-category items; otherwise it must be supplied. Empty bins are omitted.
    Returns Axes without showing the figure; titles and layout are left to
    the caller.
    """
    if not 0 < prob < 1:
        raise ValueError("prob must lie between zero and one")
    selected = calibration.sel(item=item)
    valid = selected.category.values[selected.valid_category.values]
    if category is None:
        if len(valid) != 2:
            raise ValueError("Select a category for an ordinal item")
        category = 1
    if category not in valid:
        raise ValueError(f"Category {category} is not valid for item {item!r}")
    selected = selected.sel(category=category)
    selected = selected.isel(bin=selected["count"] > 0)
    values = selected.predicted_rate
    sample_dims = [d for d in ("chain", "draw", "sample") if d in values.dims]
    mean = values.mean(sample_dims).values
    if ax is None:
        _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean, selected.observed_rate.values, "o-", label=f"{item}: category {category}")
    # Draw intervals directly: a skewed posterior's mean can lie outside its
    # HDI, which would make errorbar's offsets negative.
    if selected.sizes["bin"]:
        bounds = az.hdi(values, prob=prob, dim=sample_dims)
        ax.hlines(selected.observed_rate.values,
                  bounds.sel(ci_bound="lower").values,
                  bounds.sel(ci_bound="upper").values,
                  color=ax.lines[-1].get_color())
    ax.set(xlabel="Mean predicted probability", ylabel=f"Observed proportion (Y = {category})",
           xlim=(0, 1), ylim=(0, 1))
    return ax
