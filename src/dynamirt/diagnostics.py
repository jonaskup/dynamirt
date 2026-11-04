"""Posterior plots and conditional response probabilities and means."""

from collections.abc import Mapping, Sequence

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp

from .fit import FitResult
from ._replay import _replay_category_log_probs, _replay_conditional_means
from numpyro import handlers


def plot_trajectories(
    idata: xr.DataTree,
    *,
    time: str,
    coords: Mapping[str, object] | None = None,
    var_name: str = "theta",
    latent: str | int | Sequence[str | int] | None = None,
    prob: float = 0.95,
    ax: Axes | None = None,
    line_kwargs: Mapping[str, object] | None = None,
    fill_kwargs: Mapping[str, object] | None = None,
) -> Axes:
    """Plot one respondent's latent medians and pointwise posterior HDIs.

    Args:
        idata: ArviZ data with an obs-by-latent posterior variable.
        time: Name of the observation coordinate containing time values.
        coords: Observation coordinate selections, e.g. {"respondent_id": 42}.
            Each value must be scalar; selections are combined. With None,
            the data must already describe one respondent. Defaults to None.
        var_name: Posterior trajectory variable. Defaults to "theta".
        latent: Latent coordinate label or labels, including integer labels.
            None plots all latents. Defaults to None.
        prob: Probability mass of the pointwise HDI. Defaults to 0.95.
        ax: Axes to draw on. Creates one if None. Defaults to None.
        line_kwargs: Extra keyword arguments for the median lines
            (ax.plot), overriding the defaults marker="o", markersize=3 and
            the latent label. Use {"marker": ""} to hide markers. Applied to
            every latent, so a "color" or "label" here is shared by all lines.
            Defaults to None.
        fill_kwargs: Extra keyword arguments for the HDI bands
            (ax.fill_between), overriding the defaults alpha=0.3 and the
            matching line color. Defaults to None.

    Returns:
        The matplotlib Axes. Lines are labeled using latent coordinates;
        call ax.legend() to display a legend.

    Raises:
        KeyError: If a requested variable or coordinate label is missing.
        ValueError: If dimensions or selections are invalid, no observations
            remain, or selected times are missing or duplicated.
    """
    if not 0 < prob < 1:
        raise ValueError("prob must lie between zero and one")
    if var_name not in idata["posterior"]:
        raise KeyError(f"Posterior variable missing: {var_name}")
    values = idata["posterior"][var_name]
    sample_dims = [dim for dim in ("chain", "draw", "sample") if dim in values.dims]
    if (
        set(sample_dims) not in ({"chain", "draw"}, {"draw"}, {"sample"})
        or set(values.dims) != {*sample_dims, "obs", "latent"}
    ):
        raise ValueError(
            f"{var_name!r} must have obs and latent dimensions plus chain/draw or sample"
        )

    for name in {time, *(coords or {})}:
        if name not in values.coords:
            raise KeyError(f"Unknown coordinate: {name}")
        if values[name].dims != ("obs",):
            raise ValueError(f"{name} must be an obs coordinate")
    for name, label in (coords or {}).items():
        if not np.isscalar(label):
            raise ValueError("Selections must be scalar")
        values = values.isel(obs=values[name] == label)
    if not values.sizes["obs"]:
        raise ValueError("No observations selected")
    times = values[time]
    if times.isnull().any().item() or len(np.unique(times)) != times.size:
        raise ValueError("Times must be present and unique")

    values = values.sortby(time)
    if latent is not None:
        labels = [latent] if np.isscalar(latent) else list(latent)
        values = values.sel(latent=labels)

    median = values.median(dim=sample_dims)
    interval = az.hdi(values, dim=sample_dims, prob=prob)
    if ax is None:
        _, ax = plt.subplots()
    for label in values.latent.values:
        line, = ax.plot(
            values[time].values, median.sel(latent=label).values,
            **{"marker": "o", "markersize": 3, "label": str(label), **(line_kwargs or {})},
        )
        ax.fill_between(
            values[time].values,
            interval.sel(latent=label, ci_bound="lower").values,
            interval.sel(latent=label, ci_bound="upper").values,
            **{"color": line.get_color(), "alpha": 0.3, **(fill_kwargs or {})},
        )
    ax.set_xlabel(time)
    ax.set_ylabel(var_name)
    return ax

def plot_loadings(
    idata: xr.DataTree,
    *,
    var_name: str = "loadings",
    item: str | int | Sequence[str | int] | None = None,
    latent: str | int | Sequence[str | int] | None = None,
    prob: float = 0.95,
    annotate: bool = True,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    colorbar: bool = True,
    ax: Axes | None = None,
    imshow_kwargs: Mapping[str, object] | None = None,
    text_kwargs: Mapping[str, object] | None = None,
    colorbar_kwargs: Mapping[str, object] | None = None,
) -> Axes:
    """Plot posterior median loadings as a heatmap.

    Args:
        idata: ArviZ data with an item-by-latent posterior variable.
        var_name: Posterior variable to plot. Defaults to "loadings".
        item: Item coordinate label or labels. None selects all. Defaults to None.
        latent: Latent coordinate label or labels. None selects all. Defaults to None.
        prob: Probability mass of annotated HDIs. Defaults to 0.95.
        annotate: Show medians and HDIs; bold means the HDI excludes zero.
            Defaults to True.
        cmap: Matplotlib colormap. Defaults to "RdBu_r".
        vmax: Positive limit for the symmetric color range. None uses the
            largest absolute median, or 1 for all-zero medians. Defaults to None.
        colorbar: Add a colorbar to the figure. Defaults to True.
        ax: Axes to draw on. Creates one if None. Defaults to None.
        imshow_kwargs: Extra keyword arguments for the heatmap (ax.imshow),
            overriding cmap, the symmetric vmin/vmax and aspect="auto".
            Defaults to None.
        text_kwargs: Extra keyword arguments for the cell annotations
            (ax.text), overriding fontsize=8 and the contrast-based color and
            HDI-based fontweight in every cell. Defaults to None.
        colorbar_kwargs: Extra keyword arguments for the colorbar
            (Figure.colorbar), overriding shrink=0.8. Defaults to None.

    Returns:
        The matplotlib Axes. Titles and figure layout are left to the caller.
    """
    values = idata["posterior"][var_name]
    for dim, selection in (("item", item), ("latent", latent)):
        if selection is not None:
            labels = [selection] if np.isscalar(selection) else list(selection)
            values = values.sel({dim: labels})
    sample_dims = [dim for dim in ("chain", "draw", "sample") if dim in values.dims]
    median = values.median(dim=sample_dims).transpose("item", "latent").values
    if annotate:
        interval = az.hdi(values, prob=prob, dim=sample_dims)
        lo = interval.sel(ci_bound="lower").transpose("item", "latent").values
        hi = interval.sel(ci_bound="upper").transpose("item", "latent").values

    if vmax is None:
        vmax = float(np.abs(median).max()) or 1.0
    if vmax <= 0:
        raise ValueError("vmax must be positive")
    n_items, n_lat = median.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(1.7 * n_lat + 2, 0.75 * n_items + 1.5))
    im = ax.imshow(
        median,
        **{"cmap": cmap, "vmin": -vmax, "vmax": vmax, "aspect": "auto", **(imshow_kwargs or {})},
    )

    if annotate:
        for i in range(n_items):
            for j in range(n_lat):
                # Choose black or white by contrast against the displayed cell.
                rgba = np.asarray(im.cmap(im.norm(median[i, j])))
                rgb = rgba[:3] * rgba[3] + np.asarray(ax.get_facecolor()[:3]) * (1 - rgba[3])
                linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
                ax.text(
                    j, i, f"{median[i, j]:.2f}\n[{lo[i, j]:.2f}, {hi[i, j]:.2f}]",
                    **{
                        "ha": "center", "va": "center", "fontsize": 8,
                        "color": "black" if luminance > 0.179 else "white",
                        "fontweight": "bold" if (lo[i, j] > 0) or (hi[i, j] < 0) else "normal",
                        **(text_kwargs or {}),
                    },
                )

    ax.set_xticks(range(n_lat), values["latent"].values)
    ax.set_yticks(range(n_items), values["item"].values)
    ax.set_xticks(np.arange(n_lat + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_items + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", lw=1.5)
    ax.tick_params(which="minor", length=0)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, **{"shrink": 0.8, **(colorbar_kwargs or {})})
    return ax


def response_curves(
    result: FitResult,
    *,
    statistic: str = "auto",
    covariates: Mapping[str, object] | None = None,
    latent: str | int | Sequence[str | int] | None = None,
    theta_range: tuple[float, float] = (-3, 3),
    n_points: int = 100,
    posterior: xr.Dataset | xr.DataTree | None = None,
    num_samples: int | None = None,
) -> xr.Dataset:
    """Evaluate conditional response probabilities or means on a latent grid.
    Vary one or two latent dimensions, holding other dimensions at zero.

    Args:
        result: FitResult from a model built with dynamirt or GLLVM.
        statistic: "auto" uses probabilities for distributions with enumerable
            discrete support, and conditional means otherwise. Explicit
            "probability" or "mean" overrides this choice. Means require an
            implemented distribution mean; ordinal means use numeric category
            scores. Undefined means are propagated.
        covariates: Values required by the measurement model, including DIF.
            Scalars broadcast across the theta grid; vectors follow its point
            order. Use the fitted coding/scaling and group indices. Defaults
            to an empty mapping; training covariates are not filled in.
        latent: One or two latent coordinate labels to vary. Required for
            multidimensional models; other latent coordinates stay at zero.
        theta_range: Increasing (lower, upper) bounds, shared by grid axes.
        n_points: Points per axis (a two-factor grid has ``n_points**2`` points).
        posterior: Optional posterior Dataset or idata["posterior"] DataTree
            node, retaining all stochastic measurement parameters in model
            order. Defaults to result.to_idata()["posterior"]. Reuse this
            argument to evaluate the same draws on different grids.
        num_samples: SVI draw count when posterior is omitted; defaults to
            the conversion default (500). Leave None for MCMC or posterior.

    Returns:
        For probabilities, a Dataset with probability ``(*sample_dims, point, item, category)``,
        theta (point, latent), and valid_category (item, category). Sampling
        dimensions and coordinate labels are retained. The varied_latent
        attribute records the grid axes in order. Invalid categories have
        probability zero. All original measurement terms, including DIF, are
        evaluated at the supplied covariates. Latent terms are replaced by the
        theta grid. These are conditional slices, not averages over the other
        latent dimensions or training covariates. The supplied profile is not
        stored in the output. For means, ``mean`` has dimensions
        ``(*sample_dims, point, item)`` and category variables are omitted.
        These are conditional expectations, not future response draws.
        The resolved statistic is recorded in the Dataset attributes.
        DIF terms must support evaluation on this grid;
        this does not provide GP conditioning at new inputs.
    """
    if statistic not in ("auto", "probability", "mean"):
        raise ValueError("statistic must be 'auto', 'probability', or 'mean'")
    if posterior is None:
        posterior = result.to_idata(num_samples=num_samples)["posterior"]
    elif num_samples is not None:
        raise ValueError("num_samples applies only when posterior is omitted")
    
    if isinstance(posterior, xr.DataTree):
        posterior = posterior.to_dataset()
        
    n_latent = result.model._dynamirt_n_latent
    n_items = np.shape(result.responses)[1]
    labels = np.asarray(posterior.coords.get(
        "latent", result.coords.get("latent", np.arange(n_latent))
    ))
    
    if latent is None:
        if n_latent != 1:
            raise ValueError("Select one or two latent labels for a multidimensional model")
        selected = labels.tolist()
    else:
        selected = [latent] if np.isscalar(latent) else list(latent)
    if len(selected) not in (1, 2) or len(set(selected)) != len(selected):
        raise ValueError("Select one or two distinct latent labels")
    
    indices = [labels.tolist().index(label) for label in selected]
    
    if n_points < 2 or theta_range[0] >= theta_range[1]:
        raise ValueError("Use n_points >= 2 and increasing theta_range bounds")
    
    axis = np.linspace(*theta_range, n_points)
    mesh = np.meshgrid(*[axis] * len(selected), indexing="ij")
    theta = np.zeros((n_points ** len(selected), n_latent))
    theta[:, indices] = np.stack([m.ravel() for m in mesh], axis=-1)

    def grid_term(ctx, n_latent):
        return jnp.asarray(theta)
    
    grid_term.name = "theta_grid"

    def grid_model():
        result.model(
            responses=None, covariates={} if covariates is None else covariates,
            n_obs=len(theta), n_var=n_items,
            latent_regression=[grid_term],
        )

    # Inspect the actual distribution so custom family callables work too.
    response_dist = handlers.trace(handlers.seed(grid_model, jax.random.key(0))).get_trace()["Y"]["fn"]
    enumerable = response_dist.support.is_discrete and response_dist.has_enumerate_support
    if statistic == "auto":
        statistic = "probability" if enumerable else "mean"
    if statistic == "probability" and not enumerable:
        raise ValueError("Probability curves require enumerable finite-category responses")

    data = {"theta": (("point", "latent"), theta)}
    category_coords = {}
    if statistic == "probability":
        log_probs, sample_dims = _replay_category_log_probs(grid_model, posterior, jax.random.key(0))
        probabilities = np.asarray(jnp.exp(log_probs))
        n_cat = probabilities.shape[-1]
        counts = np.broadcast_to(getattr(result.model, "_dynamirt_n_cat", n_cat), (n_items,))
        data["probability"] = ((*sample_dims, "point", "item", "category"), probabilities)
        data["valid_category"] = (("item", "category"), np.arange(n_cat) < counts[:, None])
        category_coords["category"] = np.arange(n_cat)
    else:
        means, sample_dims = _replay_conditional_means(grid_model, posterior, jax.random.key(0))
        data["mean"] = ((*sample_dims, "point", "item"), np.asarray(means))

    return xr.Dataset(
        data,
        coords={
            **{d: np.asarray(posterior[d]) for d in sample_dims},
            "point": np.arange(len(theta)), "latent": labels,
            "item": np.asarray(posterior.coords.get(
                "item", result.coords.get("item", np.arange(n_items))
            )),
            **category_coords,
        },
        attrs={"varied_latent": selected, "statistic": statistic},
    )


def plot_response_curves(
    curves: xr.Dataset,
    *,
    item: str | int = 0,
    category: int | None = None,
    prob: float = 0.95,
    ax: Axes | None = None,
    line_kwargs: Mapping[str, object] | None = None,
    fill_kwargs: Mapping[str, object] | None = None,
    mesh_kwargs: Mapping[str, object] | None = None,
    colorbar_kwargs: Mapping[str, object] | None = None,
) -> Axes:
    """Plot one item's conditional response curves or a category heatmap.

    Args:
        curves: Dataset returned by response_curves.
        item: Item coordinate label (not positional index).
        category: Category label. None shows category 1 for binary items or
            all valid categories for ordinal 1D curves. Ordinal 2D plots
            require a category. Not applicable to mean curves.
        prob: Pointwise posterior HDI mass for 1D curves. Two-dimensional
            heatmaps show posterior medians without intervals.
        ax: Optional matplotlib Axes. Titles and layout are left to the caller.
        line_kwargs: Extra keyword arguments for 1D median lines (ax.plot),
            overriding the category label. Applied to every category.
        fill_kwargs: Extra keyword arguments for 1D HDI bands
            (ax.fill_between), overriding alpha=0.3 and the line color.
        mesh_kwargs: Extra keyword arguments for the 2D heatmap
            (ax.pcolormesh), overriding shading="auto"; probabilities also default to vmin=0, vmax=1.
        colorbar_kwargs: Extra keyword arguments for the 2D colorbar
            (Figure.colorbar), overriding the statistic label.

    Returns:
        Matplotlib Axes. One-dimensional lines have category labels for use
        with ax.legend(); two-dimensional plots include a statistic colorbar.
    """
    selected = curves.attrs["varied_latent"]
    is_mean = curves.attrs.get("statistic") == "mean" or "probability" not in curves
    if is_mean:
        if category is not None:
            raise ValueError("category is not applicable to conditional mean curves")
        categories = ["mean"]
        values = curves["mean"].sel(item=item).expand_dims(category=categories)
    else:
        valid = curves.category.values[curves.valid_category.sel(item=item).values]
        if category is None:
            if len(valid) == 2:
                categories = [1]
            elif len(selected) == 2:
                raise ValueError("Select a category for a two-dimensional ordinal plot")
            else:
                categories = valid
        else:
            if category not in valid:
                raise ValueError(f"Category {category} is not valid for item {item!r}")
            categories = [category]
        values = curves.probability.sel(item=item, category=categories)
    sample_dims = [d for d in ("chain", "draw", "sample") if d in values.dims]
    median = values.median(sample_dims)
    if ax is None:
        _, ax = plt.subplots()
    x = curves.theta.sel(latent=selected[0]).values
    if len(selected) == 1:
        interval = az.hdi(values, dim=sample_dims, prob=prob)
        for cat in categories:
            line, = ax.plot(
                x, median.sel(category=cat).values,
                **{"label": str(cat), **(line_kwargs or {})},
            )
            ax.fill_between(
                x, interval.sel(category=cat, ci_bound="lower").values,
                interval.sel(category=cat, ci_bound="upper").values,
                **{"color": line.get_color(), "alpha": 0.3, **(fill_kwargs or {})},
            )
        if not is_mean:
            ax.set_ylim(0, 1)
        ax.set_ylabel("Conditional mean" if is_mean else "Probability")
    else:
        y = curves.theta.sel(latent=selected[1]).values
        xs, ys = np.unique(x), np.unique(y)
        z = median.sel(category=categories[0]).values.reshape(len(xs), len(ys))
        im = ax.pcolormesh(
            xs, ys, z.T,
            **{"shading": "auto", **({} if is_mean else {"vmin": 0, "vmax": 1}), **(mesh_kwargs or {})},
        )
        ax.figure.colorbar(
            im, ax=ax,
            **{"label": "E[Y | theta, x]" if is_mean else f"P(Y = {categories[0]})", **(colorbar_kwargs or {})},
        )
        ax.set_ylabel(str(selected[1]))
    ax.set_xlabel(str(selected[0]))
    return ax
