"""Composition and smoke tests for `gllvm`.

Two layers:

  0. the model has a finite log density and finite gradients (no inference)
  1. SVI runs and the posterior predictive tracks the simulated trend

Layer 0 is the real composition test: it runs a forward *and* backward pass over
the whole assembly, so it catches NaN gradients, shape mismatches in the
`eta`/`u` accumulation, and site-name collisions. Layer 1 only checks the fit is
not unreasonable; it is not a recovery test.

All terms are fitted against the same dataset: a smooth trend over time that
differs by group, loaded onto `n_var` responses. Every term can represent that
except `AR1`, which is stationary by construction -- hence its lower threshold.

Set MINI_GLLVM_FIGURES=1 to write per-term diagnostic plots to ./figures.
"""

import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True) # necessary for GP underflow

import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import initialize_model
from numpyro.optim import Adam

from tinygp import kernels

from dynamirt.gllvm.model import gllvm
from dynamirt.gllvm.families import Gaussian
from dynamirt.gllvm.parameters import Param
from dynamirt.gllvm.terms.linear import Linear
from dynamirt.gllvm.terms.discrete import AR1, GRW
from dynamirt.gllvm.terms.continuous import HSGP, GP


N_TIME = 100
N_GROUPS = 2
N_OBS = N_TIME * N_GROUPS
N_VAR = 5
NOISE = 0.5

FIGURES = Path(__file__).parent / "figures" if os.environ.get("MINI_GLLVM_FIGURES") else None


TERMS = {
    "linear": (Linear("trend", "t_", group_by="g"), 0.8),
    "grw": (GRW("trend", order_by="t", group_by="g", scale=Param(dist.HalfNormal(0.2))), 0.8),
    "ar1": (AR1("trend", order_by="t", group_by="g", scale=Param(dist.HalfNormal(0.2))), 0.4),
    "hsgp": (HSGP("trend", "t_", kernel="Matern", nu=1.5, ell=2.0, m=15, group_by="g"), 0.8),
    "gp": (GP("trend", "t_",
                         kernel=lambda p: p["amplitude"] ** 2 * kernels.Matern52(p["length"]),
                         params={"amplitude": Param(1.0),
                                 "length": Param(dist.InverseGamma(5, 5))},
                         group_by="g"), 0.8),
}


@pytest.fixture(scope="module")
def covariates():
    time = np.tile(np.arange(N_TIME), N_GROUPS)
    return {
        "g": np.repeat(np.arange(N_GROUPS), N_TIME),   # grouping factor
        "t": time,                                     # index for ordered terms
        "t_": time / (N_TIME - 1) * 2 - 1,             # same, scaled to [-1, 1]
    }


@pytest.fixture(scope="module")
def responses(covariates):
    """Fixed truth, random noise."""
    rng = np.random.default_rng(0)

    slope = np.array([1.5, -1.0])[covariates["g"]]
    trend = slope * covariates["t_"]                   # (n_obs,)

    intercept = rng.normal(0.0, 1.0, N_VAR)
    loading = rng.uniform(0.6, 1.4, N_VAR)             # positive, so the sign of trend is fixed

    mu = intercept + np.outer(trend, loading)
    return jnp.asarray(mu + rng.normal(0.0, NOISE, mu.shape))


def build_model(covariates, latent_term):
    return partial(gllvm,
                   index_sizes={"g": N_GROUPS, "t": N_TIME},
                   covariates=covariates,
                   family=Gaussian(),
                   n_latent=1,
                   full_rank_regression=[Linear("intercept", "one_")],
                   latent_regression=[Linear("baseline", "one_", group_by="g"), latent_term])


def centred(a):
    """Remove per-response means, leaving the trend."""
    a = np.asarray(a)
    return a - a.mean(0)


def save_figure(name, covariates, observed, fitted, corr):
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, N_GROUPS, figsize=(10, 4), sharey=True)

    for group, ax in enumerate(axes):
        rows = covariates["g"] == group
        x = covariates["t_"][rows]
        ax.scatter(x, observed[rows].mean(1), s=8, alpha=0.4, label="observed")
        ax.plot(x, fitted[rows].mean(1), color="crimson", label="fitted")
        ax.set_title(f"group {group}")
        ax.set_xlabel("t_")

    axes[0].set_ylabel("mean over responses (centred)")
    axes[0].legend()
    fig.suptitle(f"{name}  (r = {corr:.2f})")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=120)
    plt.close(fig)


@pytest.mark.parametrize("name", TERMS)
def test_log_density_and_gradients_are_finite(covariates, responses, name):
    term, _ = TERMS[name]
    model = build_model(covariates, term)

    init = initialize_model(random.PRNGKey(0), model, model_kwargs={"responses": responses})

    assert jnp.isfinite(init.param_info.potential_energy)
    assert all(jnp.isfinite(g).all() for g in jax.tree_util.tree_leaves(init.param_info.z_grad))


@pytest.mark.slow
@pytest.mark.parametrize("name", TERMS)
def test_fit_tracks_the_trend(covariates, responses, name):
    term, min_corr = TERMS[name]
    model = build_model(covariates, term)
    guide = AutoNormal(model)

    result = SVI(model, guide, Adam(1e-2), Trace_ELBO()).run(
        random.PRNGKey(1), 5000, responses=responses, progress_bar=False)

    assert jnp.isfinite(result.losses).all()

    predictive = Predictive(model, guide=guide, params=result.params, num_samples=100)
    draws = predictive(random.PRNGKey(2), responses=None, n_obs=N_OBS, n_var=N_VAR)["Y"]

    # averaging over draws leaves (approximately) the fitted mu; centring removes
    # the intercepts, so the correlation reflects the trend only
    fitted, observed = centred(draws.mean(0)), centred(responses)
    corr = np.corrcoef(fitted.ravel(), observed.ravel())[0, 1]

    if FIGURES:
        save_figure(name, covariates, observed, fitted, corr)

    assert corr > min_corr


def test_stacked_terms_compose(covariates, responses):
    """Several terms in both regressions at once."""
    model = partial(gllvm,
                    index_sizes={"g": N_GROUPS, "t": N_TIME},
                    covariates=covariates,
                    family=Gaussian(),
                    n_latent=1,
                    full_rank_regression=[Linear("intercept", "one_"),
                                          Linear("slope", "t_", group_by="g", corr="variables"),
                                          HSGP("wiggle", "t_", kernel="ExpSquared", ell=2.0, m=10)],
                    latent_regression=[GRW("drift", order_by="t", group_by="g"),
                                       Linear("latent_slope", "t_", group_by="g")])

    init = initialize_model(random.PRNGKey(0), model, model_kwargs={"responses": responses})

    assert jnp.isfinite(init.param_info.potential_energy)
    assert all(jnp.isfinite(g).all() for g in jax.tree_util.tree_leaves(init.param_info.z_grad))

    sites = {name for name, site in init.model_trace.items() if site["type"] == "sample"}
    assert {"item.intercept.coef.raw", "item.slope.coef.raw", "item.wiggle.basis_weights.raw", "latent.drift.innovations.raw"} <= sites