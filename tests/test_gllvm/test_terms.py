# Test if all terms correctly return (n_obs, n_var)

import pytest

import numpy as np

from tinygp import kernels

import numpyro
import numpyro.distributions as dist

from dynamirt.gllvm.context import ModelContext
from dynamirt.gllvm.terms.linear import Linear
from dynamirt.gllvm.terms.discrete import AR1, GRW
from dynamirt.gllvm.terms.continuous import HSGP, GP
from dynamirt.gllvm.parameters import Param

@pytest.fixture
def ctx():
    n_obs = 60
    covariates = {"x": np.linspace(-1, 1, n_obs),
                  "g": np.repeat([0, 1, 2], 20),
                  "t": np.tile(np.arange(20), 3),
                  "one_": np.array([1.0]),
                  "obs_": np.arange(n_obs)}
    return ModelContext(None, covariates, n_obs, n_var=4, n_latent=2, index_sizes={"g": 3, "t": 20})

def trace_term(term, ctx, n_vars, seed=0):
    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=seed):
        out = term(ctx, n_vars)
    return out, tr

@pytest.mark.parametrize("term, shared", [
    (Linear("b", "x"), False),
    (Linear("b", "x", coef=Param(dist.Normal(0, 1), by_variable="shared")), True),

    (GRW("w", order_by="t", group_by="g"), False),
    (GRW("w", order_by="t", group_by="g", varies_over_variables=False), True),

    (AR1("a", order_by="t", group_by="g"), False),
    (AR1("a", order_by="t", group_by="g", varies_over_variables=False), True),

    (HSGP("f", "x", kernel="Matern", ell=3.0, m=8), False),
    (HSGP("f", "x", kernel="Matern", ell=3.0, m=8, varies_over_variables=False), True),

    (GP("gp", "x",
             kernel=lambda p: p["amplitude"] ** 2 * kernels.Matern52(p["length"]),
             params={"amplitude": Param(1.0),
                     "length": Param(dist.InverseGamma(5, 5))},
             group_by="g"), False),
    (GP("gp", "x",
             kernel=lambda p: p["amplitude"] ** 2 * kernels.Matern52(p["length"]),
             params={"amplitude": Param(1.0),
                     "length": Param(dist.InverseGamma(5, 5))}), False),
])
def test_term_output_shape(ctx, term, shared):
    out, _ = trace_term(term, ctx, ctx.n_var)
    assert out.shape == (ctx.n_obs, 1 if shared else ctx.n_var)