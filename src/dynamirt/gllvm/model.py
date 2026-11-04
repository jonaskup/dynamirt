from typing import Sequence, Mapping, List, Callable

import numpyro
import numpy as np

import jax.numpy as jnp
from jax.typing import ArrayLike

from functools import partial

from .context import ModelContext
from .loadings import Full
from .families import Gaussian


def _validate_responses(responses, discrete, n_cat=None):
    values = np.asarray(responses)
    observed = ~np.isnan(values)
    if not np.all(np.isfinite(values[observed])):
        raise ValueError("Responses must be finite or NaN")
    if discrete and np.any(values[observed] != np.floor(values[observed])):
        raise ValueError("Responses must be integers or NaN")
    if n_cat is not None and np.any(observed & ((values < 0) | (values >= n_cat))):
        raise ValueError("Response category out of range")


def GLLVM(
    n_latent: int =1,
    loadings: Callable | None =None,
    family: Callable | None =None,
    latent_terms: Sequence[Callable] | None =None,
    item_terms: Sequence[Callable] | None =None,
    index_sizes: Mapping[str, int] | None =None,
    ) -> Callable:

    """Build a generalized linear latent variable model for MCMC or SVI.

    Args:
        n_latent: Number of latent dimensions. Defaults to 1.
        loadings: Loading-matrix factory. Defaults to unconstrained ``Full()``.
        family: Callable ``family(linear_predictor, ctx)`` returning a
            NumPyro distribution. Defaults to ``Gaussian()``.
        latent_terms: Nonempty sequence of latent term callables. No iid
            latent effects are added automatically. For example, supply
            ``[Linear("residuals", group_by="obs_")]`` for row-level effects.
        item_terms: Optional full-rank terms. No intercept is added
            automatically; supply ``Linear("intercept")`` here if needed
            and not already provided by the family.
        index_sizes: Fixed sizes of named grouping and discrete-time axes.
            The ``obs_`` axis is supplied automatically.

    Returns:
        A model callable ``model(responses, covariates, ...)`` for
        ``fit_mcmc`` and ``fit_svi``, with semantic ArviZ dimensions and
        latent scores recorded as ``theta``. Category counts are copied from
        ``family.n_cat`` when available.

    Raises:
        ValueError: If no latent terms are supplied.
    """

    latent_terms = [] if latent_terms is None else list(latent_terms)
    item_terms = [] if item_terms is None else list(item_terms)
    if not latent_terms:
        raise ValueError("No latent terms: pass a nonempty latent_terms sequence.")

    if family is None:
        family = Gaussian()

    model = partial(
        gllvm,
        n_latent=n_latent,
        full_rank_regression=item_terms,
        latent_regression=latent_terms,
        loadings=loadings,
        family=family,
        latent_site_name="theta",
        index_sizes={} if index_sizes is None else dict(index_sizes),
    )

    # ----------------- state for downstream analysis/ArviZ -----------------
    model._dynamirt_dims = {
        "theta": ["obs", "latent"],
        "loadings": ["item", "latent"],
        "Y": ["obs", "item"],
        **{f"latent.{term.name}.contribution": ["obs", "latent"] for term in latent_terms},
        **{f"item.{term.name}.contribution": ["obs", "item"] for term in item_terms},
    }

    model._dynamirt_n_latent = n_latent
    if hasattr(family, "n_cat"):
        model._dynamirt_n_cat = family.n_cat

    return model


def gllvm(
    responses: None | ArrayLike,
    covariates: Mapping[str, ArrayLike],
    family: Callable,
    n_latent: int=1,
    full_rank_regression: List[Callable]=None,
    latent_regression: List[Callable]=None,
    loadings: Callable | None =None,
    n_obs: int=None,
    n_var: int=None,
    latent_site_name: str="u",
    index_sizes: Mapping[str, int] | None =None,
    ):
    
    """Generalized linear latent variable model.

    Regresses `n_var` responses jointly on a full-rank linear predictor plus a
    reduced-rank (`n_latent`) term:
    
    mu = eta + u @ loadings.T
    
    where eta is the sum of full-rank regression terms, u collects
    the latent scores, and loadings is a (n_var, n_latent) matrix.
    NaN entries in `responses` are masked out of the likelihood.
    
    Args:
        responses: Observation matrix of shape (n_obs, n_var). NaN marks
            missing data. Pass None for prior-predictive sampling, in
            which case n_obs and n_var must be given explicitly.
        covariates: Mapping from covariate name to array of length n_obs
            (or broadcastable scalar). Two keys are added automatically:
            ``"one_"`` (constant 1, for intercepts) and ``"obs_"``
            (observation index, a stand-in for a respondent ID when
            n_obs equals the number of respondents).
        family: Called as ``family(mu, ctx)`` and must return a
            ``numpyro.distributions.Distribution`` over responses.
        n_latent: Dimensionality of the latent space. Defaults to 1.
        full_rank_regression: List of term callables, each with signature
            ``term(ctx, n_var) -> (n_obs, n_var)``. Their outputs are
            summed into the full-rank predictor eta. Sites are scoped under
            ``item.<term.name>``; the row-aligned output is ``contribution``.
        latent_regression: List of term callables, each with signature
            ``term(ctx, n_latent) -> (n_obs, n_latent)``. Their outputs
            are summed into the latent scores u. Sites are scoped under
            ``latent.<term.name>``; the row-aligned output is ``contribution``.
        loadings: A loading-matrix factory ``loadings(ctx) ->
            (n_var, n_latent)``. Defaults to ``Full()`` (unconstrained).
        n_obs: Number of observations. Required when responses is None.
        n_var: Number of response variables. Required when responses is
            None.
        latent_site_name: Name of the NumPyro deterministic site that
            stores the latent scores u. Defaults to ``"u"``.
        index_sizes: Fixed sizes of named integer-coded grouping and time
            axes. Codes may select any subset of 0,...,size-1. The reserved
            obs_ axis is set automatically to n_obs.
    """
    
    if responses is None and (n_obs is None or n_var is None):
        raise ValueError("n_obs and n_var must be specified explicitly if no responses are given.")
    
    # loadings default to unconstrained
    loadings = Full() if loadings is None else loadings

    # n_obs: total number of obs. Doesn't necessarily need to be number of sites/respondents in repeated measurement scenarios
    # n_var: number of variables to simultanously regress on
    if responses is not None:
        n_obs, n_var = responses.shape 
        
    covariates = {**covariates,
                  "one_": np.array([1.0]),  # for intercepts
                  "obs_": np.arange(n_obs)} # stand-in for ID column in scenarios w/o repeated measures where n_obs == n_site/respondent
        
    ctx = ModelContext(
        responses,
        covariates,
        n_obs, 
        n_var, 
        n_latent,
        index_sizes={**(index_sizes or {}), "obs_": n_obs},
        ) # context to pass to subfunctions like term and family
    
    # full-rank regression (DIF under IRT)
    eta = jnp.zeros((n_obs, n_var))
    if full_rank_regression is not None:
        for term in full_rank_regression:
            with numpyro.handlers.scope(prefix=f"item.{term.name}", divider="."):
                contribution = term(ctx, n_var) # (n_obs, n_var)
                numpyro.deterministic("contribution", contribution)
            eta += contribution

    
    # reduced-rank regression
    u = jnp.zeros((n_obs, n_latent))
    if latent_regression is not None:
        for term in latent_regression:
            with numpyro.handlers.scope(prefix=f"latent.{term.name}", divider="."):
                contribution = term(ctx, n_latent) # (n_obs, n_latent)
                numpyro.deterministic("contribution", contribution)
            u += contribution
    
    with numpyro.handlers.scope(prefix="loadings", divider="."):
        loadings_matrix = loadings(ctx) # (n_var, n_latent)
        
    latent_contributions = u @ loadings_matrix.T # (n_obs, n_var)
    
    mu = eta + latent_contributions
    
    numpyro.deterministic(latent_site_name, u)
    numpyro.deterministic("loadings", loadings_matrix)
    
    # likelihood based on user supplied family function
    # responses should always be float to carry nan
    # must be manually cast to int if the distribution returned by family_fn only has discrete support (e.g., ordinal models)
    with numpyro.handlers.scope(prefix="measurement", divider="."):
        family_fn = family(mu, ctx)
    
    if responses is None:
        numpyro.sample("Y", family_fn)
    else:
        _validate_responses(responses, family_fn.support.is_discrete, getattr(family, "n_cat", None))
        mask = jnp.isnan(responses)
        obs=jnp.where(mask, 0.0, responses)
        
        if family_fn.support.is_discrete:
            obs = obs.astype(jnp.result_type(int))
        numpyro.sample("Y", family_fn.mask(~mask), obs=obs)
