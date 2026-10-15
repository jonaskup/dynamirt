"""NumPy simulation and model setup for static 1PL–4PL recovery tests."""

import numpy as np
import numpyro.distributions as dist

from dynamirt import Full, dynamirt
from .config import SMALL, SEED
from .helpers import fit_responses, save_estimates


def simulate(intercept, discrimination, rng, lower=0.0, upper=1.0, n_respondents=300):
    """Independent NumPy simulation: p = lower + (upper-lower) * sigmoid(eta)."""
    theta = rng.normal(size=n_respondents)
    eta = intercept + theta[:, None] * discrimination
    probability = np.exp(-np.logaddexp(0.0, -eta))
    probability = lower + (upper - lower) * probability
    return rng.binomial(1, probability).astype(float), theta


def fit_simulated(model_type, replication, settings=SMALL, output=None):
    # Separate simulation/fitting seeds; adding replications preserves earlier ones.
    # IDs 3–5 belong to the polytomous models; preserve all existing seeds.
    model_id = {"1PL": 1, "2PL": 2, "3PL": 6, "4PL": 7}[model_type]
    seeds = np.random.SeedSequence([SEED, model_id, replication]).spawn(2)
    simulation_seed, fitting_seed = [int(s.generate_state(1)[0]) for s in seeds]
    intercept = np.linspace(-1.5, 1.5, settings.n_items)
    discrimination = np.ones(settings.n_items)
    if model_type != "1PL":
        discrimination = np.random.default_rng(314159).permutation(np.linspace(0.6, 1.6, settings.n_items))
    lower = np.linspace(0.05, 0.30, settings.n_items) if model_type in ("3PL", "4PL") else 0.0
    upper = np.linspace(0.95, 0.70, settings.n_items) if model_type == "4PL" else 1.0
    responses, theta = simulate(intercept, discrimination, np.random.default_rng(simulation_seed),
                                lower=lower, upper=upper, n_respondents=settings.n_respondents)

    # Standard-normal traits fix location/scale; positive slopes fix polarity.
    kwargs = {} if model_type == "1PL" else {"loadings": Full(dist.LogNormal(0, 0.5))}
    model = dynamirt(model_type=model_type, n_latent=1, corr=False, **kwargs)
    raw, divergences, metadata = fit_responses(
        model, responses, fitting_seed, settings, diagnostics=output is not None)
    # These raw sites equal the parameters for the unpooled, iid specification.
    truth = {"intercept": intercept, "theta": theta}
    draws = {"intercept": np.asarray(raw["measurement.intercept"]),
             "theta": np.asarray(raw["latent.residuals.coef.raw"])[:, :, 0, 0]}
    if model_type != "1PL":
        truth["discrimination"] = discrimination
        draws["discrimination"] = np.asarray(raw["loadings.full"])[:, :, 0]
    if model_type in ("3PL", "4PL"):
        truth["lower_asymptote"] = lower
        draws["lower_asymptote"] = np.asarray(raw["measurement.lower_asymptote"])
    if model_type == "4PL":
        truth["upper_asymptote"] = upper
        # The sampled gap is a fraction of the remaining probability range.
        lower_draws = draws["lower_asymptote"]
        draws["upper_asymptote"] = lower_draws + (1 - lower_draws) * np.asarray(raw["measurement.upper_asymptote_fraction"])
    if output is not None:
        save_estimates(model_type, replication, truth, draws, metadata, simulation_seed, output, settings)
    return truth, draws, divergences
