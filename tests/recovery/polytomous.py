"""NumPy simulation for GRM and PCM/GPCM, with 2–5 categories between items."""

import numpy as np
import numpyro.distributions as dist

from dynamirt import Full, dynamirt
from .config import SMALL, SEED
from .helpers import fit_responses, save_estimates


def category_layout(n_items):
    counts = np.resize([2, 3, 4, 5], n_items)
    valid = np.arange(counts.max() - 1) < counts[:, None] - 1
    return counts, valid


# The small pytest checks use this mask; study fits construct their own.
_, VALID = category_layout(SMALL.n_items)


def simulate(model_type, rng, settings=SMALL):
    """Generate probabilities directly, independently of the package likelihood."""
    counts, valid = category_layout(settings.n_items)
    theta = rng.normal(size=settings.n_respondents)
    discrimination = (np.ones(settings.n_items) if model_type == "PCM" else
                      np.random.default_rng(314159).permutation(np.linspace(0.6, 1.6, settings.n_items)))
    thresholds = np.zeros(valid.shape)
    responses = np.empty((settings.n_respondents, settings.n_items))
    for item, count in enumerate(counts):
        levels = np.linspace(-0.6 * (count - 2), 0.6 * (count - 2), count - 1)
        # PCM/GPCM steps need not be ordered; exercise that as well.
        if model_type != "GRM" and item % 2:
            levels = levels[::-1]
        levels = levels + np.linspace(-0.5, 0.5, settings.n_items)[item]
        thresholds[item, :count - 1] = levels
        eta = discrimination[item] * theta[:, None]
        if model_type == "GRM":
            cumulative = np.exp(-np.logaddexp(0.0, eta - levels))
            probability = np.diff(np.column_stack([np.zeros(settings.n_respondents), cumulative,
                                                   np.ones(settings.n_respondents)]), axis=1)
        else:
            logits = np.column_stack([np.zeros(settings.n_respondents), np.cumsum(eta - levels, axis=1)])
            probability = np.exp(logits - logits.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
        assert np.all(probability >= 0)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        responses[:, item] = (rng.random((settings.n_respondents, 1)) >
                              probability.cumsum(axis=1)[:, :-1]).sum(axis=1)
    name = "cutpoints" if model_type == "GRM" else "steps"
    truth = {name: thresholds, "theta": theta}
    if model_type != "PCM":
        truth["discrimination"] = discrimination
    return responses, truth


def fit_simulated(model_type, replication, settings=SMALL, output=None):
    counts, valid = category_layout(settings.n_items)
    model_id = {"GRM": 3, "PCM": 4, "GPCM": 5}[model_type]
    seeds = np.random.SeedSequence([SEED, model_id, replication]).spawn(2)
    simulation_seed, fitting_seed = [int(s.generate_state(1)[0]) for s in seeds]
    responses, truth = simulate(model_type, np.random.default_rng(simulation_seed), settings)
    assert ((responses >= 0) & (responses < counts)).all()
    kwargs = {} if model_type == "PCM" else {"loadings": Full(dist.LogNormal(0, 0.5))}
    model = dynamirt(model_type=model_type, n_latent=1, corr=False,
                     model_type_kwargs={"n_cat": counts}, **kwargs)
    raw, divergences, metadata = fit_responses(
        model, responses, fitting_seed, settings, diagnostics=output is not None)
    name = "cutpoints" if model_type == "GRM" else "steps"
    if model_type == "GRM":
        thresholds = np.zeros((len(raw["latent.residuals.coef.raw"]), *valid.shape))
        for count in np.unique(counts):
            p = np.asarray(raw[f"measurement.baseline_probs_{count}"])
            below = p.cumsum(axis=-1)[..., :-1]
            above = p[..., ::-1].cumsum(axis=-1)[..., ::-1][..., 1:]
            thresholds[:, counts == count, :count - 1] = np.log(below) - np.log(above)
    else:
        compact = np.asarray(raw[f"measurement.{name}.raw"])
        thresholds = np.zeros((len(compact), *valid.shape))
        thresholds[:, valid] = compact
    draws = {name: thresholds, "theta": np.asarray(raw["latent.residuals.coef.raw"])[:, :, 0, 0]}
    if model_type != "PCM":
        draws["discrimination"] = np.asarray(raw["loadings.full"])[:, :, 0]
    if output is not None:
        save_estimates(model_type, replication, truth, draws, metadata, simulation_seed, output, settings, valid=valid)
    return truth, draws, divergences
