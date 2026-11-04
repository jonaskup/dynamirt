# Plots

## Loadings

The estimated loading structure can be visualized as a heatmap using {py:func}`plot_loadings <dynamirt.plot_loadings>`. In the below example, no cross-loadings will be visible because we specify a simple structure in the Q matrix:

```python
from dynamirt import dynamirt, fit_mcmc, plot_loadings

ITEMS = ["cheerful", "energetic", "interested", "relaxed", "anxious", "irritable", "down", "bored"]

Q = q_matrix([[0, 1, 2, 3], [4, 5, 6, 7]], items=8)

model = dynamirt(model_type="2PL", n_latent=2, loadings=Confirmatory(Q, positive))
fit = fit_mcmc(model, responses)
idata = fit.to_idata(coords={"latent": ["positive_affect", "negative_affect"], "item": ITEMS})

plot_loadings(idata)
```

## Item Characteristic Curves

The package can automatically visualize the estimated item characteristic curves independently of the selected measurement model. Internally, this is done by replacing the latent terms with a regular grid in the posterior and tracing across it. 

This example plots the items characteristic curves for the four items on the positive affect latent on a single matplotlib axis:

```python
import matplotlib.pyplot as plt
from dynamirt import response_curves, plot_response_curves

icc = response_curves(fit, posterior=idata["posterior"], latent=["positive_affect"])

fig, ax = plt.subplots()

for item in ["cheerful", "energetic", "interested", "relaxed"]:
    plot_response_curves(icc, item=item, ax=ax)

```
The example above considers only the positive affect latent conditioned on negative_affect == 0. This is not an issue here because we specified a simple loading structure (each item loads either onto the positive or negative affect latent). For a two-dimensional model that also includes some cross loadings, we can visualize the decision boundaries in the latent space instead. The {py:func}`response_curves <dynamirt.response_curves>` function varies up to two latents, holding any remaining dimensions at zero.

```python
icc = response_curves(fit, posterior=idata["posterior"], latent=["positive_affect", "negative_affect"])
plot_response_curves(icc, item="cheerful")
```

Documentation on item curves in the presence of differential item functioning to be added.

## Latent trajectories

Because dynamirt is geared toward time series modeling, it also includes a helper to quickly plot latent trajectories over time for a single respondent. This is done through the {py:func}`plot_trajectories <dynamirt.plot_trajectories>` function. You need to specify the name of the covariate that corresponds to the time dimensions, as well as filter down the posterior to a single respondent through the `coords` keyword which will be a familiar workflow to ArviZ users.

```python
import numpy as np
from dynamirt import Linear, plot_trajectories

covariates = {
    "respondent_id": np.repeat(np.arange(9), 3),
    "time_point": np.tile(np.arange(3), 9)
}

model = dynamirt(
    model_type="GRM", 
    n_latent=2, 
    loadings=Confirmatory(Q, positive),
    model_type_kwargs={"n_cat": 5},
    latent_terms=[
        Linear("intercept", group_by="respondent_id"),
        Linear("slope", predictors="time_point", group_by="respondent_id")
        ],
    index_sizes={"respondent_id": 9}
    )

fit = fit_mcmc(model, responses, covariates)
idata = fit.to_idata(coords={"latent": ["positive_affect", "negative_affect"]})

plot_trajectories(idata, time="time_point", coords={"respondent_id": 0}, latent="positive_affect")
```

By default this makes use of `"theta"`, i.e., the combined contributions of all latent terms. You can filter this down to a specific term by using the `var_name` keyword. The below only plots the respondent's slope while ignoring the intercept:

```python
plot_trajectories(idata, time="time_point", var_name="latent.slope.contribution", coords={"respondent_id": 0}, latent="positive_affect")
```

Refer to the [guide on scoped parameter names](trace_names.md) to understand parameter naming schemes.

[Fitting Models](fitting.md) · [Static tutorial](../tutorials/tutorial_static.ipynb) · [Dynamic tutorial](../tutorials/tutorial_dynamic.ipynb)
