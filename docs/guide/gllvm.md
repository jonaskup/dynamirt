# Generalized Linear Latent Variable Models (GLLVM)

Item response theory can be seen as a specialization and extension of generalized linear latent variable models (GLLVMs) to binary and ordinal data, where it's reasonable to think in terms of probabilities of endorsement. GLLVMs on the other hand can be used for any exponential family distribution and are useful, for example, for continuous (i.e., [factor analysis](https://en.wikipedia.org/wiki/Factor_analysis)) and count data. GLLVMs are prominently used in ecology where they're used for species abundance and occurence models (see the [gllvm](https://github.com/JenniNiku/gllvm/) R-package). There, spatially and/or temporally correlated latent traits are also often a reasonable assumption.

Because IRT and GLLVMs share a lot of the same building blocks, `dynamirt` is built on top of a GLLVM basis that is then extended with IRT functionality. However, the `GLLVM(...)` function can still be used to fit a range of non-IRT measurement models. Below is an example of fitting a negative-binomial model, e.g., for abundance data.

In constrast to the `dynamirt(...)` model builder, `GLLVM(...)` does not add any full-rank or latent residuals by default so they need to be added manually. Other than that, usage is much the same:

```python
from dynamirt import GLLVM, Confirmatory, Linear, fit_svi
from dynamirt.gllvm.families import NegBinom

model = GLLVM(
    n_latent=2,
    family=NegBinom(),
    loadings=Confirmatory(Q),
    item_terms=[Linear("intercept")],
    latent_terms=[Linear("site", group_by="obs_")],
)

result = fit_svi(model, responses)
idata = result.to_idata()
```

Models constructed with `GLLVM(...)` can be used with `fit_mcmc()` and `fit_svi()` just as models constructed with `dynamirt(...)`. The latent-to-full-rank measurement functions can also be inspected with `response_curves()`. For distributions with unbounded support, the conditional mean will be displayed instead of a category probability. However, the scoring functions (`log_loss()` and `ranked_probability_score()`) assume categorical support of the response family and will fail if that is not given.