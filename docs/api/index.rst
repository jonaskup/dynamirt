API reference
=============

Model construction
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dynamirt.dynamirt

Fitting and results
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dynamirt.fit_mcmc
   dynamirt.fit_svi
   dynamirt.fit.FitResult
   dynamirt.fit.SVIState

Terms
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   dynamirt.Linear
   dynamirt.GRW
   dynamirt.AR1
   dynamirt.GP
   dynamirt.HSGP
   dynamirt.CustomTerm
   dynamirt.ModelContext

.. _api-loading-structures:

Loading structures
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dynamirt.Full
   dynamirt.Fixed
   dynamirt.Confirmatory
   dynamirt.Sparsity
   dynamirt.q_matrix

Parameters
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dynamirt.Param
   dynamirt.Pool

Plots
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   dynamirt.plot_trajectories
   dynamirt.plot_loadings
   dynamirt.response_curves
   dynamirt.plot_response_curves

Scoring
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dynamirt.log_loss
   dynamirt.ranked_probability_score

Experimental calibration
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dynamirt.calibration_curves.calibration
   dynamirt.calibration_curves.plot_calibration
