"""Cardiac-specific UQ metric helpers.

Re-exported from the legacy ``uncertainty_sde_combined_acdc_v3.test_fast``
module so the metric internals (LV-seg propagation through warps, AHA
breakdown, Dice/HD95) stay in one place.
"""

from uncertainty_sde_combined_acdc_v3.test_fast import (
    _compute_sparsification_curves,
    _compute_risk_coverage_curves,
    _plot_avg_sparsification,
    _plot_avg_risk_coverage,
    compute_registration_metrics,
)

__all__ = [
    "_compute_sparsification_curves",
    "_compute_risk_coverage_curves",
    "_plot_avg_sparsification",
    "_plot_avg_risk_coverage",
    "compute_registration_metrics",
]
