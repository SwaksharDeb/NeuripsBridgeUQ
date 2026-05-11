"""Brain-specific UQ metric helpers.

These are re-exported from the legacy ``uncertainty_brain_sde_v3.test_fast``
module so we have a single source of truth for the metric implementations
(brain uses different segmentation conventions than cardiac, so they cannot
be unified).
"""

from uncertainty_brain_sde_v3.test_fast import (
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
