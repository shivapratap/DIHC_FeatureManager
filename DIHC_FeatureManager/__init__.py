"""Public package exports for DIHC_FeatureManager."""

from .DIHC_FeatureManager import DIHC_FeatureManager

# Keep these available for existing user code without triggering wildcard
# circular imports inside the package modules.
try:
    from .DIHC_FeatureDetails import *  # noqa: F401,F403
except ImportError:
    pass

__all__ = ["DIHC_FeatureManager"]
