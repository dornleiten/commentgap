"""Isolated legacy runtime for released AQuA adapters.

The main CommentGap environment may import :mod:`aqua_runtime.schema`, but
model dependencies are imported only inside :mod:`aqua_runtime.model`.
"""

from .schema import AQUA_FEATURES, AQUA_SCHEMA_VERSION

__all__ = ["AQUA_FEATURES", "AQUA_SCHEMA_VERSION"]
