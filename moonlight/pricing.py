# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim for moonlight.pricing.

The model registry and cost functions have moved to :mod:`moonlight.llm`,
which covers all providers.  This module re-exports the same API so
existing code that imports from ``moonlight.pricing`` continues to work.

New code should import from ``moonlight.llm`` directly::

    from moonlight.llm import LLMClient, MODELS, model_id, cost
"""
from __future__ import annotations

from moonlight.llm import MODELS, cost, model_id

__all__ = ["MODELS", "model_id", "cost"]
