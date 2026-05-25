# SPDX-License-Identifier: Apache-2.0
"""Shared rate-limiter for the moonlight web API."""
from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

TRANSLATE_DAILY_CAP_USD = float(
    os.environ.get("MOONLIGHT_TRANSLATE_DAILY_CAP_USD", "5.0")
)

limiter = Limiter(key_func=get_remote_address)
