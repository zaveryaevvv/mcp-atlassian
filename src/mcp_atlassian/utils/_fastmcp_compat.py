"""Patch fastmcp <2.14 for Pydantic compatibility with ctx parameter exclusion.

fastmcp 2.13.0–2.13.2's ``create_function_without_params`` removes *ctx*
from ``__annotations__`` but **not** from ``__signature__``.  Newer Pydantic
(≥ 2.12) iterates ``inspect.signature().parameters`` and then looks the
name up in ``get_type_hints()``, which no longer contains *ctx* → KeyError.

fastmcp 2.13.0.2 (corporate mirror build) refactored this entirely — it uses
``ParsedFunction`` / ``compress_schema`` and never modifies the function
signature, so the patch is not needed and ``create_function_without_params``
no longer exists.  This module applies the patch only when the symbol is
present, making it safe for both build lines.

Import this module **before** any ``@mcp.tool()`` decorators execute.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import fastmcp.tools.tool as _tool_mod
import fastmcp.utilities.types as _types_mod

_original = getattr(_types_mod, "create_function_without_params", None)

if _original is not None:
    # fastmcp 2.13.0–2.13.2: patch needed — ctx left in __signature__
    def _patched_create_function_without_params(
        fn: Callable[..., Any], exclude_params: list[str]
    ) -> Callable[..., Any]:
        new_func = _original(fn, exclude_params)
        sig = inspect.signature(fn)
        new_params = [
            p for name, p in sig.parameters.items() if name not in exclude_params
        ]
        new_func.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            new_params, return_annotation=sig.return_annotation
        )
        return new_func

    # Patch both the canonical location and the already-imported local binding
    # (``from … import`` creates a separate reference in tool.py).
    _types_mod.create_function_without_params = (  # type: ignore[attr-defined]
        _patched_create_function_without_params
    )
    _tool_mod.create_function_without_params = (  # type: ignore[attr-defined]
        _patched_create_function_without_params
    )
# else: fastmcp 2.13.0.2+ — ParsedFunction handles ctx via compress_schema,
#       no signature mutation, no patch required.
