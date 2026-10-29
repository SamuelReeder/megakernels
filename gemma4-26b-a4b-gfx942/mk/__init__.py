"""Puts <repo>/common first on sys.path, so `import mkcore` loads this tree's copy
(common/DESIGN.md D3); every entry point imports mk first."""
import os as _os
import sys as _sys

_COMMON = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "common")
if _sys.path[:1] != [_COMMON]:
    _sys.path.insert(0, _COMMON)
import mkcore as _mkcore  # noqa: E402

assert _os.path.dirname(_os.path.abspath(_mkcore.__file__)) == _os.path.join(_COMMON, "mkcore"), (
    f"mkcore from {_mkcore.__file__}, not {_COMMON}"
)
