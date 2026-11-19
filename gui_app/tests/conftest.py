import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:                       # nếu dự án thật có sẵn core/ thì dùng; ngược lại dùng bản giả
    import core.phase_schema  # noqa: F401
except Exception:
    from gui_app.tests import fakes
    fakes.install()


import sys as _sys, traceback as _tb
ERRORS = []

def _hook(et, ev, tb):                 # PyQt5 abort nếu slot ném lỗi mà không có excepthook
    ERRORS.append("".join(_tb.format_exception(et, ev, tb)))
    print(ERRORS[-1])

_sys.excepthook = _hook
