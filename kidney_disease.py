# =================================== ENTRY POINT ================================= #
# This file is kept for backwards compatibility.
# The application logic lives under app/

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.kidney_disease import app, server  # noqa: F401
