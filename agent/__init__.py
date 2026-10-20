"""The analyst agent over the data lake. See WALKTHROUGH.md.

The agent reuses code from extraction/ (llm.py, qa_tools.py, qa_agent.py), which
imports its neighbours by plain name (`import llm`). Putting extraction/ at the END
of sys.path makes those imports work without letting extraction's graph.py or
evaluate.py shadow ours.
"""

import os
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]        # the repo root (or /app in the container)
_EXTRACTION = str(ML / "extraction")
if _EXTRACTION not in sys.path:
    sys.path.append(_EXTRACTION)
# extraction/common.py imports mlflow, which prints an install hint on import.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
