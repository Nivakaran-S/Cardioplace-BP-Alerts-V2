"""Put the repository root on sys.path for pytest.

pytest inserts the rootdir when a conftest.py lives there, which is what lets `import src`
resolve from inside tests/. The test files carry their own bootstrap as well, so they work
when executed directly -- CI runs them both ways.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
