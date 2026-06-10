import sys
from pathlib import Path
PICASSO_OUTER_DIR = Path(__file__).resolve().parent / "picasso"
if str(PICASSO_OUTER_DIR) not in sys.path:
    sys.path.insert(0, str(PICASSO_OUTER_DIR))
    from src.cli import app
if __name__ == "__main__":
    app()