import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("HOME_CREDIT_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
BRONZE_DIR   = PROJECT_ROOT / "data" / "bronze"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "output"


print(PROJECT_ROOT)