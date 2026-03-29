"""
ASG 04 – Step 1: Data Ingestion
Reads raw train.csv and saves it to the ingested/ folder.
"""

from pathlib import Path
import pandas as pd

BASE_DIR     = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE   = BASE_DIR.parent / "data/raw/train.csv" 
OUTPUT_FILE  = INGESTED_DIR / "spaceship_train.csv"


def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return

    df = pd.read_csv(INPUT_FILE)

    assert not df.empty, "empty"

    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"berhasil!")
    print(f"Sumber: {INPUT_FILE}")
    print(f"Tujuan: {OUTPUT_FILE}")


if __name__ == "__main__":
    ingest_data()