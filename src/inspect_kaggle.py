import pandas as pd
from pathlib import Path

raw_dir = Path("data/kaggle_raw")
csvs = list(raw_dir.rglob("*.csv"))

if not csvs:
    print(f"No CSVs found in {raw_dir}. Did the download/unzip work?")
else:
    for path in csvs:
        print(f"\n=== {path} ===")
        try:
            df = pd.read_csv(path, nrows=5)
            print("columns:", list(df.columns))
            print(df.head(2))
        except Exception as e:
            print("FAILED to read:", e)