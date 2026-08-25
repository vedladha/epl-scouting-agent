import pandas as pd
 
path = "data/kaggle_raw/Squad_PlayerStats__stats_standard.csv"
df = pd.read_csv(path)
 
print("shape:", df.shape)
print("\nALL columns:")
for c in df.columns:
    print(" -", c)
 
print("\nFirst 3 rows:")
print(df.head(3).to_string())
 