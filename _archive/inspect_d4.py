import pandas as pd
import os

if os.path.exists('kaggle_esrd.csv'):
    df = pd.read_csv('kaggle_esrd.csv')
    print("D4 Columns List:")
    for c in df.columns:
        print(f"'{c}'")
else:
    print("D4 Not found!")
