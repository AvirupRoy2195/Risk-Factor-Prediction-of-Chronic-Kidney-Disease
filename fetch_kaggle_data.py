import kagglehub
import pandas as pd
import os
import shutil

def download_and_inspect():
    print("⬇️ Downloading D1: mansoordaku/ckdisease...")
    path1 = kagglehub.dataset_download("mansoordaku/ckdisease")
    
    print("\n⬇️ Downloading D2: miadul/kidney-function-health-dataset...")
    path2 = kagglehub.dataset_download("miadul/kidney-function-health-dataset")
    
    print("\n⬇️ Downloading D3: himelsarder/kidney-dataset...")
    path3 = kagglehub.dataset_download("himelsarder/kidney-dataset")
    
    print("\n⬇️ Downloading D4: s3programmerlead/end-stage-renal-disease-esrd-dataset...")
    path4 = kagglehub.dataset_download("s3programmerlead/end-stage-renal-disease-esrd-dataset")
    print(f"✅ D4 downloaded to: {path4}")
    
    # Inspect D3 & D4
    csv_file_3 = None
    # ... (existing D3 logic) ...

    # Inspect D4
    for root, dirs, files in os.walk(path4):
        for file in files:
            if file.endswith(".csv"):
                print(f"📂 Found D4 CSV: {file}")
                shutil.copy(os.path.join(root, file), "kaggle_esrd.csv")
                
    if os.path.exists("kaggle_esrd.csv"):
        df4 = pd.read_csv("kaggle_esrd.csv")
        print("\n📊 D4 Info:")
        print(df4.info())
        print(df4.head())
    for root, dirs, files in os.walk(path3):
        for file in files:
            if file.endswith(".csv"):
                csv_file_3 = os.path.join(root, file)
                # Keep looking if there are multiple (e.g. Test/Train)
                print(f"📂 Found D3 File: {file}")
                
                # Check for test in name
                if 'test' in file.lower():
                    print("🎯 Found Explicit Test Set!")
                    shutil.copy(os.path.join(root, file), "kaggle_d3_test.csv")
                else:
                    csv_file_3 = os.path.join(root, file)
                    shutil.copy(os.path.join(root, file), "kaggle_d3_train.csv") # Assume train if not test

    if os.path.exists("kaggle_d3_train.csv"):
        df3 = pd.read_csv("kaggle_d3_train.csv")
        print("\n📊 D3 Info:")
        print(df3.info())
        print("\n👀 D3 Head:")
        print(df3.head())
        # Check for GFR
        gfr_col = [c for c in df3.columns if 'gfr' in c.lower()]
        print(f"🧪 GFR Columns found: {gfr_col}")

if __name__ == "__main__":
    download_and_inspect()
