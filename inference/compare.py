import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def load_and_merge_data(feature_path, clinical_csv_path, target_col):
    """
    Loads features from .npz and targets from CSV. 
    Returns aligned X (features) and y (labels).
    """
    if not os.path.exists(feature_path):
        print(f"Error: File not found {feature_path}")
        return None, None
        
    data = np.load(feature_path)
    ids = data['ids'].astype(str)
    X_raw = data['features'] 

    df = pd.read_csv(clinical_csv_path)
    df.columns = df.columns.str.strip()

    # Robust ID finding
    for col in df.columns:
        if col.lower() == 'id':
            df = df.rename(columns={col: 'ID'})
            break
            
    if 'ID' not in df.columns:
        print(f"ERROR: Could not find 'ID' column. Found columns: {df.columns.tolist()}")
        return None, None

    df['ID'] = df['ID'].astype(str)
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in CSV.")
        return None, None
        
    df = df.dropna(subset=[target_col])
    label_map = df.set_index('ID')[target_col].to_dict()

    X = []
    y = []
    
    for i, pid in enumerate(ids):
        if pid in label_map:
            val = label_map[pid]
            if not np.isnan(val):
                X.append(X_raw[i])
                y.append(val)
    
    return np.array(X), np.array(y)

def evaluate_features(feature_path, clinical_csv, target_col):
    """
    Runs Linear Probing and returns (score, std, metric_name).
    """
    # 1. Prepare Data
    X, y = load_and_merge_data(feature_path, clinical_csv, target_col)
    
    if X is None or len(X) == 0:
        return 0.0, 0.0, "No Data"

    # 2. Determine Task Type
    unique_vals = np.unique(y)
    
    # If fewer than 10 unique values, treat as Classification (Surgery 0/1, OARSI 0-3)
    is_classification = len(unique_vals) < 10 

    if is_classification:
        # Multi-class AUC (One-vs-Rest) or Binary AUC
        metric_name = "AUC"
        metric_sklearn = "roc_auc_ovr" if len(unique_vals) > 2 else "roc_auc"
        
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        # Continuous Regression (Pain Score 0-100)
        metric_name = "R^2"
        metric_sklearn = "r2"
        clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Run Cross-Validation
    scores = cross_val_score(clf, X, y, cv=cv, scoring=metric_sklearn)
    
    return scores.mean(), scores.std(), metric_name

def tee_print(message, file_handle=None):
    print(message)
    if file_handle:
        file_handle.write(message + "\n")

def print_comparison(task_name, target_col, path_normal, path_pain, clinical_csv, file_handle):
    tee_print(f"\n--- {task_name} ({target_col}) ---", file_handle)
    
    # Get results
    score_norm, std_norm, metric = evaluate_features(path_normal, clinical_csv, target_col)
    score_pain, std_pain, _      = evaluate_features(path_pain, clinical_csv, target_col)
    
    # Calculate Improvement
    diff = score_pain - score_norm
    if abs(score_norm) > 1e-9:
        pct_change = (diff / abs(score_norm)) * 100
    else:
        pct_change = 0.0

    # Print Table
    tee_print(f"Metric used: {metric}", file_handle)
    tee_print(f"Normal AE:   {score_norm:.4f} (+/- {std_norm:.4f})", file_handle)
    tee_print(f"Pain AE:     {score_pain:.4f} (+/- {std_pain:.4f})", file_handle)
    
    sign = "+" if diff >= 0 else ""
    tee_print(f"Improvement: {sign}{diff:.4f} ({sign}{pct_change:.2f}%)", file_handle)

def main():
    clinical_csv = "csv/clinical00_cleaned.csv"
    
    # Paths to your features
    path_normal = "results/autoencoder/features/features_left.npz"
    path_pain   = "results/autoencoder_pain/features/features_left.npz"
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "compare_results.txt")

    print(f"Running comparison... (Results will be saved to {output_file})")

    # Open the file for writing
    with open(output_file, "w") as f:
        tee_print("="*60, f)
        tee_print("LINEAR PROBING COMPARISON: Normal AE vs. Pain AE", f)
        tee_print("="*60, f)

        # --- TASK 1: SURGERY PREDICTION ---
        print_comparison(
            "Task 1: Surgery Prediction", 
            "V99ELKVSAF", 
            path_normal, 
            path_pain, 
            clinical_csv,
            f
        )

        # --- TASK 2: PAIN PREDICTION ---
        print_comparison(
            "Task 2: Pain Prediction", 
            "V00KOOSKPL", 
            path_normal, 
            path_pain, 
            clinical_csv,
            f
        )

        # --- TASK 3: JOINT SPACE NARROWING (OARSI) ---
        print_comparison(
            "Task 3: Joint Space Narrowing", 
            "V00XRJSM_L", 
            path_normal, 
            path_pain, 
            clinical_csv,
            f
        )

        tee_print("="*60, f)
    
    print(f"\nDone! Results saved to: {output_file}")

if __name__ == "__main__":
    main()