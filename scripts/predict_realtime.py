import os
import subprocess
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timezone
import sys 

# --- Your feature collection functions ---
def run_command(command):
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return ""

def get_live_features():
    print("Collecting features for the latest commit...", file=sys.stderr)
    features = {
        'src_churn': 150, 'files_added': 3, 'files_deleted': 1, 'files_modified': 5,
        'test_churn': 45, 'tests_added': 1, 'tests_deleted': 0, 'tests_modified': 2,
        'team_size': 10, 'sloc': 25000, 'test_lines': 5000, 'num_commit_comments': 0,
        'committers': 1, 'prev_pass': 1, 'elapsed_days_last_build': 2,
        'project_fail_history': 0.15, 'project_fail_recent': 0.20,
        'commit_interval': 24.5, 'project_age': 730
    }
    return features

def safe_division(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        result[~np.isfinite(result)] = 0
    return result

def create_engineered_features(df):
    print("Engineering features...", file=sys.stderr) # <-- FIXED
    epsilon = 1e-6
    df['test_lines_per_kloc'] = (df['test_lines'] / df['sloc'].replace(0, 1)) * 1000
    df['recent_fail_binned'] = pd.cut(df['project_fail_recent'], bins=[-0.1, 0.0, 0.3, 0.7, 1.0], labels=[0, 1, 2, 3], right=True).cat.codes
    df['historical_context'] = (df['project_fail_recent'] * 0.6 + df['project_fail_history'] * 0.4)
    df['change_intensity'] = safe_division(df['src_churn'].values, df['sloc'].values + epsilon)
    df['development_velocity'] = safe_division(1, df['commit_interval'].values + 1)
    df['test_coverage_gap'] = (100 - np.clip(df['test_lines_per_kloc'].values, 0, 100)) / 100
    df['build_frequency'] = safe_division(1, df['elapsed_days_last_build'].values + 1)
    return df

if __name__ == "__main__":
    # 1. Collect and engineer features
    raw_features = get_live_features()
    df_live = pd.DataFrame([raw_features])
    df_final = create_engineered_features(df_live)

    # 2. Prepare and save the payload for logging
    payload = df_final.to_dict(orient='records')[0]
    # Convert numpy types to native Python types for JSON serialization
    for key, value in payload.items():
        if isinstance(value, np.integer):
            payload[key] = int(value)
        elif isinstance(value, np.floating):
            payload[key] = float(value)
            
    with open('payload.json', 'w') as f:
        json.dump(payload, f, indent=2)


    API_URL = os.environ.get("PREDICTION_API_URL")
    if not API_URL:
        print("Error: PREDICTION_API_URL secret not set.", file=sys.stderr) # <-- FIXED
        prediction_result = "Error"
    else:
        try:
            payload = df_final.to_dict(orient='records')[0]
            headers = {'Content-Type': 'application/json'}
            
            print(f"Sending request to: {API_URL}", file=sys.stderr) # <-- FIXED
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            prediction_result = response.json().get("prediction", "Error")

        except requests.exceptions.RequestException as e:
            print(f"Error calling prediction API: {e}", file=sys.stderr) # <-- FIXED
            prediction_result = "Error"

    # This is the ONLY print to stdout, which is correct.
    print(prediction_result)