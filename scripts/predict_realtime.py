import os
import subprocess
import pandas as pd
import numpy as np
import json
import requests
import sys
from datetime import datetime, timezone

# --- 1. DYNAMIC FEATURE COLLECTION ---

def run_command(command):
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return ""

def make_api_request(endpoint):
    token = os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPOSITORY')
    if not token or not repo: return None
    url = f"https://api.github.com/repos/{repo}/{endpoint}"
    headers = {'Authorization': f'token {token}'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def get_detailed_churn():
    features = {
        'src_churn': 0, 'files_added': 0, 'files_deleted': 0, 'files_modified': 0,
        'test_churn': 0, 'tests_added': 0, 'tests_deleted': 0, 'tests_modified': 0
    }
    numstat = run_command("git diff --numstat HEAD~1 HEAD")
    status = run_command("git diff --name-status HEAD~1 HEAD")
    line_changes = {parts[2]: int(parts[0]) + int(parts[1]) for line in numstat.splitlines() if len(parts := line.split('\t')) == 3 and parts[0].isdigit()}
    
    for line in status.splitlines():
        change_type, path = line.split('\t')
        churn = line_changes.get(path, 0)
        is_test = 'test' in path.lower()
        
        if is_test:
            features['test_churn'] += churn
            if 'A' in change_type: features['tests_added'] += 1
            elif 'D' in change_type: features['tests_deleted'] += 1
            else: features['tests_modified'] += 1
        else:
            features['src_churn'] += churn
            if 'A' in change_type: features['files_added'] += 1
            elif 'D' in change_type: features['files_deleted'] += 1
            else: features['files_modified'] += 1
    return features

def get_sloc_with_cloc():
    features = {'sloc': 0, 'test_lines': 0}
    cloc_json_str = run_command("cloc . --json --quiet")
    if not cloc_json_str: return features
    try:
        data = json.loads(cloc_json_str)
        features['sloc'] = data.get('Java', {}).get('code', 0)
        # Add other languages if needed
    except json.JSONDecodeError: pass
    return features

def get_project_history():
    features = {'prev_pass': 1, 'elapsed_days_last_build': 0, 'project_fail_history': 0.0, 'project_fail_recent': 0.0, 'commit_interval': 0.0, 'project_age': 0}
    repo_data = make_api_request("")
    if repo_data and 'created_at' in repo_data:
        features['project_age'] = (datetime.now(timezone.utc) - datetime.fromisoformat(repo_data['created_at'].replace('Z', '+00:00'))).days
        
    runs_data = make_api_request(f"actions/runs?branch={os.environ.get('GITHUB_REF_NAME')}&per_page=10")
    if runs_data and runs_data.get('workflow_runs'):
        completed = [r for r in runs_data['workflow_runs'] if r['status'] == 'completed']
        if completed:
            features['prev_pass'] = 1 if completed[0]['conclusion'] == 'success' else 0
            outcomes = [1 if r['conclusion'] == 'success' else 0 for r in completed]
            features['project_fail_history'] = 1.0 - np.mean(outcomes)
            features['project_fail_recent'] = 1.0 - np.mean(outcomes[:5])
    return features

def get_live_features():
    print("Collecting dynamic features for the latest commit...", file=sys.stderr)
    features = {'team_size': 1, 'num_commit_comments': 0, 'committers': 1} # Start with sane defaults
    features.update(get_detailed_churn())
    features.update(get_sloc_with_cloc())
    features.update(get_project_history())
    return features

# --- 2. FEATURE ENGINEERING & SCRIPT EXECUTION ---

def create_engineered_features(df):
    print("Engineering features...", file=sys.stderr)
    # ... (This function remains the same as your provided image)
    epsilon = 1e-6
    df['test_lines_per_kloc'] = (df['test_lines'] / df['sloc'].replace(0, 1)) * 1000
    df['recent_fail_binned'] = pd.cut(df['project_fail_recent'], bins=[-0.1, 0.0, 0.3, 0.7, 1.0], labels=[0, 1, 2, 3], right=True).cat.codes
    df['historical_context'] = (df['project_fail_recent'] * 0.6 + df['project_fail_history'] * 0.4)
    # ... and so on for all engineered features
    return df

if __name__ == "__main__":
    # ... (This main execution block remains the same, saving the payload)
    raw_features = get_live_features()
    df_live = pd.DataFrame([raw_features])
    df_final = create_engineered_features(df_live)
    
    payload = df_final.to_dict(orient='records')[0]
    for key, value in payload.items():
        if isinstance(value, (np.integer, np.int64)): payload[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)): payload[key] = float(value)
            
    with open('payload.json', 'w') as f:
        json.dump(payload, f, indent=2)
        
    API_URL = os.environ.get("PREDICTION_API_URL")
    prediction = "Error"
    if not API_URL:
        print("Error: PREDICTION_API_URL secret not set.", file=sys.stderr)
    else:
        try:
            print(f"Sending request to: {API_URL}", file=sys.stderr)
            response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status()
            prediction = response.json().get("prediction", "Error")
        except requests.exceptions.RequestException as e:
            print(f"Error calling prediction API: {e}", file=sys.stderr)
    
    # Final output for the workflow
    with open('prediction.txt', 'w') as f:
        f.write(prediction)