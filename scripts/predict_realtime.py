import os
import subprocess
import pandas as pd
import numpy as np
import json
import requests
import sys
import time # Added for logging duration
from datetime import datetime, timezone
import traceback # Added for detailed error logging

# --- 1. DYNAMIC FEATURE COLLECTION ---

def run_command(command):
    """Executes a shell command and returns its stdout."""
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}\n{e.stderr}", file=sys.stderr)
        return ""

def make_api_request(endpoint):
    """Makes an authenticated request to the GitHub API."""
    token = os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPOSITORY')
    if not token or not repo:
        print("Error: GITHUB_TOKEN or GITHUB_REPOSITORY not set.", file=sys.stderr)
        return None
    
    # Correctly build the URL
    base_url = f"https://api.github.com/repos/{repo}"
    url = f"{base_url}/{endpoint}" if endpoint else base_url
    
    headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}", file=sys.stderr)
        return None
        
def get_detailed_churn():
    """
    Calculates aggregate churn features and NEW detailed file-level changes.
    """
    features = {
        'src_churn': 0, 'files_added': 0, 'files_deleted': 0, 'files_modified': 0,
        'test_churn': 0, 'tests_added': 0, 'tests_deleted': 0, 'tests_modified': 0,
        'changed_files_details': []
    }
    
    # Get churn (lines added/deleted) per file
    numstat = run_command("git diff --numstat HEAD~1 HEAD")
    # Get change type (A, M, D, R) per file
    status = run_command("git diff --name-status HEAD~1 HEAD")

    # Create a lookup for {path: churn_amount}
    line_changes = {}
    for line in numstat.splitlines():
        parts = line.split('\t')
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            line_changes[parts[2]] = int(parts[0]) + int(parts[1])

    # Process each file's status
    for line in status.splitlines():
        parts = line.split('\t')
        if not parts:
            continue
        
        change_type = parts[0]
        path = parts[-1] 
        
        churn = line_changes.get(path, 0)
        is_test = 'test' in path.lower()

        # Store detailed file info
        file_detail = {
            "path": path,
            "change_type": change_type,
            "churn": churn
        }
        features['changed_files_details'].append(file_detail)

        # Aggregate logic
        if is_test:
            features['test_churn'] += churn
            if 'A' in change_type or 'R' in change_type:
                features['tests_added'] += 1
            elif 'D' in change_type:
                features['tests_deleted'] += 1
            else:
                features['tests_modified'] += 1
        else:
            features['src_churn'] += churn
            if 'A' in change_type or 'R' in change_type:
                features['files_added'] += 1
            elif 'D' in change_type:
                features['files_deleted'] += 1
            else:
                features['files_modified'] += 1
                
    return features

def get_sloc_with_cloc():
    """Uses cloc to find Source Lines of Code for Java."""
    features = {'sloc': 0, 'test_lines': 0}
    cloc_json_str = run_command("cloc . --json --quiet")
    if not cloc_json_str:
        print("cloc command failed or returned empty.", file=sys.stderr)
        return features
    try:
        data = json.loads(cloc_json_str)
        if 'Java' in data:
            features['sloc'] = data['Java'].get('code', 0)
    except json.JSONDecodeError:
        print("Failed to decode cloc JSON output.", file=sys.stderr)
    return features

def get_project_history():
    """Gets repository history features from the GitHub API."""
    features = {
        'prev_pass': 1,
        'elapsed_days_last_build': 0,
        'project_fail_history': 0.0,
        'project_fail_recent': 0.0,
        'commit_interval': 0.0,
        'project_age': 0
    }
    
    # Get Project Age
    repo_data = make_api_request("")
    if repo_data and 'created_at' in repo_data:
        try:
            created_dt = datetime.fromisoformat(repo_data['created_at'].replace('Z', '+00:00'))
            features['project_age'] = (datetime.now(timezone.utc) - created_dt).days
        except ValueError:
            print("Could not parse project 'created_at' date.", file=sys.stderr)

    # Get Workflow Run History
    branch_name = os.environ.get('GITHUB_REF_NAME')
    if not branch_name:
        print("GITHUB_REF_NAME not set. Cannot fetch build history.", file=sys.stderr)
        return features

    runs_data = make_api_request(f"actions/runs?branch={branch_name}&per_page=10&status=completed")
    
    if runs_data and runs_data.get('workflow_runs'):
        completed = runs_data['workflow_runs']
        if completed:
            # Get last build status
            features['prev_pass'] = 1 if completed[0]['conclusion'] == 'success' else 0
            
            # Get last build date
            try:
                last_run_dt = datetime.fromisoformat(completed[0]['updated_at'].replace('Z', '+00:00'))
                features['elapsed_days_last_build'] = (datetime.now(timezone.utc) - last_run_dt).days
            except ValueError:
                print("Could not parse last run 'updated_at' date.", file=sys.stderr)

            # Calculate failure rates
            outcomes = [1 if r['conclusion'] == 'success' else 0 for r in completed]
            if outcomes:
                features['project_fail_history'] = 1.0 - np.mean(outcomes)
                features['project_fail_recent'] = 1.0 - np.mean(outcomes[:5])
                
    return features

def get_live_features():
    """Collects all dynamic features for the current commit."""
    print("📊 Collecting dynamic features for the latest commit...", file=sys.stderr)
    features = {
        'team_size': 1,
        'num_commit_comments': 0,
        'committers': 1
    }
    features.update(get_detailed_churn())
    features.update(get_sloc_with_cloc())
    features.update(get_project_history())
    return features

# --- 2. FEATURE ENGINEERING & EXECUTION ---

def create_engineered_features(df):
    """Engineers new features from the raw collected data."""
    print("🛠️ Engineering features...", file=sys.stderr)
    
    df['test_lines_per_kloc'] = (df['test_lines'] / (df['sloc'] + 1e-6)) * 1000
    
    df['recent_fail_binned'] = pd.cut(df['project_fail_recent'], 
                                     bins=[-0.1, 0.0, 0.3, 0.7, 1.0], 
                                     labels=[0, 1, 2, 3], 
                                     right=True).cat.codes
    
    df['historical_context'] = (df['project_fail_recent'] * 0.6 + df['project_fail_history'] * 0.4)
    
    return df

if __name__ == "__main__":
    start_time = time.time()
    
    raw_features = get_live_features()
    
    df_live = pd.DataFrame([raw_features])
    df_final = create_engineered_features(df_live)

    # Convert DataFrame row back to a dictionary for JSON payload
    payload = df_final.to_dict(orient='records')[0]
    
    # Clean up numpy types for JSON serialization
    for key, value in payload.items():
        if isinstance(value, (np.integer, np.int64)):
            payload[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            payload[key] = float(value)
        elif isinstance(value, (np.bool_)):
            payload[key] = bool(value)

    # Save the payload for debugging and artifact upload
    with open('payload.json', 'w') as f:
        json.dump(payload, f, indent=2)
        print("\n✅ Payload generated:\n", json.dumps(payload, indent=2), file=sys.stderr)

    # --- CRITICAL FIX: URL Construction ---
    API_URL_BASE = os.environ.get("PREDICTION_API_URL")
    ENDPOINT = "/predict"
    
    if not API_URL_BASE:
        print("❌ Error: PREDICTION_API_URL environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    FULL_API_URL = f"{API_URL_BASE}{ENDPOINT}"
    # --- END CRITICAL FIX ---

    try:
        print(f"📡 Sending request to: {FULL_API_URL}", file=sys.stderr)
        
        # Send the request
        response = requests.post(
            FULL_API_URL, 
            headers={'Content-Type': 'application/json'}, 
            json=payload
        )
        response.raise_for_status() # Raises HTTPError for bad status codes (4xx or 5xx)

        result = response.json()
        prediction = result.get("prediction", None)

        if prediction is None:
            print("❌ 'prediction' field missing or null in response!", file=sys.stderr)
            print("🔎 Response content:", result, file=sys.stderr)
            sys.exit(1)

        # Write prediction to file for GitHub Actions output
        with open('prediction.txt', 'w') as f:
            f.write(str(prediction))

        print(f"✅ Prediction result: {prediction}")
        
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred: {http_err}", file=sys.stderr)
        print(f"🔎 Response URL: {FULL_API_URL}", file=sys.stderr)
        print(f"🔎 Response content: {response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"❌ Unexpected error: {err}", file=sys.stderr)
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✨ Script completed in {time.time() - start_time:.2f} seconds.")