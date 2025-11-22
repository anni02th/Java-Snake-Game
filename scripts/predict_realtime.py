import os
import subprocess
import pandas as pd
import numpy as np
import json
import requests
import sys
from datetime import datetime, timezone
import traceback

# --- 1. DYNAMIC FEATURE COLLECTION ---

def run_command(command):
    """Executes a shell command and returns its stdout."""
    try:
        # Added check=True to raise error on non-zero exit status
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
    Calculates aggregate churn features and detailed file-level changes.
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

def get_full_diff_limited(limit_lines=200):
    """
    Executes git diff (content of changes) and limits the output to 200 lines.
    """
    print(f"📝 Collecting first {limit_lines} lines of code changes...", file=sys.stderr)
    # 1. Get the full unified diff content (actual code changes)
    full_diff = run_command("git diff HEAD~1 HEAD")
    
    # 2. Split the output into lines and take only the first 'limit_lines'
    diff_lines = full_diff.splitlines()
    limited_diff_lines = diff_lines[:limit_lines]
    
    # 3. Join the limited lines back into a single string
    return "\n".join(limited_diff_lines)

def get_sloc_with_cloc():
    """Uses cloc to find Source Lines of Code for Java."""
    features = {'sloc': 0, 'test_lines': 0}
    # Running cloc in the root directory
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
    
    features['limited_diff_content'] = get_full_diff_limited(limit_lines=200) 
    
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

def request_prediction(payload, api_url):
    """Sends the feature payload to the prediction API and returns the result."""
    print(f"📡 Sending prediction request to: {api_url}", file=sys.stderr)
    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()

        result = response.json()
        prediction = result.get("prediction")

        if prediction is None:
            print("❌ 'prediction' field missing or null in response!", file=sys.stderr)
            print("🔎 Response content:", result, file=sys.stderr)
            return None

        # Write prediction to file for GitHub Actions output
        with open('prediction.txt', 'w') as f:
            f.write(str(prediction))

        print(f"✅ Prediction result: {prediction}")
        return prediction

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred: {http_err}", file=sys.stderr)
        print(f"🔎 Response content: {response.text}", file=sys.stderr)
        return None
    except Exception as err:
        print(f"❌ Unexpected error during prediction: {err}", file=sys.stderr)
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        return None

def request_explanation(payload, api_url, ml_prediction):
    """
    Sends the feature payload (including diff content) to the GenAI explanation API
    and returns the explanation report.
    """
    print(f"🧠 ML predicted '{ml_prediction}'. Requesting AI explanation from: {api_url}", file=sys.stderr)

    # Add the ML prediction result to the payload for context
    payload['ml_prediction_result'] = ml_prediction
    
    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()

        result = response.json()
        
        # Check for the expected output field (e.g., 'report' or 'explanation')
        explanation = result.get("report") or result.get("explanation")
        
        if explanation is None:
            print("❌ Explanation field ('report' or 'explanation') missing in response!", file=sys.stderr)
            print("🔎 Response content:", result, file=sys.stderr)
            return "Internal API Error: GenAI explanation field was missing."

        print("✅ Successfully received AI explanation.", file=sys.stderr)
        return explanation

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred during explanation call: {http_err}", file=sys.stderr)
        return f"API Failure (HTTP {response.status_code}): Could not get explanation."
    except Exception as err:
        print(f"❌ Unexpected error during explanation: {err}", file=sys.stderr)
        return f"Unexpected Error: {err}"


if __name__ == "__main__":
    
    # 1. Collect and Prepare Data
    raw_features = get_live_features()
    
    # Store diff content temporarily
    diff_content = raw_features.pop('limited_diff_content', '')
    
    # Convert to DataFrame for feature engineering
    df_live = pd.DataFrame([raw_features])
    df_final = create_engineered_features(df_live)

    # Convert DataFrame row back to a dictionary for JSON payload
    payload = df_final.to_dict(orient='records')[0]
    
    # Re-add the diff content and commit metadata to the final payload
    payload['limited_diff_content'] = diff_content
    payload['commit_sha'] = os.environ.get('GITHUB_SHA', 'unknown_sha')
    payload['author'] = os.environ.get('GITHUB_ACTOR', 'unknown_actor')
    payload['commit_message'] = os.environ.get('COMMIT_MESSAGE', 'No commit message available')
    
    # Clean up numpy types for JSON serialization
    for key, value in payload.items():
        if isinstance(value, (np.integer, np.int64)):
            payload[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            payload[key] = float(value)
        elif isinstance(value, (np.bool_)):
            payload[key] = bool(value)

    # Save the full payload (with diff) for debugging and artifact upload
    with open('payload.json', 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        print("\n✅ Payload generated and saved to payload.json", file=sys.stderr)

    # 2. Handle Execution Modes (Prediction or Explanation)
    
    # Get the execution mode passed as a command-line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else 'predict' 
    
    if mode == 'predict':
        # PREDICTION MODE: Called from ml-api and gen-ai-api jobs
        api_url = os.environ.get("PREDICTION_API_URL")
        if not api_url:
            print("❌ Error: PREDICTION_API_URL environment variable not set.", file=sys.stderr)
            sys.exit(1)
            
        prediction_endpoint = "/predict"
        full_api_url = f"{api_url}{prediction_endpoint}"
        request_prediction(payload, full_api_url)

    elif mode == 'explain':
        # EXPLANATION MODE: Called from notify-and-build job
        explain_api_url = os.environ.get("EXPLAIN_API_URL")
        ml_prediction = os.environ.get("ML_PREDICTION_RESULT")
        
        if not explain_api_url or not ml_prediction:
            print("❌ Error: EXPLAIN_API_URL or ML_PREDICTION_RESULT not set for explanation mode.", file=sys.stderr)
            sys.exit(1)
        
        # Request the explanation
        explanation_report = request_explanation(payload, explain_api_url, ml_prediction)
        
        # Write explanation to file for GitHub Actions output
        with open('explanation.txt', 'w') as f:
            f.write(explanation_report)
        
        print(f"✅ Explanation generation complete. Result saved to explanation.txt", file=sys.stderr)

    else:
        print(f"❌ Invalid mode provided: {mode}", file=sys.stderr)
        sys.exit(1)