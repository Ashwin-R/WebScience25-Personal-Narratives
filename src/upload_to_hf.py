import os
from huggingface_hub import HfApi, HfFolder, create_repo, whoami

def upload_model():
    """
    Uploads the contents of the model directory to the Hugging Face Hub.
    
    This script assumes you have manually created a README.md file
    in the MODEL_DIR before running.
    """
    # --- Configuration ---
    HF_USERNAME = "tejasvichebrolu"
    
    MODEL_NAME = "personal-narrative-classifier"
    REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
    MODEL_DIR = "../trained_model"

    # Check that the model directory exists
    if not os.path.isdir(MODEL_DIR):
        print(f"Error: Model directory not found at '{MODEL_DIR}'.")
        print("Please run the training script first.")
        return

    # Check that the README.md file exists in the model directory
    readme_path = os.path.join(MODEL_DIR, "README.md")
    if not os.path.exists(readme_path):
        print(f"Error: README.md not found in '{MODEL_DIR}'.")
        print("Please create the README.md file in that directory before uploading.")
        return

    # 1. Check if user is logged in
    try:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError("Login required.")
        user_info = whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception:
        print("Hugging Face token not found. Please log in first.")
        print("Run this in your terminal: huggingface-cli login")
        return

    # 2. Create a repository on the Hub
    print(f"Creating repository '{REPO_ID}' on the Hub (if it doesn't exist)...")
    create_repo(repo_id=REPO_ID, exist_ok=True)
    
    # 3. Upload the model files
    api = HfApi()
    print(f"Uploading files from '{MODEL_DIR}' to '{REPO_ID}'...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Model upload from WebSci'25 paper"
    )
    
    print("\n--- Upload Complete! ---")
    print(f"Your model is now available at: https://huggingface.co/{REPO_ID}")

if __name__ == '__main__':
    upload_model()