import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from collections import OrderedDict

# --- Configuration ---
# The base model used for fine-tuning. This is crucial.
BASE_MODEL = "falkne/storytelling-LM-europarl-mixed-en" 

# The path to the saved .pth state dictionary file.
PTH_MODEL_PATH = '../models/final_model.pth'

# The directory where you want to save the converted, Hugging Face-compatible model.
OUTPUT_DIR = '../trained_model' 

# --- Conversion Logic ---

def convert_pth_to_hf_format(base_model, pth_path, output_dir):
    """
    Loads a model's state_dict from a .pth file, cleans the keys if necessary,
    and saves the full model and tokenizer in the Hugging Face `save_pretrained` format.
    """
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model .pth file not found at: {pth_path}")

    print(f"Loading base model architecture: {base_model}")
    # 1. Load the model architecture and tokenizer from the base model
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading state dictionary from: {pth_path}")
    # 2. Load the state dictionary (the fine-tuned weights)
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))

    # 3. Create a new state_dict without the 'module.' prefix
    # This is necessary if the model was saved using torch.nn.DataParallel
    print("Checking for 'module.' prefix in state_dict keys...")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v # If no 'module.' prefix, keep the key as is
    
    print("Prefixes removed. Loading the cleaned state_dict into the model.")
    # 4. Load the corrected weights into the model architecture
    model.load_state_dict(new_state_dict)
    print("Successfully loaded weights into the model.")

    # 5. Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving model and tokenizer to: {output_dir}")
    # 6. Save the complete model and tokenizer in the standard format
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Conversion complete. Your model is now ready for use and upload.")


if __name__ == '__main__':
    # Adjust the paths in the Configuration section above if needed.
    convert_pth_to_hf_format(BASE_MODEL, PTH_MODEL_PATH, OUTPUT_DIR)