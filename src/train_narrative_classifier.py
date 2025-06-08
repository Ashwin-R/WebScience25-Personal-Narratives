import pandas as pd
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import os

class NarrativeDataset(Dataset):
    """Custom PyTorch Dataset for our narrative data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # The tokenizer returns a dictionary-like object. We need to access its items.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class NarrativeClassifierTrainer:
    """
    A class to handle the training and evaluation of a narrative classification model.
    """
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self._set_seed(config['seed'])
        print(f"Using device: {self.device}")

    def _get_device(self):
        """Selects the best available device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _set_seed(self, seed):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    def _load_and_prepare_data(self):
        """Loads data, cleans labels, splits, and tokenizes."""
        df = pd.read_csv(self.config['data_path'])

        # The label column is 'tensor(1)' or 'tensor(0)', we need to extract the integer.
        # This handles the specific format you mentioned.
        df['label'] = df['label'].apply(lambda x: int(str(x).strip('tensor()')))

        X = df['comment'].tolist()
        y = df['label'].tolist()

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.config['seed'], stratify=y
        )

        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizer(X_val, truncation=True, padding=True, max_length=512)

        self.train_dataset = NarrativeDataset(train_encodings, y_train)
        self.val_dataset = NarrativeDataset(val_encodings, y_val)
        
        # Store validation texts and labels for classification report
        self.val_texts = X_val
        self.val_labels = y_val

    def train(self):
        """Main training and evaluation loop."""
        self._load_and_prepare_data()

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=2
        )
        model.to(self.device)

        train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        optimizer = AdamW(model.parameters(), lr=self.config['learning_rate'])
        total_steps = len(train_loader) * self.config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        print("Starting training...")
        for epoch in range(self.config['epochs']):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config['epochs']}", leave=False)

            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch + 1} | Average Training Loss: {avg_train_loss:.4f}")

            # Evaluate at the end of each epoch
            self.evaluate(model)

        print("Training complete.")
        self.save_model(model)

    def evaluate(self, model):
        """Evaluates the model and prints a classification report."""
        val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'])
        model.eval()
        
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        print("\n--- Classification Report ---")
        print(classification_report(true_labels, predictions, target_names=['Not a Narrative', 'Personal Narrative']))

    def save_model(self, model):
        """Saves the model and tokenizer to the specified directory."""
        output_dir = self.config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model and tokenizer saved successfully.")


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'model_name': 'falkne/storytelling-LM-europarl-mixed-en',
        'data_path': '../data/complete_training_data.csv',
        'output_dir': '../trained_model',
        'epochs': 10,
        'batch_size': 16, # Lowered a bit to be safer on memory
        'learning_rate': 5e-5,
        'seed': 42
    }

    # --- Run Training ---
    trainer = NarrativeClassifierTrainer(config)
    trainer.train()