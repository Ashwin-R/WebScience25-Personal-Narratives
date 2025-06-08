import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class NarrativePredictor:
    def __init__(self, model_path):
        self.device = self._get_device()
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def _get_device(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def predict(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        prediction_idx = torch.argmax(outputs.logits, dim=-1).item()
        labels = ['Not a Personal Narrative', 'Personal Narrative']
        
        return {
            "prediction": labels[prediction_idx],
            "probabilities": {
                "Not a Personal Narrative": f"{probabilities[0]:.4f}",
                "Personal Narrative": f"{probabilities[1]:.4f}"
            }
        }

if __name__ == '__main__':
    # This path is now fixed to your specified model directory
    MODEL_PATH = '../trained_model'

    try:
        predictor = NarrativePredictor(model_path=MODEL_PATH)
    except OSError:
        print(f"Error: Model not found at '{MODEL_PATH}'.")
        print("Please run the training script `src/train_narrative_classifier.py` first.")
        exit()

    example_texts = [
        "I’m a cdl-a truck driver in the USA. I work 60 hours over 5 days.",
        "The GDP is expected to grow by 2% in the next quarter.",
        "I faked being a liberal for my years at university. They never suspected a thing.",
        "This approach, however, excludes an important group of individuals."
    ]
    
    for text in example_texts:
        result = predictor.predict(text)
        print(f"\nText: \"{text}\"")
        print(f"  -> Prediction: {result['prediction']}")
        print(f"  -> Probabilities: {result['probabilities']}")