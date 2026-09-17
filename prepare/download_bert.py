from pathlib import Path
from transformers import BertTokenizer, BertModel

MODEL_ID = "bert-base-uncased"
TARGET = Path(__file__).resolve().parents[1] / "config_file" / "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
model = BertModel.from_pretrained(MODEL_ID)

TARGET.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(TARGET)
model.save_pretrained(TARGET)
print(f"saved to {TARGET}")
