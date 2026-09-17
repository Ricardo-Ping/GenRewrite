from pathlib import Path
from transformers import LongformerTokenizer, LongformerModel

MODEL_ID = "allenai/longformer-base-4096"
TARGET = Path(__file__).resolve().parents[1] / "config_file" / "longformer"

tokenizer = LongformerTokenizer.from_pretrained(MODEL_ID)
model = LongformerModel.from_pretrained(MODEL_ID)

TARGET.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(TARGET)
model.save_pretrained(TARGET)
print(f"saved to {TARGET}")