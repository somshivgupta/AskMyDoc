# save as download_models.py and run it
import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"

print("Downloading sentence transformer...")
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Done!")

print("Downloading flan-t5-base...")
from transformers import pipeline
pipeline("text2text-generation", model="google/flan-t5-base")
print("✅ Done!")