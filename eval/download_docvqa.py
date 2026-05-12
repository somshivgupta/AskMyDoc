from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceM4/DocumentVQA",
    split="validation",
    streaming=True
)

samples = list(dataset.take(50))

print(f"Loaded   : {len(samples)} samples")
print(f"Fields   : {list(samples[0].keys())}")
print(f"Question : {samples[0]['question']}")
print(f"Answers  : {samples[0]['answers']}")