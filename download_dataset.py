from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", cache_dir='data',version="3.0.0")
print(f"Features: {dataset['train'].column_names}")