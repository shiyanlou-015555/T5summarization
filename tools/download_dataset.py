from datasets import load_dataset

dataset = load_dataset("xsum", cache_dir='/data1/ach/project/T5summarization/data_xsum')
print(f"Features: {dataset['train'].column_names}")
print(len(dataset['train']))