from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data1/ach/project/T5summarization/model/bart-large-cnn")
# text = ["I","t5tokenizer"]
text = ["summarize: by. emily crane."]
tokenized_input = tokenizer(text)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
print(tokens)
# [21603, 10, 57, 5, 3, 15, 51, 9203, 25126, 5, 1]
# [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, None]