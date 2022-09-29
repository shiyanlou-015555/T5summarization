from datasets import load_dataset
import json
dataset = load_dataset("/data1/ach/project/T5summarization/data_xsum")
train = []
for i in dataset['train']:
    temp = {}
    temp['text'] = i['article']
    temp['summary'] = i['highlights']
    train.append(temp)
val = []
for i in dataset['validation']:
    temp = {}
    temp['text'] = i['article']
    temp['summary'] = i['highlights']
    val.append(temp)
test = []
for i in dataset['test']:
    temp = {}
    temp['text'] = i['article']
    temp['summary'] = i['highlights']
    test.append(temp)
print('train is {}, val is {},test is {}'.format(len(train),len(val),len(test)))
with open('./train.json',"w",encoding="utf8") as f:
    json.dump(train,f,ensure_ascii=False)
with open('./val.json',"w",encoding="utf8") as f:
    json.dump(val,f,ensure_ascii=False)
with open('./test.json',"w",encoding="utf8") as f:
    json.dump(test,f,ensure_ascii=False)