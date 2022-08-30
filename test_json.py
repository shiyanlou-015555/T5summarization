import json
data = json.load(open("/data1/ach/project/T5summarization/data/tagged/test.json",'r'))
res = data[:10]
with open('/data1/ach/project/T5summarization/data/tagged/test_ddp.json',"w",encoding="utf8") as f:
    json.dump(res,f,ensure_ascii=False)