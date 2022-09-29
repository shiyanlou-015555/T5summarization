import json
import os
from rouge_score import rouge_scorer
from tqdm import tqdm
dirs = os.listdir('./cnn_cln')
os.makedirs('./cnn_cln_tagged', exist_ok=True)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
rouges = []

for fileName in ['train', 'test', 'val']:
    rouges.append([])
    fileSrc = open('/data1/gj/cnn_cln/cnn_cln_tagged/' + fileName + '.hypo.tokenized', 'r', encoding='utf8')
    fileTrg = open('/data1/gj/cnn_cln/cnn_cln_tagged/' + fileName + '.hypo.target', 'r', encoding='utf8')
    srcs = fileSrc.readlines()
    trgs = fileTrg.readlines()
    for i in tqdm(range(len(srcs)), desc=fileName):
        allRouge = scorer.score(trgs[i], srcs[i])
        rouges[-1].append(allRouge)

        
        
for dir in dirs:
    if dir.endswith('json'):
        fileHandler = open('./cnn_cln/'+dir,'r', encoding='utf8')
        items = json.loads(fileHandler.read())
        if dir.startswith('train'):
            allRouge = rouges[0]
        elif dir.startswith('test'):
            allRouge = rouges[1]
        elif dir.startswith('val'):
            allRouge = rouges[2]
            
        for idx, item in tqdm(enumerate(items), desc=dir):
            myRouge = allRouge[idx]
            for k, v in myRouge.items():
                items[idx][k] = v
                
        fileHandler.close()
        fileHandler = open('./cnn_cln_tagged/'+dir,'w', encoding='utf8')
        fileHandler.write(json.dumps(items))
        fileHandler.close()
    else:
        continue

print('succeed')