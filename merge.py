import os
data = [0,1,2]
hypo  = []
target = []
for i in data:
    with open(os.path.join('/data1/ach/project/T5summarization/output/bart/1e-5-noprompt-2048/eval3','{}.hypo'.format(i)),'r') \
        as f:
        for idx in f.readlines():
            hypo.append(idx.strip())
    with open(os.path.join('/data1/ach/project/T5summarization/output/bart/1e-5-noprompt-2048/eval3','{}.target'.format(i)),'r') \
        as f:
        for idx in f.readlines():
            target.append(idx.strip())
with open(os.path.join('/data1/ach/project/T5summarization/output/bart/1e-5-noprompt-2048/eval3','test.hypo'),'w') \
    as f:
    for i in hypo:
        f.writelines(i+'\n')
with open(os.path.join('/data1/ach/project/T5summarization/output/bart/1e-5-noprompt-2048/eval3','test.target'),'w') \
    as f:
    for i in target:
        f.writelines(i+'\n')