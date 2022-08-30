import os
file_num = ['0','1','2','3']
# file_type = [".target",".hypo"]
target = []
hypo = []
data_dir = "/data1/ach/project/T5summarization/output/Bart_large_cnn_zero_cln"
for i in file_num:
    with open(os.path.join(data_dir,i+".hypo"),'r',encoding='utf8') as f:
        for line in f.readlines():
            hypo.append(line.strip())
    with open(os.path.join(data_dir,i+".target"),'r',encoding='utf8') as f:
        for line in f.readlines():
            target.append(line.strip())   
with open(os.path.join(data_dir,"test.hypo"),'w',encoding='utf8') as f:
    for i in hypo:
        f.writelines(i+'\n')
with open(os.path.join(data_dir,"test.target"),'w',encoding='utf8') as f:
    for i in target:
        f.writelines(i+'\n')