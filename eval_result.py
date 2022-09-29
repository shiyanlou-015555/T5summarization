from genericpath import samefile
import json
import os
from random import sample 
averaged_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                    'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                    'rouge-l': {'f': 0, 'p': 0, 'r': 0}}
path = '/data1/ach/project/T5summarization/experiment_for_zero_shot/t5_base_zero'
file_list = os.listdir(path)
file_list = [i for i in file_list if i.endswith('json')]
file_list = [os.path.join(path,i) for i in file_list]
sample = 0
for i in file_list:
    scores = json.load(open(i,'r'))
    sample += scores.pop('num')
    for metric in averaged_scores.keys():
            for sub_metric in averaged_scores[metric]:
                averaged_scores[metric][sub_metric] += scores[metric][sub_metric]
for key in averaged_scores.keys():
    for sub_key in averaged_scores[key].keys():
        averaged_scores[key][sub_key] /= len(file_list)
print(averaged_scores)