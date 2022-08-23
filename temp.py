torch.nn.parallel.DistributedDataParallel(module, 
                                          device_ids=None, 
                                          output_device=None, 
                                          dim=0, 
                                          broadcast_buffers=True, 
                                          process_group=None, 
                                          bucket_cap_mb=25, 
                                          find_unused_parameters=False, 
                                          check_reduction=False)
from cProfile import label
from ctypes.wintypes import tagRECT
import logging
from transformers import T5ForConditionalGeneration
import torch
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
import re
from transformers import T5ForConditionalGeneration
from transformers import (
    T5Tokenizer,
)
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import json
import time
import argparse
import sys
from unicodedata import category
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--n_gpus',
                    default=2,
                    type=int,
                    help='world size for distributed training')
args = parser.parse_args()
class T5Dataset(Dataset):
    def __init__(self,model_input,model_output):
        self.input = model_input
        self.output = model_output
    def __len__(self):
        return len(self.input)
    def __getitem__(self, index):
        return self.input[index],self.output[index]

model_name_path = "/data1/ach/project/Outline2Story/t5/out_attack2_new/checkpoint-15500"
# model_name_path = "/data/ach/project/lot-benchmark/datasets/outgen_nlgiw_2021_task2_story_generation_with_outline_large_cmrc_dp/checkpoint-1500"
name = "raw_data2"
#/data1/ach/lot-benchmark/baselines/generation/data/dev_online
task_name = "test"
with open("/data1/ach/project/Outline2Story/t5/%s/%s.source" % (name, task_name), "r") as fin:
    ipt = [line.strip() for line in fin]
with open("/data1/ach/project/Outline2Story/t5/%s/%s.target" % (name, task_name), "r") as fin:
    opt = [line.strip() for line in fin]
# chrs = (chr(i) for i in range(sys.maxunicode + 1))
# punctuation = set(c for c in chrs if category(c).startswith("P"))
local_rank = args.local_rank
n_gpus = args.n_gpus
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
test_dataset = T5Dataset(ipt,opt)
test_sampler = DistributedSampler(test_dataset)
test_dataloader = DataLoader(test_dataset,batch_size=8,sampler=test_sampler)
tokenizer = T5Tokenizer.from_pretrained(model_name_path)
# pad_token_id = tokenizer.pad_token_id
# tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>" % k for k in range(100)]})
rouge = Rouge()
model = T5ForConditionalGeneration.from_pretrained(model_name_path)
model.cuda(local_rank)
# model = DistributedDataParallel(model,device_ids=[local_rank],output_device=0)
model = DistributedDataParallel(model,device_ids=[local_rank])
file_out = "/data1/ach/project/Outline2Story/t5/test_attack2_new"
print("write to %s" % file_out)
bleu4_sum = 0
rouge_scores_values_sum = [0.0] * 9
n_samples = 0
start = time.time()
def pro(token_list,outline_list, tokenizer):
    id_sample = 0
    string = tokenizer.decode(token_list)
    string = string[:string.find("</s>")].replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
    for i in range(1,100,1):
        if i<len(outline_list):
            id_sample+=1
            string = string.replace("<extra_id_%d>" % i, " "+outline_list[i])
        else:
            string = string.replace("<extra_id_%d>" % i, "")
    # string = string.replace("<","")
    # string = "".join(string.strip().split())
    # print(string)
    # string = strB2Q(string)
    return string,id_sample/len(outline_list)
res = []
for (inputs,target) in tqdm(test_dataloader, desc=f"Evaluate:"):
    input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                            max_length=512).input_ids.cuda(local_rank)
    gen = model.module.generate(input_ids, do_sample=True, max_length=1024,top_p=0.9,
                            decoder_start_token_id=0,num_return_sequences=1)
    for ip, op, truth in zip(inputs, gen, target):
        temp = re.split('<extra_id_\d*>',ip)
        outline_list = [i for i in temp if i!=""]   
        temp,recover = pro(op,outline_list, tokenizer)
        bleu4 = sentence_bleu(truth.split(),temp, smoothing_function=SmoothingFunction().method7)
        rouge_scores = rouge.get_scores(temp, truth)
        rouge_scores_values = [v for k in rouge_scores[0].keys() for v in rouge_scores[0][k].values()]
        res.append({"hyp":temp,"story":truth,"bleu4":bleu4,"rouge":rouge_scores_values,"recover":recover})
with open("{}/infer_ddp_{}.jsonl".format(file_out,local_rank),'w',encoding="utf8") as file :
    file.write(json.dumps(res,ensure_ascii=False))