import argparse
from ast import Num
from lib2to3.pgen2 import token
import os
from syslog import LOG_SYSLOG
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# from models.model_ve import ALBEF
# from models.vit import interpolate_pos_embed
# from models.tokenization_bert import BertTokenizer
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
# from rouge import rouge
# scorer = rouge.Rouge()
import utils
from dataset import create_dataset, create_sampler, create_loader
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from scheduler import create_scheduler
from optim import create_optimizer

 
def pre(token_list, tokenizer):
    res = []
    for i in token_list:
        temp = tokenizer.decode(i)
        temp = temp.replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
        res.append(temp)
    return res
@torch.no_grad()
def predict(model, data_loader, tokenizer, device, config,args):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    res = []
    header = 'Evaluation:'
    print_freq = 50
    sample = 0
    averaged_scores = {'rouge1': {'p': 0, 'r': 0,'f': 0},
                       'rouge2': { 'p': 0, 'r': 0, 'f': 0},
                        'rougeLsum': {'p': 0, 'r': 0, 'f': 0}}
    # loss_all = 0
    text_res = []
    summary_res = []
    target_res = []
    for text,summary in metric_logger.log_every(data_loader, print_freq, header):
        
        text_inputs = tokenizer(text,max_length=1024,truncation=True,padding=True,return_tensors="pt").to(device) 
        # text_inputs = tokenizer(text,padding='longest', return_tensors="pt").to(device)
        # summary_inputs = tokenizer(summary, padding='longest',return_tensors="pt").to(device) 
    #           "early_stopping": true,
    #   "length_penalty": 2.0,
    #   "max_length": 200,
    #   "min_length": 30,
    #   "no_repeat_ngram_size": 3,
    #   "num_beams": 4,
        if args.distributed:
            output = model.module.generate(text_inputs['input_ids'],
                                max_length = 200,min_length=30,no_repeat_ngram_size=3, num_beams = 4,
                                early_stopping = True,bos_token_id = 0).cpu()
        else:
            output = model.generate(text_inputs['input_ids'],
                                max_length = 200,min_length=30,no_repeat_ngram_size=3, num_beams = 4,
                                early_stopping = True,bos_token_id = 0).cpu()        
        # output = utils.concat_all_gather(output,args.distributed).cpu()
        res_temp = pre(output,tokenizer)
        assert len(text)==len(res_temp)
        assert len(target_res)==len(summary_res)
        sample += len(text)
        target_res.extend(summary)
        summary_res.extend(res_temp)
        for source,target,predict in zip(text,summary,res_temp):
            temp = {}
            temp["source"] = source
            temp["summary"] = target
            temp["predict"] = predict
            # text_res.append(target)
            # target_res.append(target)
            # summary_res.append(predict)
            scores = scorer.score(target,predict)
            # scores = scorer.get_scores(target,predict)
            for metric in averaged_scores.keys():
                # for values in scores:
                    for sub_metric in range(len(averaged_scores[metric].keys())):
                        averaged_scores[metric][list(averaged_scores[metric].keys())[sub_metric]] += scores[metric][sub_metric]
    for key in averaged_scores.keys():
        for sub_key in averaged_scores[key].keys():
            averaged_scores[key][sub_key] /= sample
    print("rank is{}".format(utils.get_rank()))
    print(averaged_scores)
    averaged_scores['num'] = sample
    with open(os.path.join(args.output_dir,'{}.json'.format(utils.get_rank())),'w') as f:
        json.dump(averaged_scores,f,ensure_ascii=False)
    # with open(os.path.join(args.output_dir,'{}.target'.format(utils.get_rank())),'w') as f:
    #     for i in text_res:
    #         f.writelines(i+'\n')
    with open(os.path.join(args.output_dir,'{}.hypo'.format(utils.get_rank())),'w') as f:
        for i in summary_res:
            f.writelines(i+'\n')
    with open(os.path.join(args.output_dir,'{}.target'.format(utils.get_rank())),'w') as f:
        for i in target_res:
            f.writelines(i+'\n')
    # gather the stats from all processes
    
    
def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset("summarization",config) 
    # datasets[0].__getitem__(10)   
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size_train']]+[16]*2,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None])

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    #### Model #### 
    print("Creating model")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)

    model = model.to(device)   
    if args.best_checkpoint:    
        checkpoint = torch.load(args.best_checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict,strict=True)
        print('load checkpoint from %s'%args.best_checkpoint)
        print(msg)    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    # max_epoch = config['schedular']['epochs']
    # warmup_steps = config['schedular']['warmup_epochs']
    # best = 100000
    # best_epoch = 0
    # val_stats = evaluate(model, val_loader, tokenizer, device, config)
    print("Start training")
    start_time = time.time()

            
    test_stats = predict(model, test_loader, tokenizer, device, config,args)
        # test_stats = evaluate(model, test_loader, tokenizer, device, config)
        
        # if args.evaluate:
        #     break
        # lr_scheduler.step(epoch+warmup_steps+1)  
        # dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    # if utils.is_main_process():   
    #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
    #         f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Bart.yaml')
    parser.add_argument('--output_dir', default='output')  
    parser.add_argument('--checkpoint', default='/data1/ach/project/T5summarization/model/bart-large-cnn')   
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda:1')
    # parser.add_argument('--best_checkpoint',default='')
    parser.add_argument('--best_checkpoint',default='/data1/ach/project/T5summarization/output/bart/bart_large_cnn/3e-5-prompt-1024-grad/checkpoint_0.pth')
    # parser.add_argument('--prefix', default='cuda:6')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    print(args)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)