import argparse
from itertools import accumulate
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
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
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# from models.model_ve import ALBEF
# from models.vit import interpolate_pos_embed
# from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
# from transformers import T5Tokenizer
# from transformers import T5ForConditionalGeneration
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
# from transformers import AutoTokenizer
# from transformers import AutoModelForSeq2SeqLM
from scheduler import create_scheduler
from optim import create_optimizer
from apex import amp
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
from apex.parallel import DistributedDataParallel as apexDDP
from label_smoothing_loss import label_smoothing_loss
from prefix_model import PrefixSummarizationModule
    
def main(args, config):
    utils.init_distributed_mode(args)    
    print(config)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    print(args)
    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset("summarization",config) 
    datasets[0].__getitem__(1)   
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None])

    tokenizer = BartTokenizer.from_pretrained(args.checkpoint)

    #### Model #### 
    print("Creating model")
    # model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    model = PrefixSummarizationModule(config)

    model = model.to(device)   
    model_without_ddp = model
    optimizer = optim.Adam(model.parameters())
    if args.distributed:
        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer,
                                  opt_level='O1')  
            model = apexDDP(model)
            model_without_ddp = model.module  
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module    
    
    
    max_epoch = config['epochs']
    start_time = time.time()
    if config["smooth"] > 0:
        mle_fn = label_smoothing_loss(ignore_index=tokenizer.pad_token_id, epsilon=config["smooth"])
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    train_stats = train(model, train_loader, optimizer, tokenizer, max_epoch, device, mle_fn, config,args,val_loader)  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    # if utils.is_main_process():   
    #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
    #         f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/prefix.yaml')
    parser.add_argument('--output_dir', default='output/prefix')  
    parser.add_argument('--checkpoint', default='/data1/ach/project/T5summarization/model/bart-large-cnn')   
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda:1')
    # parser.add_argument('--prefix', default='cuda:6')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--fp16', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
