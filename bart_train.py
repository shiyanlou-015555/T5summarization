import argparse
from itertools import accumulate
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

import utils
from dataset import create_dataset, create_sampler, create_loader
# from transformers import T5Tokenizer
# from transformers import T5ForConditionalGeneration
# from transformers import BartTokenizer
# from transformers import BartForConditionalGeneration
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from scheduler import create_scheduler
from optim import create_optimizer
from apex import amp
from apex.parallel import DistributedDataParallel as apexDDP

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config,args):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
 
    for i,(text,summary) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    
        # images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
        
        text_inputs = tokenizer(text,max_length=1024,padding=True,truncation=True,return_tensors="pt").to(device) 
        summary_inputs = tokenizer(summary,max_length=512,padding=True,truncation=True,return_tensors="pt").to(device) 
        loss = model(input_ids=text_inputs.input_ids,attention_mask = text_inputs.attention_mask,decoder_attention_mask = summary_inputs.attention_mask,labels = summary_inputs.input_ids).loss    
        # loss = model(input_ids=text_inputs.input_ids,attention_mask = text_inputs.attention_mask,labels = summary_inputs.input_ids).loss    

    ##############################################################
        loss = loss/config["accumulation_step"]
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward() 
        if (i+1)%config["accumulation_step"] == 0:
            optimizer.step()
            optimizer.zero_grad()    
        # del output
        # del text_inputs
        # del summary_inputs
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.cpu().item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and scheduler is not None: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    sample = 0
    # loss_all = 0
    for text,summary in metric_logger.log_every(data_loader, print_freq, header):
        
        text_inputs = tokenizer(text,max_length=1024,padding=True,truncation=True,return_tensors="pt").to(device) 
        summary_inputs = tokenizer(summary, max_length=512,padding=True,truncation=True,return_tensors="pt").to(device) 
        output =  model(input_ids=text_inputs.input_ids,attention_mask = text_inputs.attention_mask,decoder_attention_mask = summary_inputs.attention_mask,labels = summary_inputs.input_ids)
        loss = output.loss
        sample+=1
        metric_logger.meters['loss'].update(loss.cpu().item(), n=len(text))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats loss:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss    
    
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

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    #### Model #### 
    print("Creating model")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer,
                                  opt_level='O1')  
            model = apexDDP(model)
            model_without_ddp = model.module  
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module    

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 100000
    best_epoch = 0
    # val_stats = evaluate(model, val_loader, tokenizer, device, config)
    print("Start training")
    start_time = time.time()
    # if args.smooth > 0:
    #     mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    # else:
    #     mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config,args)  
            
        val_stats = evaluate(model, val_loader, tokenizer, device, config)
        # test_stats = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():  
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                            #  **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                
            else:
                save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_{}.pth'.format(epoch)))     
                print("best is {}".format(best))    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                            #  **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if float(val_stats['loss'])<best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    best = float(val_stats['loss'])
                    best_epoch = epoch
        
        if args.evaluate:
            break
        if lr_scheduler is not None:
            lr_scheduler.step(epoch+warmup_steps+1)  
        if args.distributed:
            dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Bart.yaml')
    parser.add_argument('--output_dir', default='output/bart/prompt')  
    parser.add_argument('--checkpoint', default='/data1/ach/project/T5summarization/model/bart-large-cnn')   
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda:6')
    # parser.add_argument('--prefix', default='cuda:6')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--fp16', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
