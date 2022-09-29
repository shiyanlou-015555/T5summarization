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
def train(model, data_loader, optimizer, tokenizer, max_epoch, device, mle_fn, config, args, val_loader):
    # model, train_loader, optimizer, tokenizer, epoch, device, mle_fn, config,args
    # train
    model.train() 
    best = 1000 
    all_step_cnt = 1
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    print("dataset config is {}".format(config[args.dataset]))
    for epoch in range(max_epoch):
        header = 'Train Epoch: [{}]'.format(epoch)
        print_freq = 50   
        step_cnt = 0
        # warmup_iterations = warmup_steps*step_size  
        for i,(text,summary) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            step_cnt += 1         
            text_inputs = tokenizer.batch_encode_plus(text,max_length=1024,pad_to_max_length=False,truncation=True,return_tensors="pt").to(device) 
            summary_inputs = tokenizer.batch_encode_plus(summary,max_length=config[args.dataset]['max_target_len'],pad_to_max_length=False,truncation=True,padding=True,return_tensors="pt").to(device) 
            output = model(input_ids=text_inputs.input_ids,attention_mask = text_inputs.attention_mask,decoder_attention_mask = summary_inputs.attention_mask,labels = summary_inputs.input_ids)   
            # loss = model(input_ids=text_inputs.input_ids,attention_mask = text_inputs.attention_mask,labels = summary_inputs.input_ids).loss    
            loss = mle_fn(output.logits.transpose(1,2),summary_inputs.input_ids)
        ##############################################################
            loss = loss/config["accumulation_step"]
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if step_cnt == config["accumulation_step"]:
                step_cnt = 0
                all_step_cnt += 1
                lr = config["max_lr"] * min(all_step_cnt ** (-0.5), all_step_cnt * (config["warmup_steps"] ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
            if all_step_cnt % 1000 == 0:
                val_stats = evaluate(model, val_loader, tokenizer, device, config, args)
                if utils.is_main_process():
                    print("best is {}".format(best))    
                    log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                                #  **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,
                                'step':all_step_cnt
                                }

                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    if float(val_stats['loss'])<best:
                        save_obj = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_{}.pth'.format(all_step_cnt))) 
                        best = float(val_stats['loss'])
                if args.distributed:
                    dist.barrier()   
                        # best_epoch = epoch
            # del output
            # del text_inputs
            # del summary_inputs
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.cpu().item())
            
  
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    # return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config, args):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    sample = 0
    # loss_all = 0
    for text,summary in metric_logger.log_every(data_loader, print_freq, header):
        
        text_inputs = tokenizer(text,max_length=1024,padding=True,truncation=True,return_tensors="pt").to(device) 
        summary_inputs = tokenizer(summary, max_length=config[args.dataset]['max_target_len'], padding=True,truncation=True,return_tensors="pt").to(device) 
        output =  model(input_ids=text_inputs.input_ids,attention_mask = text_inputs.attention_mask,decoder_attention_mask = summary_inputs.attention_mask,labels = summary_inputs.input_ids)
        loss = output.loss
        sample+=1
        metric_logger.meters['loss'].update(loss.cpu().item(), n=len(text))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats loss:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
# def pre(token_list, tokenizer):
#     res = []
#     for i in token_list:
#         temp = tokenizer.decode(i)
#         temp = temp.replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
#         res.append(temp)
#     return res
# @torch.no_grad()
# def predict(model, data_loader, tokenizer, device, config,args):
#     # test
#     model.eval()
            
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     res = []
#     header = 'Evaluation:'
#     print_freq = 50
#     sample = 0
#     averaged_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
#                        'rouge-2': {'f': 0, 'p': 0, 'r': 0},
#                         'rouge-l': {'f': 0, 'p': 0, 'r': 0}}
#     # loss_all = 0
#     text_res = []
#     summary_res = []
#     for text,summary in metric_logger.log_every(data_loader, print_freq, header):
        
#         text_inputs = tokenizer(text,max_length=1024,truncation=True,padding=True,return_tensors="pt").to(device) 
#         if args.distributed:
#             output = model.module.generate(text_inputs['input_ids'],
#                                 max_length = 200,min_length=30,no_repeat_ngram_size=3, num_beams = 4,
#                                 early_stopping = True,bos_token_id = 0).cpu()
#         else:
#             output = model.generate(text_inputs['input_ids'],
#                                 max_length = 200,min_length=30,no_repeat_ngram_size=3, num_beams = 4,
#                                 early_stopping = True,bos_token_id = 0).cpu()        
#         # output = utils.concat_all_gather(output,args.distributed).cpu()
#         res_temp = pre(output,tokenizer)
#         assert len(text)==len(res_temp)
#         sample += len(text)
#         for source,target,predict in zip(text,summary,res_temp):
#             temp = {}
#             temp["source"] = source
#             temp["summary"] = target
#             temp["predict"] = predict
#             text_res.append(target)
#             summary_res.append(predict)
#             scores = scorer.score(target,predict)
#             for metric in averaged_scores.keys():
#                 for values in scores:
#                     for sub_metric in averaged_scores[metric]:
#                         averaged_scores[metric][sub_metric] += values[metric][sub_metric]
#     for key in averaged_scores.keys():
#         for sub_key in averaged_scores[key].keys():
#             averaged_scores[key][sub_key] /= sample
    
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
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint)

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
            
        # val_stats = evaluate(model, val_loader, tokenizer, device, config)
        # # test_stats = evaluate(model, test_loader, tokenizer, device, config)

        # if utils.is_main_process():  
        #     if args.evaluate:
        #         log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
        #                     #  **{f'test_{k}': v for k, v in test_stats.items()},
        #                      'epoch': epoch,
        #                     }

        #         with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        #             f.write(json.dumps(log_stats) + "\n")                
        #     else:
        #         save_obj = {
        #                 'model': model_without_ddp.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 # 'lr_scheduler': lr_scheduler.state_dict(),
        #                 'config': config,
        #                 'epoch': epoch,
        #             }
        #         torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_{}.pth'.format(epoch)))     
        #         print("best is {}".format(best))    
        #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                      **{f'val_{k}': v for k, v in val_stats.items()},
        #                     #  **{f'test_{k}': v for k, v in test_stats.items()},
        #                      'epoch': epoch,
        #                     }

        #         with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        #             f.write(json.dumps(log_stats) + "\n")

        #         if float(val_stats['loss'])<best:
        #             save_obj = {
        #                 'model': model_without_ddp.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 # 'lr_scheduler': lr_scheduler.state_dict(),
        #                 'config': config,
        #                 'epoch': epoch,
        #             }
        #             torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
        #             best = float(val_stats['loss'])
        #             best_epoch = epoch
        
        # if args.evaluate:
        #     break
        # if lr_scheduler is not None:
        #     lr_scheduler.step(epoch+warmup_steps+1)  
        # if args.distributed:
        #     dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    # if utils.is_main_process():   
    #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
    #         f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Bart_brio.yaml')
    parser.add_argument('--output_dir', default='output/bart/prompt')  
    parser.add_argument('--checkpoint', default='/data1/ach/project/T5summarization/model/bart-large-cnn')   
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda:7')
    # parser.add_argument('--prefix', default='cuda:6')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--fp16', default=False, type=bool)
    parser.add_argument('--dataset', default="XSUM", type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
