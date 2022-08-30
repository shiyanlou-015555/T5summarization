import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
# import torch


class summarization_dataset(Dataset):
    def __init__(self, ann_file,max_words=30,prefix="",use_prompt=False):        
        self.ann = json.load(open(ann_file,'r'))
        self.max_words = max_words
        self.prefix = prefix
        self.use_prompt = use_prompt
        if use_prompt:
            average_rouge_L = 0
            for temp in self.ann:
                average_rouge_L+=temp["rougeL"][-1]
            self.average_rouge_L = average_rouge_L/len(self.ann)
        # pass
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        '''
        Input agnostic model
        '''
        ann = self.ann[index]
        if self.use_prompt:
            # text = self.prefix+"generate {} abstractive summary. ".format('high' if ann["rougeL"][-1]< self.average_rouge_L else "low" )+ann["text"]
            text = self.prefix+"generate {} abstractive summary. ".format('high' if ann["rougeL"][-1]> self.average_rouge_L else "low" )+ann["text"]
            # pass
        else:
            text = self.prefix+ann["text"]
        # text = pre_caption(text,self.max_words)
        summary = ann["summary"].replace('\n'," ") 
        # summary = pre_caption(summary,self.max_words)
        return text, summary
    