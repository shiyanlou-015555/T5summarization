import argparse
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score
from rouge_score import rouge_scorer
import tqdm
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
def main(args):
    idx = []
    with open(os.path.join(args.output,"test.idx"),'r') as f:
        for i in f.readlines():
            idx.append(i.strip())
    predict = []
    with open(os.path.join(args.output,"test.hypo.tokenized"),'r') as f:
        for i in f.readlines():
            predict.append(i.strip())
    target = []
    with open(os.path.join(args.output,"test.hypo.target"),'r') as f:
        for i in f.readlines():
            target.append(i.strip())
    source = []
    with open(os.path.join(args.target,"test.text.tokenized"),'r') as f:
        for i in f.readlines():
            source.append(i.strip())
    res_target = []
    res_predict = []
    for i,j,k in tqdm.tqdm(zip(idx,predict,target)):
# i表示j的索引
        score1 = scorer.score(j,source[int(i)])# 预测结果
        score2 = scorer.score(k,source[int(i)])# 原始结果
        rougeLsum1 = score1["rouge2"][1]
        rougeLsum2 = score2["rouge2"][1]
        # 0为高抽象，1为低抽象
        if rougeLsum2 < args.average_pre:
            res_target.append(0)
        else:
            res_target.append(1)
        if rougeLsum1 < args.average_pre:
            res_predict.append(0)
        else:
            res_predict.append(1)
    print(sum(res_target))
    print("acc is {}".format(accuracy_score(res_target,res_predict)))
    print("pre is {}".format(precision_score(res_target,res_predict)))
    print("recall is {}".format(recall_score(res_target,res_predict)))
    print("f is {}".format(f1_score(res_target,res_predict))) 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/data1/ach/project/T5summarization/output/bart/bart_large_cnn/rouge2/3e-5-prompt-1024-grad-brio-large-low/eval')
    parser.add_argument('--target', default='/data1/ach/project/T5summarization/data/cnn_cln_tagged')  
    parser.add_argument('--average_pre', default='0.502',type=float)   
    args = parser.parse_args()
    main(args)