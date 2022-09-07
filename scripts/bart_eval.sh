# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export MASTER_PORT=29515
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env bart_eval1.py \
#     --config ./configs/Bart.yaml \
#     --output_dir /data1/ach/project/T5summarization/output/bart/bart_large_cnn/3e-5-prompt-1024-grad/eval \
#     --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
#     --best_checkpoint /data1/ach/project/T5summarization/output/bart/bart_large_cnn/3e-5-prompt-1024-grad/checkpoint_0.pth \
#     --distributed true \
#     --device cuda
export CUDA_VISIBLE_DEVICES=1
# export MASTER_PORT=29515
# python bart_eval1.py \
#     --config ./configs/Bart.yaml \
#     --output_dir /data1/ach/project/T5summarization/output/bart/bart_large_cnn/3e-5-prompt-1024 \
#     --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
#     --best_checkpoint /data1/ach/project/T5summarization/output/bart/bart_large_cnn/3e-5-prompt-1024/checkpoint_0.pth \
#     --distributed true \
#     --device cuda
python bart_eval1.py \
    --config ./configs/Bart.yaml \
    --output_dir /data1/ach/project/T5summarization/output/bart/bart_large_cnn/3e-5-prompt-1024-grad/eval \
    --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
    --best_checkpoint /data1/ach/project/T5summarization/output/bart/bart_large_cnn/3e-5-prompt-1024-grad/checkpoint_0.pth \
    --distributed true \
    --device cuda