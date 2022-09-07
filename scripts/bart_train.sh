export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29519
python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env brio_bart.py \
    --config ./configs/Bart_brio.yaml \
    --output_dir output/bart/bart_large_cnn/3e-5-prompt-1024-grad-brio \
    --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
    --distributed true \
    --device cuda
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export MASTER_PORT=29519
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env brio_bart.py \
#     --config ./configs/Bart_brio.yaml \
#     --output_dir output/bart/bart_large_cnn/3e-5-prompt-1024-grad-brio \
#     --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
#     --distributed true \
#     --device cuda
# export CUDA_VISIBLE_DEVICES=4
# export MASTER_PORT=29505
# python bart_train.py \
#     --config ./configs/Bart.yaml \
#     --output_dir output/bart/1e-5-noprompt \
#     --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
#     --device cuda