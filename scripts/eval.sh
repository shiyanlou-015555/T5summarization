# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export MASTER_PORT=29507
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env eval.py \
#     --config ./configs/T5summarization.yaml \
#     --output_dir output/t5_zero \
#     --checkpoint /data1/ach/project/T5summarization/model/t5-small \
#     --distributed true \
#     --device cuda
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export MASTER_PORT=29508
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env eval.py \
#     --config ./configs/T5summarization.yaml \
#     --output_dir output/t5_base_zero \
#     --checkpoint /data1/ach/project/T5summarization/model/t5-base \
#     --distributed true \
#     --device cuda
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29512
python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env eval.py \
    --config ./configs/T5summarization.yaml \
    --output_dir output/t5_base_zero \
    --checkpoint /data1/ach/project/T5summarization/model/t5-base \
    --distributed true \
    --device cuda
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export MASTER_PORT=29521
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env bart_eval.py \
#     --config ./configs/Bart.yaml \
#     --output_dir /data1/ach/project/T5summarization/output/bart/zero_bart_cnn \
#     --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
#     --distributed true \
#     --device cuda 
    # --best_checkpoint /data1/ach/project/T5summarization/output/1e-3/checkpoint_0.pth \
# export CUDA_VISIBLE_DEVICES=4
# export MASTER_PORT=29521
# python  bart_eval.py \
#     --config ./configs/Bart.yaml \
#     --output_dir /data1/ach/project/T5summarization/output/bart/zero_bart_cnn \
#     --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
#     --distributed false \
#     --device cuda 