export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29504
python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env main.py \
    --config ./configs/T5summarization.yaml \
    --output_dir output/3.0.0_small1 \
    --checkpoint /data1/ach/project/T5summarization/model/t5-small \
    --distributed true \
    --device cuda
# export CUDA_VISIBLE_DEVICES=4
# python3 main.py \
#     --config ./configs/T5summarization.yaml \
#     --output_dir output/3.0.0_small \
#     --checkpoint /data1/ach/project/T5summarization/model/t5-small \
#     --distributed False \
#     --device cuda