export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=29502
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} --use_env main.py \
    --config ./configs/T5summarization.yaml \
    --output_dir output/t5/1e-5-prompt \
    --checkpoint /data1/ach/project/T5summarization/model/t5-small \
    --distributed true \
    --device cuda
# export CUDA_VISIBLE_DEVICES=4
# python3 main.py \
#     --config ./configs/T5summarization.yaml \
#     --output_dir output/test_no_attention_mask \
#     --checkpoint /data1/ach/project/T5summarization/model/t5-small \
#     --distributed False \
#     --device cuda