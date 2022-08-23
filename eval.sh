export CUDA_VISIBLE_DEVICES=5,6,7
export MASTER_PORT=29505
python -m torch.distributed.launch --nproc_per_node=3 --master_port=${MASTER_PORT} --use_env eval.py \
    --config ./configs/T5summarization.yaml \
    --output_dir output/3.0.0_small_eval \
    --checkpoint /data1/ach/project/T5summarization/model/t5-small \
    --distributed true \
    --device cuda