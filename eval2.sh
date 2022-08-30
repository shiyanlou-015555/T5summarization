export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29513
python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env eval.py \
    --config ./configs/T5summarization.yaml \
    --output_dir output/t5_base_zero2 \
    --checkpoint /data1/ach/project/T5summarization/model/t5-base \
    --distributed true \
    --device cuda