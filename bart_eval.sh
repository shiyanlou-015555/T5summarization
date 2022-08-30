export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29513
python -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} --use_env bart_eval1.py \
    --config ./configs/Bart.yaml \
    --output_dir output/Bart_large_cnn_zero_cln \
    --checkpoint /data1/ach/project/T5summarization/model/bart-large-cnn \
    --distributed true \
    --device cuda