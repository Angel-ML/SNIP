lr_a=(1e-5 2e-5 3e-5 4e-5 5e-5)
for (( i = 0; i <= 4; i++ ))
do
  for(( j = 0; j <= 0; j++ ))
  do
    TN="mnli"
    mkdir "./buffer_${TN}/"
    #export CUDA_VISIBLE_DEVICES=4,5,6,7
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    OUTPUT_DIR="./${TN}-4ka-output/qat-output_lr${lr_a[i]}/"
    python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 8009 main.py   \
      --model_path "/bert-base-uncased-mnli/"\
      --qat \
      --disable_deepspeed \
      --ddp \
      --output_dir ${OUTPUT_DIR}  \
      --per_gpu_eval_batch_size=96        \
      --per_gpu_train_batch_size=12       \
      --num_train_epochs 5 \
      --taskname ${TN} \
      --calib_step 80 \
      --percentile 99.99 \
      --calibrator percentile \
      --buffer_path "./buffer_${TN}/" \
      --save_steps 100

  done
done

