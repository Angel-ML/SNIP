lr_a=(1e-5 2e-5 3e-5 4e-5 5e-5)
for (( i = 0; i <= 0; i++ ))
do
  for(( j = 0; j <= 0; j++ ))
  do
    #export CUDA_VISIBLE_DEVICES=2,3
    OUTPUT_DIR="./FAKE-1ka-output-moresave/qat-output_lr${lr_a[i]}/"
    python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 8001 run_at48squad.py   \
      --qat \
      --disable_deepspeed \
      --ddp \
      --model_type bert   \
      --model_name_or_path bert-base-uncased     \
      --do_train  \
      --do_eval     \
      --do_lower_case     \
      --train_file /train-v1.1.json     \
      --predict_file /dev-v1.1.json     \
      --learning_rate ${lr_a[i]}     \
      --num_train_epochs 1     \
      --max_seq_length 384     \
      --doc_stride 128     \
      --output_dir ${OUTPUT_DIR}     \
      --per_gpu_eval_batch_size=64        \
      --per_gpu_train_batch_size=12       \
      --calib_step 80 \
      --percentile 99.99 \
      --calibrator percentile \
      --buffer_path "./att_reuse_buffer/" \
      --save_steps 50 \

  done
done

for (( i = 0; i <= 2; i++ ))
do
  for(( j = 0; j <= 0; j++ ))
  do
    #export CUDA_VISIBLE_DEVICES=2,3
    OUTPUT_DIR="./1ka-output-moresave/qat-output_lr${lr_a[i]}/"
    python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 8001 run_at48squad.py   \
      --qat \
      --disable_deepspeed \
      --ddp \
      --model_type bert   \
      --model_name_or_path bert-base-uncased     \
      --do_train  \
      --do_eval     \
      --do_lower_case     \
      --train_file /train-v1.1.json     \
      --predict_file /dev-v1.1.json     \
      --learning_rate ${lr_a[i]}     \
      --num_train_epochs 1     \
      --max_seq_length 384     \
      --doc_stride 128     \
      --output_dir ${OUTPUT_DIR}     \
      --per_gpu_eval_batch_size=64        \
      --per_gpu_train_batch_size=12       \
      --calib_step 50 \
      --percentile 99.99 \
      --calibrator percentile \
      --buffer_path "./att_reuse_buffer/" \
      --save_steps 50 \

  done
done
