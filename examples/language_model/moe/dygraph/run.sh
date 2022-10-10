export PYTHONPATH=$PYTHONPATH:../../../../
log_dir=dp8_ep1
rm -rf $log_dir

#export FLAGS_benchmark=1
#export FLAGS_call_stack_level=2
#export GLOG_v=0

python3 -m paddle.distributed.launch --log_dir $log_dir --gpus "0,1,2,3,4,5,6,7" run_moe_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-small-en \
    --input_dir \
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --device gpu\
    --eval_freq 1000\
    --warmup_rate 0.01\
    --global_batch_size 8\
    --micro_batch_size 1\
    --dp_degree 8\
    --mp_degree 1\
    --pp_degree 1\
    --expert_mode True\
    --logging_freq 1 \
    --num_experts 1 \
    --use_amp False \
    --scale_loss 32768 \
    --gate gshard \
    --gate_reduce_freq 1000 \
    --topo_file \
    --balance_loss_weight 1 \


#=============
# run pipeline parallel in moe
#=============

#python -m paddle.distributed.launch --log_dir $log_dir --gpus "4,5,6,7" run_moe_pipe_pretrain.py \
#    --model_type gpt \
#    --model_name_or_path gpt2-small-en \
#    --input_dir "./data"\
#    --output_dir "output"\
#    --weight_decay 0.01\
#    --grad_clip 1.0\
#    --max_steps 500000\
#    --save_steps 100000\
#    --decay_steps 320000\
#    --device gpu\
#    --eval_freq 1000\
#    --warmup_rate 0.01\
#    --global_batch_size 16\
#    --micro_batch_size 2\
#    --dp_degree 2\
#    --mp_degree 1\
#    --pp_degree 2\
#    --expert_mode True\
#    --logging_freq 1 \
#    --num_experts 2\
#    --use_amp False\
#    --scale_loss 32768

