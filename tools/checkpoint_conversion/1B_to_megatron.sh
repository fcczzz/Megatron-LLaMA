python /hy-tmp/Megatron-LLaMA/tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--load_path "/hy-tmp/models/Llama-3.2-1B" \
--save_path "/hy-tmp/models/1B-Megatron-pp2-fp8" \
--target_tensor_model_parallel_size 4 \
--target_pipeline_model_parallel_size 2 \
--target_data_parallel_size 1 \
--target_params_dtype "fp8" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "/hy-tmp/Megatron-LLaMA"