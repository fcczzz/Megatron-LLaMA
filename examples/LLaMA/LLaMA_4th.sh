#!/bin/bash
# llama 1B
DATASET_1="/hy-tmp/dataset/code-search-net-python/data_in_megatron_form/code-search-net_text_document"
DATASET="1 ${DATASET_1} "

TP_SIZE=$1
PP_SIZE=$2
echo "TP_SIZE: ${TP_SIZE}, PP_SIZE: ${PP_SIZE}"
WORLD_SIZE=8
MICRO_BATCH_SIZE=1
# The int is the number of micro steps of gradient accumulation
GLOBAL_BATCH_SIZE=$((($WORLD_SIZE * $MICRO_BATCH_SIZE) / ($TP_SIZE * $PP_SIZE) * 8))
# GLOBAL_BATCH_SIZE=128

JOB_NAME="LLaMA_tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gpus${WORLD_SIZE}"

TOKENIZER_PATH="/hy-tmp/models/Llama-3.2-1B"
TENSORBOARD_DIR="/hy-tmp/models/self_llama/runs"

LOG_DIR="/hy-tmp/logs"
LOG_NAME=${LOG_DIR}/${JOB_NAME}.log
mkdir -p ${LOG_DIR}

TRAIN_ITERS=30
EVAL_ITERS=10
EVAL_INTERVAL=1000
SAVE_INTERVAL=100
LOG_INTERVAL=1

# Setting --tensorboard-queue-size to 1 significantly slows down the training
options_1b=" \
    --finetune \
    --sequence-parallel \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers 8 \
        --hidden-size 2048 \
        --num-attention-heads 32 \
        --seq-length 32 \
        --max-position-embeddings 32 \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --swiglu \
        --ffn-hidden-size 8192\
        --disable-bias-linear \
        --RMSNorm \
        --layernorm-epsilon 1e-6 \
        --causal-lm \
    --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --make-vocab-size-divisible-by 1 \
    --init-method-std 0.01 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 6.0e-5 \
        --lr-decay-iters 10 \
        --lr-warmup-iters 5 \
        --min-lr 6.0e-6 \
        --override-opt_param-scheduler \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --overlapped-distributed-optimizer \
        --reduce-bucket-size=2e8 \
        --no-gradient-accumulation-fusion \
    --dataloader-type cyclic \
        --data-impl mmap \
        --data-path ${DATASET} \
        --split 98,2,0 \
    --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
    --save-interval ${SAVE_INTERVAL} \
    --log-interval ${LOG_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
        --tensorboard-queue-size 1000 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
    --job-name ${JOB_NAME} \
    --bf16 \
    --recompute-activations \
        --recompute-granularity selective \
    --use-flash-attn
    "
options_tiny=" \
    --finetune \
    --sequence-parallel \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers 16 \
        --hidden-size 2048 \
        --num-attention-heads 32 \
        --seq-length 32 \
        --max-position-embeddings 32 \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --swiglu \
        --ffn-hidden-size 8192\
        --disable-bias-linear \
        --RMSNorm \
        --layernorm-epsilon 1e-6 \
        --causal-lm \
    --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --make-vocab-size-divisible-by 1 \
    --init-method-std 0.01 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 6.0e-5 \
        --lr-decay-iters 10 \
        --lr-warmup-iters 5 \
        --min-lr 6.0e-6 \
        --override-opt_param-scheduler \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --overlapped-distributed-optimizer \
        --reduce-bucket-size=2e8 \
        --no-gradient-accumulation-fusion \
    --dataloader-type cyclic \
        --data-impl mmap \
        --data-path ${DATASET} \
        --split 98,2,0 \
    --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
    --save-interval ${SAVE_INTERVAL} \
    --save ${SAVE_CHECKPOINT_PATH} \
    --load ${LOAD_CHECKPOINT_PATH} \
        --no-load-optim \
    --log-interval ${LOG_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
        --tensorboard-queue-size 1000 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
    --job-name ${JOB_NAME} \
    --bf16 \
    --recompute-activations \
        --recompute-granularity selective \
    --use-flash-attn
    "

options_backup=" \
    --finetune \
    --sequence-parallel \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers 12 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --swiglu \
        --ffn-hidden-size 11008\
        --disable-bias-linear \
        --RMSNorm \
        --layernorm-epsilon 1e-6 \
        --causal-lm \
    --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --make-vocab-size-divisible-by 1 \
    --init-method-std 0.01 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 6.0e-5 \
        --lr-decay-iters 10 \
        --lr-warmup-iters 5 \
        --min-lr 6.0e-6 \
        --override-opt_param-scheduler \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --overlapped-distributed-optimizer \
        --reduce-bucket-size=2e8 \
        --no-gradient-accumulation-fusion \
    --dataloader-type cyclic \
        --data-impl mmap \
        --data-path ${DATASET} \
        --split 98,2,0 \
    --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
    --save-interval ${SAVE_INTERVAL} \
        --save ${SAVE_CHECKPOINT_PATH} \
    --load ${LOAD_CHECKPOINT_PATH} \
        --no-load-optim \
    --log-interval ${LOG_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
        --tensorboard-queue-size 1000 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
    --job-name ${JOB_NAME} \
    --bf16 \
    --recompute-activations \
        --recompute-granularity selective \
    --use-flash-attn
    "

torchrun --nproc_per_node=8 --master_port=29500 pretrain_llama.py ${options_1b} 2>&1 | tee ${LOG_NAME}
