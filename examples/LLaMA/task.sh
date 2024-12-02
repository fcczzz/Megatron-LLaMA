#!/bin/bash
for TP_SIZE in 1 2 4 8; do
    for DP_SIZE in 1 2 4 8; do
        # if  TP * DP > 8, skip
        if [ $(($TP_SIZE * $DP_SIZE)) -gt 8 ]; then
            continue
        fi
        bash /hy-tmp/Megatron-LLaMA/examples/LLaMA/LLaMA_3rd_try.sh $TP_SIZE $DP_SIZE
        PID=$!
        wait $PID
    done
done