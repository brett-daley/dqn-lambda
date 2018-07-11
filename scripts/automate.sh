#!/bin/bash

PYTHON_CMD="`which python` run_dqn_atari.py"
OUTPUT_DIR='output'

ENVS='BreakoutNoFrameskip-v4 PongNoFrameskip-v4 QbertNoFrameskip-v4 SeaquestNoFrameskip-v4'
LAMBDAS='0.0 0.6 0.7 0.8'
SEED='0'

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Must set CUDA_VISIBLE_DEVICES"
    exit 1
fi

if [ ! -e $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

function run () {
    cmd=$1
    filename=$2
    path="$OUTPUT_DIR/$filename"

    if [ ! -e $path ]; then
        (set -x; $cmd &> $path)
    else
        echo "$2 already exists -- skipping"
    fi
}

for env in $ENVS; do
    for lambda in $LAMBDAS; do
        cmd="$PYTHON_CMD --env $env --Lambda $lambda --history_len 1 --seed $SEED"
        filename="dqn_${env}_1_e${lambda}.txt"
        run "$cmd" "$filename"

        cmd="$PYTHON_CMD --env $env --Lambda $lambda --history_len 4 --seed $SEED"
        filename="dqn_${env}_4_e${lambda}.txt"
        run "$cmd" "$filename"

        cmd="$PYTHON_CMD --env $env --Lambda $lambda --history_len 4 --recurrent --seed $SEED"
        filename="dqn_${env}_rec_e${lambda}.txt"
        run "$cmd" "$filename"
    done
done
