#!/bin/bash

PYTHON_NSTEP_CMD="`which python` run_dqn_atari.py"
PYTHON_LAMBDA_CMD="`which python` run_dqnlambda_atari.py"
OUTPUT_DIR='output'

ENVS='beam_rider breakout pong qbert seaquest'
NSTEPS='1 3'
LAMBDAS='0.6 0.8'
SEEDS='0 1 2'

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
    for seed in $SEEDS; do
        for n in $NSTEPS; do
            cmd="$PYTHON_NSTEP_CMD --env $env --nsteps $n --history-len 1 --seed $seed"
            filename="dqn_${env}_len1_nsteps${n}_seed${seed}.txt"
            run "$cmd" "$filename"

            cmd="$PYTHON_NSTEP_CMD --env $env --nsteps $n --history-len 4 --seed $seed"
            filename="dqn_${env}_len4_nsteps${n}_seed${seed}.txt"
            run "$cmd" "$filename"

            cmd="$PYTHON_NSTEP_CMD --env $env --nsteps $n --history-len 4 --recurrent --seed $seed"
            filename="drqn_${env}_len4_nsteps${n}_seed${seed}.txt"
            run "$cmd" "$filename"
        done

        for lve in $LAMBDAS; do
            cmd="$PYTHON_LAMBDA_CMD --env $env --Lambda $lve --history-len 1 --seed $seed"
            filename="dqn_${env}_len1_lve${lve}_seed${seed}.txt"
            run "$cmd" "$filename"

            cmd="$PYTHON_LAMBDA_CMD --env $env --Lambda $lve --history-len 4 --seed $seed"
            filename="dqn_${env}_len4_lve${lve}_seed${seed}.txt"
            run "$cmd" "$filename"

            cmd="$PYTHON_LAMBDA_CMD --env $env --Lambda $lve --history-len 4 --recurrent --seed $seed"
            filename="drqn_${env}_len4_lve${lve}_seed${seed}.txt"
            run "$cmd" "$filename"
        done
    done
done
