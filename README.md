# DQN(λ) — Reconciling λ-Returns with Experience Replay

DQN(λ) is an instantiation of the ideas proposed in [[1](#references)] that extends DQN [[2](#references)] to efficiently utilize various types of λ-returns [[3](#references)].
These can significantly improve sample efficiency.

If you use this repository in published work, please cite the paper:

```
@inproceedings{daley2019reconciling,
  title={Reconciling $\lambda$-Returns with Experience Replay},
  author={Daley, Brett and Amato, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```


## Contents

[Setup](#setup)

[Quickstart: DQN(λ)](#quickstart-dqnλ)

[Quickstart: DQN](#quickstart-dqn)

[Atari Environment Naming Convention](#atari-environment-naming-convention)

[Return Types](#return-types)

[License, Acknowledgments, and References](#license)


---

## Setup

This repository requires Python 3.5+ along with these dependencies: `gym`, `atari-py`, `tensorflow`, and `numpy`.

To automatically install package versions that are known to work with this code, just clone the repository and run `pip`:

```
git clone https://github.com/brett-daley/dqn-lambda.git
cd dqn-lambda
pip install -r requirements.txt
```

Alternatively, you can try other versions of the packages listed in `requirements.txt`, but they might not work.

> **Note:** Training will likely be impractical without GPU support.
> See [this TensorFlow guide](https://www.tensorflow.org/install/gpu) for `tensorflow-gpu` and CUDA setup.


---

## Quickstart: DQN(λ)
### Atari Games

You can train DQN(λ) on any of the Atari games included in the OpenAI Gym (see [Atari Environment Naming Convention](#atari-environment-naming-convention)).
For example, the following command runs DQN(λ) with λ=0.75 on Pong for 1.5 million timesteps:

```
python run_dqn_atari.py --env pong --return-type pengs-0.75 --timesteps 1.5e6
```

The table in [Return Types](#return-types) lists all of the return estimators supported by `--return-type`.
To get a description of the other possible command-line arguments, run this:

```
python run_dqn_atari.py --help
```


### Classic Control Environments

You can run DQN(λ) on CartPole by simply executing `python run_dqn_cartpole.py`.
This script does not take command-line arguments; all values are hard-coded.
You need to edit the file directly to change parameters.
This file is useful to test code on laptops or low-end desktops — particularly those without GPUs.

A [one-line change to the environment name](https://github.com/brett-daley/dqn-lambda/blob/67660460461eae1ac4f399f1e877bf919e6727fa/run_dqn_cartpole.py#L20) is all you need to run [other control environments](https://gym.openai.com/envs/#classic_control) (discrete action spaces only).


---

## Quickstart: DQN

This repository also includes a standard target-network implementation of DQN for reference.
Add the `--legacy` flag to run it instead of DQN(λ):

```
python run_dqn_atari.py --legacy
```

Note that setting any DQN(λ)-specific arguments (`--cache-size`, `--block-size`, or `--priority`) will throw an error, because they are undefined for DQN.
For example:

```
python run_dqn_atari.py --cache-size 10000 --legacy

Traceback (most recent call last):
  File "run_dqn_atari.py", line 82, in <module>
    main()
  File "run_dqn_atari.py", line 56, in main
    assert args.cache_size == 80000      # Ensure cache-related args have not been set
AssertionError
```

Similarly, trying to use a [return type](#return-types) other than the _n_-step return will also throw an error:

```
python run_dqn_atari.py --return-type pengs-0.75 --legacy

Traceback (most recent call last):
  File "run_dqn_atari.py", line 82, in <module>
    main()
  File "run_dqn_atari.py", line 59, in main
    replay_memory = make_legacy_replay_memory(args.return_type, replay_mem_size, args.history_len, discount)
  File "/home/brett/dqn-lambda/replay_memory_legacy.py", line 10, in make_legacy_replay_memory
    raise ValueError('Legacy mode only supports n-step returns but requested {}'.format(return_type))
ValueError: Legacy mode only supports n-step returns but requested pengs-0.75
```


---

## Atari Environment Naming Convention

The `--env` argument expects a string that is syntactically distinct from the typical OpenAI Gym format.
Environment names should be lowercase and use underscores instead of CamelCase.
The trailing `-v0` should also be removed.
For example, the six games tested in [[1](#references)] become the following:

```
BeamRider-v0 --> beam_rider
Breakout-v0 --> breakout
Pong-v0 --> pong
Qbert-v0 --> qbert
Seaquest-v0 --> seaquest
SpaceInvaders-v0 --> space_invaders
```

This pattern applies to [all of the Atari games supported by OpenAI Gym](https://gym.openai.com/envs/#atari).


---

## Return Types

The `--return-type` argument accepts a string that is parsed to determine which return estimator should be used.
The estimator might be parameterized by an `<int>` (greater than 0) or a `<float>` (between 0.0 and 1.0 — decimal point mandatory) extracted from the string.
The table below summarizes all of the possible return types supported by DQN(λ).

Return Type | Format | Example | Description
--- | --- | --- | ---
_n_-step | `nstep-<int>` | `nstep-3` | Classic _n_-step return [[3](#references)].<br>Standard DQN uses _n_=1.<br>_n_=`<int>`
Peng's Q(λ) | `pengs-<float>` | `pengs-0.75` | λ-return, unconditionally uses<br>max Q-values [[4](#references)].<br>A good "default" λ-return.<br>λ=`<float>`
Peng's Q(λ)<br>+ median | `pengs-median` |  | Peng's Q(λ)<br>+ median λ selection [[1](#references)].
Peng's Q(λ)<br>+ bounded 𝛿 | `pengs-maxtd-<float>` | `pengs-maxtd-0.01` | Peng's Q(λ)<br>+ bounded-error λ selection [[1](#references)].<br>𝛿=`<float>`
Watkin's Q(λ) | `watkins-<float>` | `watkins-0.75` | Peng's Q(λ), but sets λ=0<br>if Q-value is non-max [[4](#references)].<br>Ensures on-policy data.<br>λ=`<float>`
Watkin's Q(λ)<br>+ median | `watkins-median` |  | Watkin's Q(λ)<br>+ median λ selection [[1](#references)].
Watkin's Q(λ)<br>+ bounded 𝛿 | `watkins-maxtd-<float>` | `watkins-maxtd-0.01` | Watkin's Q(λ)<br>+ bounded-error λ selection [[1](#references)].<br>𝛿=`<float>`

See chapter 7.6 of [[4](#references)] for a side-by-side comparison of Peng's Q(λ) and Watkin's Q(λ).


---

## License

This code is released under the [MIT License](https://github.com/brett-daley/dqn-lambda/blob/master/LICENSE).


## Acknowledgments

This codebase evolved from the [partial DQN implementation](https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3) made available by the Berkeley Deep RL course, in turn based on Szymon Sidor's OpenAI implementation.
Special thanks to them.


## References

[1] [Reconciling λ-Returns with Experience Replay](https://arxiv.org/abs/1810.09967)

[2] [Human-Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)

[3] [Reinforcement Learning: An Introduction (2nd edition)](http://incompleteideas.net/book/the-book.html)

[4] [Reinforcement Learning: An Introduction (1st edition)](http://incompleteideas.net/book/first/the-book.html)
