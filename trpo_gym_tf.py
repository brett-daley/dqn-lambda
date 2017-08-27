from rllab.baselines.zero_baseline import ZeroBaseline
from atari_env import AtariEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy
from sandbox.rocky.tf.algos.trpo import TRPO

stub(globals())

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288
env = TfEnv(normalize(AtariEnv('Pong-v0', agent_history_length=2, force_reset=True, record_video=False)))

policy = CategoricalConvPolicy(
    name='policy',
    env_spec=env.spec,
    conv_filters=(16, 16),
    conv_filter_sizes=(4, 4),
    conv_strides=(2, 2),
    conv_pads=('VALID', 'VALID'),
    hidden_sizes=(20,)
)

baseline = ZeroBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=200,
    n_itr=120,
    discount=0.99,
    step_size=0.01,
    # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

run_experiment_lite(
    algo.train(),
    n_parallel=1,
    snapshot_mode="last",
    seed=1
)
