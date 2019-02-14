import os
from subprocess import call


template = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8192
echo start at $(date)
{cmd}
echo end at $(date)
'''

history_lens = [1, 4]
recurrent_lens = [4]
nsteps = [1, 3]
lambdas = [0.6, 0.8]
seeds = [0, 1, 2]

environments = [
    'breakout',
    'beam_rider',
    'pong',
    'qbert',
    'seaquest',
]


def mkdir(name):
    path = os.path.join(os.getcwd(), name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def make_job_name(env, len, seed, nsteps, lve, recurrent):
    name = ['drqn' if recurrent else 'dqn']
    name += [env]
    name += ['len' + str(len)]
    if nsteps is not None:
        name += ['nsteps' + str(nsteps)]
    if lve is not None:
        name += ['lve' + str(lve)]
    name += ['seed' + str(seed)]
    return '_'.join(name)

def make_cmd(runner_path, results_path, env, len, seed, nsteps, lve, recurrent):
    cmd = ['python']
    cmd += [runner_path]
    cmd += ['--env', env]
    cmd += ['--history-len', len]
    if lve is not None:
        cmd += ['--Lambda', lve]
    if nsteps is not None:
        cmd += ['--nsteps', nsteps]
    if recurrent:
        cmd += ['--recurrent']
    cmd += ['--seed', seed]
    cmd += ['&>', results_path]
    return ' '.join([str(x) for x in cmd])


if __name__ == '__main__':
    # Make sure we are in the repository's root directory
    work_dir = os.getcwd()
    nstep_runner_path = os.path.join(work_dir, 'run_dqn_atari.py')
    lambda_runner_path = os.path.join(work_dir, 'run_dqnlambda_atari.py')
    assert os.path.exists(nstep_runner_path)
    assert os.path.exists(lambda_runner_path)

    # Make directories for slurm, log, and results files
    slurm_dir   = mkdir('slurm')
    log_dir     = mkdir('logs')
    results_dir = mkdir('results')

    def dispatch(runner_path, env, len, seed, nsteps=None, lve=None, recurrent=False):
        assert (nsteps is None) ^ (lve is None)

        # Generate job name and paths
        job_name = make_job_name(env, len, seed, nsteps, lve, recurrent)

        slurm_path   = os.path.join(slurm_dir,   job_name + '.slurm')
        log_path     = os.path.join(log_dir,     job_name + '.txt')
        results_path = os.path.join(results_dir, job_name + '.txt')

        # If results already exist, do not overwrite
        if os.path.exists(results_path):
            print('Warning: skipped', job_name, 'because results already exist')
            return

        # Fill in template and save to slurm directory
        with open(slurm_path, 'w') as file:
            slurm = template.format(
                job_name=job_name,
                log_path=log_path,
                cmd=make_cmd(runner_path, results_path, env, len, seed, nsteps, lve, recurrent)
            )
            file.write(slurm)

        # Call sbatch to queue the job
        print('Dispatching', job_name)
        call(['sbatch', slurm_path])

    # Begin dispatching experiments
    for env in environments:
        for s in seeds:
            for len in history_lens:
                for n in nsteps:
                    dispatch(nstep_runner_path, env, len, s, nsteps=n)
                for lve in lambdas:
                    dispatch(lambda_runner_path, env, len, s, lve=lve)
            for len in recurrent_lens:
                for n in nsteps:
                    dispatch(nstep_runner_path, env, len, s, nsteps=n, recurrent=True)
                for lve in lambdas:
                    dispatch(lambda_runner_path, env, len, s, lve=lve, recurrent=True)
