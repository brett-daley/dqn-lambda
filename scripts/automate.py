import concurrent.futures
from subprocess import Popen, check_output
from argparse import ArgumentParser
import os
import yaml
from threading import Lock


class GPUArray:
    def __init__(self, procs_per_gpu):
        self.procs_per_gpu = procs_per_gpu
        self.n_virt_gpus = procs_per_gpu * self._num_phys_gpus()
        assert self.n_virt_gpus > 0

        self.gpu_set = set(range(self.n_virt_gpus))
        self.lock = Lock()

    def run_on_gpu(self, cmd, path):
        with open(path, 'w') as f:
            with self.lock:
                gpu_id = self.take()
                p = Popen(cmd, stdout=f, stderr=f)

        p.wait()

        with self.lock:
            self.replace(gpu_id)

    def take(self):
        assert len(self.gpu_set) > 0
        virt_id = self.gpu_set.pop()
        phys_id = virt_id // self.procs_per_gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = str(phys_id)
        return virt_id

    def replace(self, virt_id):
        assert len(self.gpu_set) < self.n_virt_gpus
        assert 0 <= virt_id < self.n_virt_gpus
        assert virt_id not in self.gpu_set
        self.gpu_set.add(virt_id)

    def num_gpus(self):
        return self.n_virt_gpus

    def _num_phys_gpus(self):
        try:
            return str(check_output(['nvidia-smi', '-L'])).count('UUID')
        except FileNotFoundError:
            return 0


def make_filename(env, history_len, recurrent, return_type, seed):
    filename = ['drqn' if recurrent else 'dqn', env, 'len' + str(history_len), return_type, 'seed' + str(seed)]
    return '_'.join(filename) + '.txt'


def make_cmd(env, history_len, recurrent, return_type, seed):
    cmd  = ['python', 'run_dqn_atari.py']
    cmd += ['--env', env]
    cmd += ['--history-len', str(history_len)]
    if recurrent:
        cmd += ['--recurrent']
    cmd += ['--return-type', return_type]
    cmd += ['--seed', str(seed)]
    return cmd


def make_joblist(experiments):
    jobs = []
    for exp in experiments:
        n_seeds = exp['n_seeds']
        recurrent = exp['recurrent']
        for env in exp['env']:
            for history_len in exp['history_len']:
                for return_type in exp['return_type']:
                    for seed in range(n_seeds):
                        cmd = make_cmd(env, history_len, recurrent, return_type, seed)
                        filename = make_filename(env, history_len, recurrent, return_type, seed)
                        path = os.path.join(args.outdir, filename)
                        jobs.append((cmd, path))
    return jobs


def run_job(cmd, path, gpu_array):
    print(' '.join(cmd), flush=True)
    gpu_array.run_on_gpu(cmd, path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('target',        type=str, help='Name of experiment to run. Use \'all\' to run everything.')
    parser.add_argument('procs_per_gpu', type=int)
    parser.add_argument('--outdir',      type=str, default='results/')
    parser.add_argument('--config',      type=str, default='scripts/_automate.yaml')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    experiments = config.values() if args.target == 'all' else config[args.target]
    jobs = make_joblist(experiments)

    gpu_array = GPUArray(args.procs_per_gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_array.num_gpus()) as executor:
        for cmd, path in jobs:
            if os.path.exists(path):
                print(path, 'already exists', flush=True)
            else:
                executor.submit(run_job, cmd, path, gpu_array)
