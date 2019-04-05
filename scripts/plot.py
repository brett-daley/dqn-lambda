import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import re
import os
import yaml
from glob import glob


def get_timesteps(text):
    return [float(m.group(1)) for m in re.finditer('Timestep ([0-9]+)', text)]


def get_rewards(text):
    return [float(m.group(1)) for m in re.finditer('Mean reward (-?[0-9]+\.[0-9]+)', text)]


def axes_from_file(file, downsample):
    with open(file, 'r') as f:
        text = f.read()
        x_axis = get_timesteps(text)[::downsample]
        y_axis = get_rewards(text)[::downsample]
    return np.array(x_axis), np.array(y_axis)


class AxisManager:
    def __init__(self, num_seeds, colors):
        self.num_seeds = num_seeds
        self.traces = {}
        self.normalized = False

    def add(self, trace, color):
        self.traces[trace] = {'x': None, 'y': None, 'stderr': None, 'color': color, 'count': 0}

    def update(self, trace, x_axis, y_axis):
        assert not self.normalized
        assert x_axis.shape == y_axis.shape
        self._increment(trace, 'x', x_axis)
        self._increment(trace, 'y', y_axis)
        self._increment(trace, 'stderr', np.square(y_axis))
        self._increment(trace, 'count', 1)

    def _increment(self, trace, key, value):
        d = self.traces[trace]
        if d[key] is None:
            d[key] = np.zeros_like(value)
        d[key] += value

    def iter_traces(self):
        assert self.normalized
        for trace, attr in self.traces.items():
            yield trace, attr['color']

    def axes(self, trace):
        assert self.normalized
        d = self.traces[trace]
        return d['x'], d['y'], d['stderr']

    def ylim(self):
        assert self.normalized
        raise NotImplementedError
        # TODO: Implement auto-sized y-axis

    def normalize(self):
        assert not self.normalized
        n = self.num_seeds

        for trace in self.traces.keys():
            d = self.traces[trace]

            if d['count'] != n:
                raise AssertionError('{} found only {} seed(s) but needs {}'.format(trace, d['count'], n))

            d['x'] /= n
            d['y'] /= n
            d['stderr'] = (d['stderr'] / n) - np.square(d['y'])
            d['stderr'] = np.sqrt(d['stderr']) / np.sqrt(n)

        self.normalized = True


def create_plot(input_dir, output_dir, filename, title, traces, colors, num_timesteps, num_seeds, downsample):
    axis_manager = AxisManager(num_seeds, colors)

    try:
        for trace, color in zip(traces, colors):
            axis_manager.add(trace, color)
            files = glob(os.path.join(input_dir, trace))

            for f in files:
                x_axis, y_axis = axes_from_file(f, downsample)
                axis_manager.update(trace, x_axis, y_axis)

        axis_manager.normalize()

    except Exception as e:
        print('  Could not generate plot:', e)
        return False

    plt.figure()
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    for trace, color in axis_manager.iter_traces():
        x_axis, y_axis, stderr = axis_manager.axes(trace)
        plt.plot(x_axis, y_axis, color)
        plt.fill_between(x_axis, (y_axis - stderr), (y_axis + stderr), color=color, alpha=0.25, linewidth=0)

    plt.title(title, fontsize=20)
    plt.grid(b=True, which='major', axis='both')

    plt.xlim([0, num_timesteps])

    plt.tight_layout(pad=0)
    fig = plt.gcf()

    filename = os.path.join(output_dir, filename + '.png')
    fig.savefig(filename)
    print('  Plot saved as', filename)

    return True


def main():
    parser = ArgumentParser()
    parser.add_argument('input_dir',    type=str)
    parser.add_argument('--output_dir', type=str, default='plots/')
    parser.add_argument('--config',     type=str, default='scripts/_plot.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    n_success = 0
    n_total = 0

    for group, params in config.items():
        print('Starting group:', group, flush=True)

        num_seeds = params['num_seeds']
        num_timesteps = params['num_timesteps']
        downsample = params['downsample']
        colors = params['colors']
        plots = params['plots']
        filenames = params['filenames']

        # print('  Creating legend for', group)
        # TODO: Write function to generate shared legend
        # print(flush=True)

        for filename, (title, traces) in zip(filenames, plots.items()):
            print('  Starting target:', filename)
            n_total += 1

            if create_plot(args.input_dir, args.output_dir, filename, title, traces, colors, num_timesteps, num_seeds, downsample):
                n_success += 1
            print(flush=True)

    print('Generated {}/{} plots.'.format(n_success, n_total))


if __name__ == '__main__':
    main()
