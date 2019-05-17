import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import re
import os
import yaml
from glob import glob


def get_realtime(text):
    seconds = [float(m.group(1)) for m in re.finditer('Realtime (-?[0-9]+\.[0-9]+)', text)]
    hours = [s / 3600. for s in seconds]
    return hours


def get_timesteps(text):
    return [float(m.group(1)) for m in re.finditer('Timestep ([0-9]+)', text)]


def get_rewards(text):
    return [float(m.group(1)) for m in re.finditer('Mean reward (-?[0-9]+\.[0-9]+)', text)]


def axes_from_file(file, downsample, realtime):
    with open(file, 'r') as f:
        text = f.read()
        x_axis = (get_realtime if realtime else get_timesteps)(text)[::downsample]
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

        try:
            self._increment(trace, 'x', x_axis)
            self._increment(trace, 'y', y_axis)
            self._increment(trace, 'stderr', np.square(y_axis))
            self._increment(trace, 'count', 1)
        except ValueError:
            print('  Warning: corrupted data, attempting to skip file')

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
                print('  {} found only {} seed(s) out of {}'.format(trace, d['count'], n))
                #raise AssertionError('  {} found only {} seed(s) but needs {}'.format(trace, d['count'], n))
                n = d['count']

            if n == 0:
                raise AssertionError

            d['x'] /= n
            d['y'] /= n
            d['stderr'] = (d['stderr'] / n) - np.square(d['y'])
            d['stderr'] = np.sqrt(d['stderr']) #/ np.sqrt(n)

        self.normalized = True


def create_plot(input_dir, output_dir, filename, title, traces, colors, legend, ylim, num_timesteps, num_seeds, downsample, realtime):
    axis_manager = AxisManager(num_seeds, colors)

    try:
        for trace, color in zip(traces, colors):
            axis_manager.add(trace, color)
            files = glob(os.path.join(input_dir, trace))

            for f in files:
                x_axis, y_axis = axes_from_file(f, downsample, realtime)
                axis_manager.update(trace, x_axis, y_axis)

        axis_manager.normalize()

    except Exception as e:
        print('  Could not generate plot:', e)
        return False

    plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    x_max = -float('inf')
    for i, (trace, color) in enumerate(axis_manager.iter_traces()):
        x_axis, y_axis, stderr = axis_manager.axes(trace)
        plt.plot(x_axis, y_axis, color, label=legend[i])
        plt.fill_between(x_axis, (y_axis - stderr), (y_axis + stderr), color=color, alpha=0.25, linewidth=0)
        x_max = max(x_max, np.max(x_axis))

    plt.title(title, fontsize=20)
    #plt.grid(b=True, which='major', axis='both')
    plt.grid(b=True, which='both', axis='both')
    plt.legend(loc='best', framealpha=1.0, fontsize=12)

    plt.xlim([0, x_max if realtime else num_timesteps])
    if ylim is not None:
        plt.ylim(ylim)

    ax = matplotlib.pyplot.gca()

    if not realtime:
        f = lambda x, pos: str(int(x * 1e-6)) + ('M' if x > 0 else '')
        mkformatter = matplotlib.ticker.FuncFormatter(f)
        ax.xaxis.set_major_formatter(mkformatter)

        ax.set_xticks([0, x_max])
        ax.set_xticks(np.linspace(start=0, stop=x_max, num=6), minor=True)

    ax.set_aspect(1.0 / ax.get_data_ratio())

    plt.tight_layout(pad=0)

    fig = plt.gcf()
    path = os.path.join(output_dir, filename)
    fig.set_size_inches(6.4, 6.4)
    m = 0.0725  # all-around margin
    s = 0.03    # left-right shift
    plt.subplots_adjust(left=m + s, bottom=m, right=1-m + s, top=1-m)
    plt.savefig(path + '.png', format='png')
    #fig.savefig(path + '.pdf', format='pdf')
    print('  Plot saved as', path)
    plt.close()

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
        legend = params['legend']
        ylim = params.get('ylim', None)
        plots = params['plots']
        filenames = params['filenames']
        realtime = params.get('realtime', False)

        for filename, (title, traces) in zip(filenames, plots.items()):
            print('  Starting target:', filename)
            n_total += 1

            if create_plot(args.input_dir, args.output_dir, filename, title, traces, colors, legend, ylim, num_timesteps, num_seeds, downsample, realtime):
                n_success += 1
            print(flush=True)

    print('Generated {}/{} plots.'.format(n_success, n_total))


if __name__ == '__main__':
    main()
