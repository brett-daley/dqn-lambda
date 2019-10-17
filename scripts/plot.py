import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import re
import os
import yaml
from glob import glob
import scipy.stats


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

    def create_trace(self, trace, color):
        self.traces[trace] = {'filename': [], 'x': [], 'y': [], 'color': color}

    def update_trace(self, trace, filename, x_axis, y_axis):
        assert trace in self.traces.keys()
        names = self.traces[trace]['filename']
        x = self.traces[trace]['x']
        y = self.traces[trace]['y']

        assert filename not in names
        if len(names) > 0 and x[0].shape != x_axis.shape:
            print(f'  Warning: {os.path.basename(filename)} shape {x_axis.shape} does not match shape {x[0].shape}; file might be corrupted. Skipping...')
            return False

        names.append(filename)
        x.append(x_axis)
        y.append(y_axis)

    def iter_traces(self):
        for trace, attr in self.traces.items():
            x, y, color = np.asarray(attr['x']), np.asarray(attr['y']), attr['color']
            assert len(x) == len(y) != 0
            n = len(x)

            x_mean = np.mean(x, axis=0)
            y_mean = np.mean(y, axis=0)
            if n > 1:
                y_error = scipy.stats.sem(y, axis=0)
                #y_error = np.std(y, axis=0)
            else:
                y_error = np.zeros_like(y_mean)

            yield trace, x_mean, y_mean, y_error, n, color


def create_plot(input_dir, output_dir, filename, title, traces, colors, legend, ylim, num_timesteps, num_seeds, downsample, realtime):
    axis_manager = AxisManager(num_seeds, colors)

    for trace, color in zip(traces, colors):
        axis_manager.create_trace(trace, color)
        files = glob(os.path.join(input_dir, trace))

        for f in files:
            x_axis, y_axis = axes_from_file(f, downsample, realtime)
            axis_manager.update_trace(trace, f, x_axis, y_axis)

    plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    x_max = -float('inf')
    y0_min = float('inf')
    for i, (trace, x, y, error, n, color) in enumerate(axis_manager.iter_traces()):
        if n == 0:
            print(f'  Error: {trace} found 0 seeds')
            return False
        if n < num_seeds:
            print(f'  Warning: {trace} found only {n} seed(s) out of {num_seeds}')

        plt.plot(x, y, color, label=legend[i])
        plt.fill_between(x, (y - error), (y + error), color=color, alpha=0.25, linewidth=0)
        x_max = max(x_max, np.max(x))
        y0_min = min(y0_min, np.min(y - error))

    plt.title(title, fontsize=20)
    plt.grid(b=True, which='both', axis='both')
    plt.legend(loc='best', framealpha=1.0, fontsize=12)

    ax = matplotlib.pyplot.gca()

    plt.xlim([0, x_max if realtime else num_timesteps])
    if ylim is not None:
        plt.ylim(ylim)
    elif y0_min > 0.0:
        plt.ylim(0.0, ax.get_ylim()[1])

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
    plt.subplots_adjust(left=m + s + 0.02, bottom=m, right=1-m + s, top=1-m)
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

    print(f'Generated {n_success}/{n_total} plots.')


if __name__ == '__main__':
    main()
