import matplotlib.pyplot as plt
import numpy as np
import os
import itertools


class Plotter:
	def __init__(self, label_x ='', label_y='', title='', adjust_right=None):
		self.plot_rank = 0
		self.fig, self.ax = plt.subplots(1, 1)
		if adjust_right:
			self.fig.subplots_adjust(right=adjust_right)
		self.ax.set_xlabel(label_x)
		self.ax.set_ylabel(label_y)
		plt.title(title)
		self.hl = list()

	def add_shaded_err_plot(self, x, y_mean, y_stdev=0, label=''):
		plt.sca(self.ax)
		hl_new, = plt.plot(x, y_mean, label=label)
		y_upper = y_mean - y_stdev
		y_lower = y_mean + y_stdev

		self.filled_lines = self.ax.fill_between(x, y_lower, y_upper, facecolor=hl_new.get_color(), alpha=0.2)

		self.hl.append(hl_new)
		self.plot_rank += 1

	def add_data_to_plot(self, data_dir, data_file, label):
		x_data, y_mean_data, y_stdev_data = np.loadtxt(os.path.join(data_dir, data_file))
		self.add_shaded_err_plot(x=x_data, y_mean=y_mean_data, y_stdev=y_stdev_data, label=label)
		return y_mean_data[-1], y_stdev_data[-1]

	def update_legend(self):
		plt.sca(self.ax)
		plt.legend(handles=[hl for hl in self.hl], loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
