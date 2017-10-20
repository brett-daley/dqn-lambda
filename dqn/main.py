import sys
sys.path.append('envs/')
sys.path.append('nn_archs/')
sys.path.append('utils/')
import tensorflow as tf
from DQNManager import DQNManager
from Atari import Atari
import os
import glob
import shutil
from ConfigParser import SafeConfigParser
from plotting import Plotter
import matplotlib.pyplot as plt
import argparse
import logging
import warnings; warnings.filterwarnings('ignore')


def configure_logger(log_file):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)

	fh = logging.FileHandler(filename=log_file)
	logger.addHandler(fh)

	formatter = logging.Formatter('%(message)s')
	for h in logger.handlers:
		h.setFormatter(formatter)

	return logger


def load_plots(data_dir):
	plots = [['Predicted Discounted Episode Return', 'predicted_disc_return'],
			 ['Actual Discounted Episode Return', 'actual_disc_return'],
			 ['Undiscounted Episode Return', 'undisc_return'],
			 ['Undiscounted Episode Return (Moving Avg)', 'mov_avg_undisc_return']]

	for plot_title, file_basename in plots:
		data_file = os.path.join(data_dir, 'traj_' + file_basename + '.txt')
		plot_file = os.path.join(data_dir, 'plot_' + file_basename + '.png')

		if not os.path.exists(data_file):
			print 'Could not find', data_file, '... skipping'
		else:
			p = Plotter(label_x='Timestep', label_y='Return', title=plot_title, adjust_right=0.73)
			p.update_palette(n_colors=1)
			p.add_data_to_plot('.', data_file, label=None)
			p.update_legend()
			print 'Successfully generated plot from', data_file

			if not os.path.exists(plot_file):
				p.fig.savefig(plot_file, bbox_inches='tight', pad_inches=0)
				print 'Saved plot in', plot_file
			else:
				print plot_file, 'already exists'

		print

	plt.show(block=True)


def train(cfg_parser, data_dir, render):
	sess = tf.InteractiveSession()

	game = Atari(cfg_parser, sess, render)
	dqn_mgr = DQNManager(cfg_parser, sess, game, render)

	traj_predicted_disc_return, traj_actual_disc_return, traj_undisc_return, traj_mov_avg_undisc_return = dqn_mgr.train_dqn(game)

	traj_predicted_disc_return.saveData(data_dir=os.path.join(data_dir, 'traj_predicted_disc_return.txt'))
	traj_actual_disc_return.saveData(data_dir=os.path.join(data_dir, 'traj_actual_disc_return.txt'))
	traj_undisc_return.saveData(data_dir=os.path.join(data_dir, 'traj_undisc_return.txt'))
	traj_mov_avg_undisc_return.saveData(data_dir=os.path.join(data_dir, 'traj_mov_avg_undisc_return.txt'))

	if render:
		dqn_mgr.dqn.plot_predicted_disc_return.fig.savefig(os.path.join(data_dir, 'plot_predicted_disc_return.png'), bbox_inches='tight')
		dqn_mgr.dqn.plot_actual_disc_return.fig.savefig(os.path.join(data_dir, 'plot_actual_disc_return.png'), bbox_inches='tight')
		dqn_mgr.dqn.plot_undisc_return.fig.savefig(os.path.join(data_dir, 'plot_undisc_return.png'), bbox_inches='tight')
		dqn_mgr.dqn.plot_mov_avg_undisc_return.fig.savefig(os.path.join(data_dir, 'plot_mov_avg_undisc_return.png'), bbox_inches='tight')

	saver = tf.train.Saver()
	saver.save(sess, save_path=os.path.join(data_dir, 'model'))
	logging.getLogger().info('Successfully saved Tensorflow model in {}'.format(data_dir))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('job_name', type=str)
	parser.add_argument('--overwrite', action='store_true')
	parser.add_argument('--no-render', action='store_true')
	args = parser.parse_args()

	data_dir = os.path.join('results', args.job_name)

	if args.overwrite and os.path.exists(data_dir):
		print 'Deleting', data_dir
		shutil.rmtree(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

		log_file = os.path.join(data_dir, 'output.txt')
		open(log_file, 'a').close()

		logger = configure_logger(log_file)
		logger.info('Creating new results directory: {}'.format(data_dir))

		# TODO: don't hardcode config name
		cfg_path = './configs/Atari.ini'
		cfg_parser = SafeConfigParser()
		cfg_parser.read(cfg_path)

		# Backup the config in the new data directory
		cfg_path_backup = os.path.join(data_dir, os.path.basename(cfg_path))
		shutil.copyfile(cfg_path, cfg_path_backup)

		train(cfg_parser, data_dir, render=(not args.no_render))

	else:
		config_list = glob.glob(os.path.join(data_dir, '*.ini'))
		# Make sure only a single config file exists, otherwise something is wrong
		assert len(config_list) == 1

		# Load and print the config for the existing directory
		cfg_path = config_list[0]
		with open(cfg_path, 'r') as fin:
			print '------ Successfully loaded', cfg_path
			print fin.read()
			print '------'

		load_plots(data_dir)


if __name__ == '__main__':
	main()
