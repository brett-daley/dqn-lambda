import sys
sys.path.append('games/')
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


def load_plots(data_dir):
	plots = [['Predicted Discounted Episode Return', 'traj_predicted_disc_return.txt'],
			 ['Actual Discounted Episode Return', 'traj_actual_disc_return.txt'],
			 ['Undiscounted Episode Return', 'traj_undisc_return.txt'],
			 ['Undiscounted Episode Return (Moving Avg)', 'traj_mov_avg_disc_return.txt']]

	for title, data_file in plots:
		data_file_path = os.path.join(data_dir, data_file)

		if not os.path.exists(data_file_path):
			print 'Could not find', data_file_path, '... skipping'
			continue

		p = Plotter(label_x='Timestep', label_y='Return', title=title, adjust_right=0.73)
		p.update_palette(n_colors=1)
		p.add_data_to_plot(data_dir, data_file, label=None)
		p.update_legend()
		print 'Successfully generated plot from', data_file_path

	plt.show(block=True)


def train(cfg_parser, data_dir):
	sess = tf.InteractiveSession()

	game = Atari(cfg_parser, sess)
	dqn_mgr = DQNManager(cfg_parser, sess, game, n_actions=game.n_actions)

	traj_predicted_disc_return, traj_actual_disc_return, traj_undisc_return, traj_mov_avg_undisc_return = dqn_mgr.train_dqn(game)

	traj_predicted_disc_return.saveData(data_dir=os.path.join(data_dir, 'traj_predicted_disc_return.txt'))
	traj_actual_disc_return.saveData(data_dir=os.path.join(data_dir, 'traj_actual_disc_return.txt'))
	traj_undisc_return.saveData(data_dir=os.path.join(data_dir, 'traj_undisc_return.txt'))
	traj_mov_avg_undisc_return.saveData(data_dir=os.path.join(data_dir, 'traj_mov_avg_disc_return.txt'))

	saver = tf.train.Saver()
	saver.save(sess, save_path=os.path.join(data_dir, 'model'))
	print 'Successfully saved Tensorflow model in', data_dir


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('job_name', type=str)
	parser.add_argument('--overwrite', action='store_true')
	# TODO: add option for disabling rendering
	args = parser.parse_args()

	data_dir = os.path.join('results', args.job_name)

	if args.overwrite and os.path.exists(data_dir):
		print 'Deleting', data_dir
		shutil.rmtree(data_dir)

	if not os.path.exists(data_dir):
		print 'Creating new results directory:', data_dir

		os.makedirs(data_dir)

		# TODO: don't hardcode config name
		cfg_path = './games/config_Atari.ini'
		cfg_parser = SafeConfigParser()
		cfg_parser.read(cfg_path)

		# Backup the config in the new data directory
		cfg_path_backup = os.path.join(data_dir, os.path.basename(cfg_path))
		shutil.copyfile(cfg_path, cfg_path_backup)

		train(cfg_parser, data_dir)

	else:
		config_list = glob.glob(os.path.join(data_dir, 'config_*'))
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
