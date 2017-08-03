import sys
sys.path.append('games/')
sys.path.append('nn_archs/')
sys.path.append('utils/')
import time
import tensorflow as tf
from DQNManager import DQNManager
from Atari import Atari
import os
from ConfigParser import SafeConfigParser
from utils_general import TfSaver
from plotting import Plotter
import matplotlib.pyplot as plt


def get_cfg_parser(game_ini_path):
	cfg_parser = SafeConfigParser()
	cfg_parser.read(game_ini_path)
	return cfg_parser

def createDataDir():
	data_dir = './results/multiagent_'+time.strftime("%Y_%m_%d_%H_%M_%S")
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	return data_dir

def train(cfg_parser, data_dir):
	# Shared sess object. Each sess assumes exclusive control of resources, can cause conflicts if multiple independent sesss.
	sess = tf.InteractiveSession()

	# Init game and DQN
	game = Atari(cfg_parser=cfg_parser, sess=sess)
	dqn_mgr = DQNManager(cfg_parser=cfg_parser, n_actions=game.n_actions, game=game, sess=sess)

	# Automatically loads checkpoint if data_dir contains it. Otherwise, starts fresh.
	tf_saver = TfSaver(sess=sess, data_dir=data_dir, vars_to_restore=dqn_mgr.teacher_vars)

	# Skip training if tf_saver loaded pre-trained model
	if not tf_saver.pre_trained:
		q_value_traj, value_traj, init_q_value_traj = dqn_mgr.train_dqn(game)
		q_value_traj.saveData(data_dir=os.path.join(data_dir, 'teacher_qvalue.txt'))
		init_q_value_traj[0].saveData(data_dir=os.path.join(data_dir, 'teacher_init_qvalue.txt'))
		value_traj.saveData(data_dir=os.path.join(data_dir, 'teacher_value.txt'))

		tf_saver.save_sess(timestep=1, save_freq=1)
	else:
		m_plotter = Plotter(label_x='iter', label_y='Actual Value Received', title='', adjust_right=0.73)
		m_plotter.update_palette(n_colors=1)
		m_plotter.add_data_to_plot(data_dir=data_dir, data_file='teacher_value.txt', label='Task')
		m_plotter.update_legend()
		plt.show()

def main():
	# Change this to load a pre-trained model
	data_dir = None

	# data_dir is empty, so train for new game
	if not data_dir:
		cfg_ini = 'config_Atari.ini'

		# Read the config file and create a data directory for it
		cfg_parser = get_cfg_parser(os.path.join('./games/', cfg_ini))
		data_dir = createDataDir()

		# Backup the config ini in the new data directory
		cfg_path_backup = os.path.join(data_dir, cfg_ini)

		with open(cfg_path_backup, 'wb') as configfile:
			cfg_parser.write(configfile)
		print '\n------ No data directory specified, created new data_dir', data_dir, '\n'
	# data_dir has been specified, so look for a config file inside
	else:
		total_config_files = 0
		for filepath in os.listdir(data_dir):
			if os.path.isfile(os.path.join(data_dir, filepath)) and 'config_' in filepath:
				total_config_files += 1
				cfg_ini = filepath
		# Make sure only a single config file exists, otherwise something is wrong
		assert total_config_files == 1

		# Load the cfg for the existing data_dir
		cfg_path = os.path.join(data_dir, cfg_ini)
		cfg_parser = get_cfg_parser(cfg_path)
		with open(cfg_path, 'r') as fin:
			print fin.read()
		print '\n------ Successfully found specified directory and loaded', cfg_path, '\n'

	train(cfg_parser, data_dir)

if __name__ == '__main__':
	main()
