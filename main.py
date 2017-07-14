import sys
sys.path.append("games/")
sys.path.append("utils/")
import numpy as np 
from random import randint
import time
import tensorflow as tf
from DQNManager import DQNManager, GameManager
import os
# from configobj import ConfigObj
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

def trainMDRQN(cfg_parser, data_dir):
	# Shared sess object. Each sess assumes exclusive control of resources, can cause conflicts if multiple independent sesss.
	sess = tf.InteractiveSession()
	data_dir_mdrqn = os.path.join(data_dir,'mdrqn_results')
	if not os.path.exists(data_dir_mdrqn):
			os.makedirs(data_dir_mdrqn)

	game_mgr = GameManager(cfg_parser = cfg_parser, sess = sess)
	dqn_mgr = DQNManager(cfg_parser = cfg_parser, n_teacher_dqns = 0, enable_mdrqn = True, enable_distiller = False,
						n_actions = game_mgr.n_actions, game_mgr = game_mgr, sess = sess)

	# Train teacher DQNs	
	_, multitask_data_dict = dqn_mgr.train_mdrqn(game_mgr = game_mgr, mdrqn = dqn_mgr.mdrqn)#, data_dir = os.path.join(data_dir,'mdqn.txt'))
	print '----- MDRQN COMPLETE. Saving multitask game results!'
	for i_game in xrange(0,len(game_mgr.games)):
		dir_game_results = os.path.join(data_dir_mdrqn,'mdrqn_jointvalue_game_'+str(i_game)+'.txt')
		print 'Saving ', dir_game_results
		multitask_data_dict['game_'+str(i_game)].saveData(data_dir = dir_game_results)

def trainWithDistillation(cfg_parser, data_dir):
	# Shared sess object. Each sess assumes exclusive control of resources, can cause conflicts if multiple independent sesss.
	sess = tf.InteractiveSession()

	# Init game and DQN
	game_mgr = GameManager(cfg_parser = cfg_parser, sess = sess)
	dqn_mgr = DQNManager(cfg_parser = cfg_parser, n_teacher_dqns = len(game_mgr.games), enable_mdrqn = False, enable_distiller = True, enable_double_distiller = False,
						n_actions = game_mgr.n_actions, game_mgr = game_mgr, sess = sess)
		
	# Automatically loads checkpoint if data_dir contains it. Otherwise, starts fresh.
	tf_saver = TfSaver(sess = sess, data_dir = data_dir, vars_to_restore = dqn_mgr.teacher_vars)

	should_distill = False
	if should_distill:
		data_dir_distillation = os.path.join(data_dir,'distilled_results')
		# Just sets up the saver, but doesn't restore distillation network (so training is done from scratch each time)
		tf_saver_distillation = TfSaver(sess=sess, data_dir = data_dir_distillation, vars_to_restore = dqn_mgr.all_vars, try_to_restore = False)
		time.sleep(10)
	
	# Skip training if tf_saver loaded pre-trained teachers
	if not tf_saver.pre_trained:
		for i_game, game in enumerate(game_mgr.games):
			q_value_traj, joint_value_traj, init_q_value_traj = dqn_mgr.train_teacher_dqn(i_game = i_game, game = game)
			q_value_traj.saveData(data_dir = os.path.join(data_dir,'teacher_qvalue_game_'+str(i_game)+'.txt'))
			if type(init_q_value_traj) is list:
				for i_agt, q_values in enumerate(init_q_value_traj):
					q_values.saveData(data_dir = os.path.join(data_dir,'teacher_init_qvalue_game_'+str(i_game)+'_agt_'+str(i_agt)+'.txt'))
			else:	
				init_q_value_traj.saveData(data_dir = os.path.join(data_dir,'teacher_init_qvalue_game_'+str(i_game)+'.txt'))
			joint_value_traj.saveData(data_dir = os.path.join(data_dir,'teacher_jointvalue_game_'+str(i_game)+'.txt'))

		tf_saver.save_sess(timestep = 1, save_freq = 1)
	else:
		m_plotter = Plotter(label_x = 'iter', label_y = 'Actual Value Received', title = '', adjust_right = 0.73)
		m_plotter.update_palette(len(game_mgr.games))
		for i_game in xrange(0,len(game_mgr.games)):
			m_plotter.add_data_to_plot(data_dir = data_dir, data_file = 'teacher_jointvalue_game_' + str(i_game) +'.txt', label = 'Task ' + str(i_game))
			m_plotter.update_legend()
		plt.show()

	# Distill DQNs
	if should_distill:
		multitask_data_dict = dqn_mgr.distill_dqns(game_mgr = game_mgr)
		print '----- DISTILLATION COMPLETE. Saving multitask game results!'
		for i_game in xrange(0,len(game_mgr.games)):
			dir_game_results = os.path.join(data_dir_distillation,'distilled_jointvalue_game_'+str(i_game)+'.txt')
			print 'Saving ', dir_game_results
			multitask_data_dict['game_'+str(i_game)].saveData(data_dir = dir_game_results)
		tf_saver_distillation.save_sess(timestep = 1, save_freq = 1)

def main():
	data_dir = None # Choose this to train fresh teachers
	# data_dir = './results/multiagent_2017_02_14_11_57_18_forced_color_obs_on_both_tasks' # Choose this to load pre-trained teachers
	# data_dir = './results/multiagent_2017_02_14_02_41_00_4x4_color_obs_good_individual_tasks' # Choose this to load pre-trained teachers
	# data_dir = './results/multiagent_2017_02_15_23_01_17_onehot_color_obs'
	# data_dir = './results/multiagent_2017_02_16_02_35_34'
	# data_dir = './results/multiagent_2017_02_20_22_53_23_mamt_hdrqn_3x3_to_6x6_2agt_obs30_final'

	# data_dir is empty, so train for new game
	if data_dir == None:
		cfg_ini = 'config_SingleAgentSingleTargetPursuit.ini'
		# cfg_ini = 'config_MultiagentTargetPursuit_2player.ini'
		# cfg_ini = 'config_MultiagentTargetPursuit_3player.ini'

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
			if os.path.isfile(os.path.join(data_dir,filepath)) and 'config_' in filepath:
				total_config_files += 1
				cfg_ini = filepath
		# Make sure only a single config file exists, otherwise something is wrong
		assert total_config_files == 1

		# Load the cfg for the existing data_dir
		cfg_path = os.path.join(data_dir,cfg_ini)
		cfg_parser = get_cfg_parser(cfg_path)
		with open(cfg_path, 'r') as fin:
			print fin.read()
		print '\n------ Successfully found specified directory and loaded', cfg_path, '\n'

	trainWithDistillation(cfg_parser, data_dir)
	# trainMDRQN(cfg_parser, data_dir)

if __name__ == '__main__':
	main()