import tensorflow as tf
from trainer import Trainer
import os
import shutil
from configparser import SafeConfigParser
import argparse
import logging


def configure_logger(log_file):
	logger = logging.getLogger()
	[logger.removeHandler(h) for h in logger.handlers]
	logger.setLevel(logging.DEBUG)

	console = logging.StreamHandler()
	logger.addHandler(console)

	fh = logging.FileHandler(filename=log_file)
	logger.addHandler(fh)

	formatter = logging.Formatter('%(message)s')
	for h in logger.handlers:
		h.setFormatter(formatter)

	return logger


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('job_name', type=str)
	parser.add_argument('--config', type=str, default='configs/Atari.ini')
	parser.add_argument('--overwrite', action='store_true')
	args = parser.parse_args()

	data_dir = os.path.join('results', args.job_name)

	if args.overwrite and os.path.exists(data_dir):
		print('Deleting', data_dir)
		shutil.rmtree(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

		log_file = os.path.join(data_dir, 'output.txt')
		open(log_file, 'a').close()

		logger = configure_logger(log_file)
		logger.info('Creating new results directory: {}'.format(data_dir))

		shutil.copy(args.config, data_dir)
		config = SafeConfigParser()
		config.read(args.config)

		trainer = Trainer(config)
		trainer.train()
		trainer.save_results(data_dir)

	else:
		print('{} already exists'.format(data_dir))
		print('Use --overwrite to remove the old job')


if __name__ == '__main__':
	main()
