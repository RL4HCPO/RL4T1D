import os
import csv
import shutil
import pandas as pd
import numpy as np

import logging
import torch
from utils.core import combined_shape
from metrics.metrics import time_in_range
import json
from omegaconf import OmegaConf
import mlflow

def set_logger(LOG_DIR):
    log_filename = LOG_DIR + '/debug.log'
    #logging.basicConfig(filename=log_filename, filemode='a', format='%(levelname)s - %(message)s', level=logging.INFO)


def setup_folders(args: dict) -> None:  # create the folder which will save experiment data.
    LOG_DIR = args.experiment.experiment_dir
    CHECK_FOLDER = os.path.isdir(LOG_DIR)
    if CHECK_FOLDER:
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR + '/checkpoints')
    os.makedirs(LOG_DIR + '/training/')
    os.makedirs(LOG_DIR + '/testing/')
    set_logger(LOG_DIR)

    with open(args.experiment.experiment_dir + '/args.json', 'w') as fp:  # save the experiments args.
        json.dump(OmegaConf.to_container(args, resolve=True), fp, indent=4)
        fp.close()

    # copy running agent code to outputs
    #copy_folder(src=MAIN_PATH + '/agents/algorithm/'+ self.opt.agent, dst=MAIN_PATH + '/results/' + self.opt.experiment_folder + '/code')


def copy_folder(src, dst):
    for folders, subfolders, filenames in os.walk(src):
        for filename in filenames:
            shutil.copy(os.path.join(folders, filename), dst)


def save_log(directory, file, data):
    with open(directory + '/' + file + '.csv', 'a+') as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(data)
        f.close()


class LogExperiment:
    def __init__(self, args):
        self.args = args
        self.model_logs = torch.zeros(7, device=self.args.device)
        save_log(self.args.experiment_dir, [['policy_grad', 'value_grad', 'val_loss', 'exp_var', 'true_var', 'pi_loss', 'avg_rew', 'constraint']], '/model_log')
        save_log(self.args.experiment_dir, [['status', 'rollout', 't_rollout', 't_update', 't_test']], '/experiment_summary')

    def save(self, log_name, data):
        save_log(self.args.experiment_dir, data, log_name)


class LogWorker:
    def __init__(self, args, mode, worker_id):
        self.args = args
        self.keys = keys
        self.worker_mode = mode
        self.worker_id = worker_id
        self.episode_history = np.zeros(combined_shape(args.max_epi_length, 13), dtype=np.float32)

    def update(self, counter, episode, state, policy_step, pump_action, rl_action, reward, info):
        self.episode_history[counter] = [episode, counter, state.CGM,
                                                  info['meal'] * info['sample_time'],
                                                  pump_action, reward, rl_action, policy_step['mu'][0],
                                                  policy_step['std'][0],
                                                  policy_step['log_prob'][0], policy_step['state_value'][0],
                                                  info['day_hour'],
                                                  info['day_min']]

    def save(self, episode, counter):

        # log raw data of the episode
        df = pd.DataFrame(self.episode_history[0:counter], columns=self.keys['worker_episode'])

        df.to_csv(self.args.experiment_dir + '/' + self.worker_mode + '/worker_episode_' + str(self.worker_id) + '.csv',
                  mode='a', header=False, index=False)

        if self.args.mlflow_track:
            mlflow.log_table(data=df, artifact_file='logs_worker_' + str(self.worker_id) + '.json')

        # log the summary stats for the episode (rollout)
        normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'])
        save_log(self.args.experiment_dir,
                [[episode, counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper, 0, 0]],
                '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_' + str(self.worker_id))
        
        return counter, normo
