import gc
import abc
import time
import torch
import random

from utils.worker import OnPolicyWorker, OffPolicyWorker
from utils.buffers import onpolicy_buffers, offpolicy_buffers
from metrics.metrics import time_in_range
from metrics.statistics import calc_stats
from utils.worker import OnPolicyWorker as Worker
from utils.core import get_flat_params_from, set_flat_params_to, compute_flat_grad

from decouple import config
MAIN_PATH = config('MAIN_PATH')

import pandas as pd
# import wandb


class Agent:
    def __init__(self, args, env_args, logger, type="None"):
        self.args = args
        self.env_args = env_args
        self.agent_type = type
        self.policy = None

        # workers run the simulations. For each worker an env is created, and the worker ID should be unique.
        self.n_training_workers = args.n_training_workers
        self.n_testing_workers = args.n_testing_workers
        self.total_interactions = args.total_interactions
        self.n_interactions_lr_decay = args.n_interactions_lr_decay
        self.n_val_trials = args.n_val_trials

        # The offset params above are for visual convenience of raw logs when going through worker logs which are saved as:
        # e.g., worker_10.csv, worker_5000.csv, workers with 5000+ are testing; workers with 6000+ are validation
        self.training_agent_id_offset = 5  # 5, 6, 7, ... (5+n_training_workers)
        self.testing_agent_id_offset = 5000  # 5000, 5001, 5002, ... (5000+n_testing_workers)
        self.validation_agent_id_offset = 6000  # 6000, 6001, 6002, ... (6000+n_val_trials)
        self.completed_interactions = 0
        self.best = 0
        self.current = 0
        self.best_normo = 0
        self.current_normo = 0
        self.best_params = None
        self.temperature = 1

        # initialise workers and buffers
        if type == "OnPolicy":
            self.training_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='training',
                                           worker_id=i+args.training_agent_id_offset) for i in range(self.args.n_training_workers)]
            self.testing_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing',
                                          worker_id=i+args.testing_agent_id_offset) for i in range(self.args.n_testing_workers)]
            self.validation_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing',
                                             worker_id=i + args.validation_agent_id_offset) for i in range(self.args.n_val_trials)]
            self.buffer = onpolicy_buffers.RolloutBuffer(self.args)

        elif type == "OffPolicy":
            self.training_agents = [OffPolicyWorker(args=self.args, env_args=self.env_args, mode='training',
                                           worker_id=i+args.training_agent_id_offset) for i in range(self.args.n_training_workers)]
            self.testing_agents = [OffPolicyWorker(args=self.args, env_args=self.env_args, mode='testing',
                                          worker_id=i+args.testing_agent_id_offset) for i in range(self.args.n_testing_workers)]
            self.validation_agents = [OffPolicyWorker(args=self.args, env_args=self.env_args, mode='testing',
                                             worker_id=i + args.validation_agent_id_offset) for i in range(self.args.n_val_trials)]
            self.buffer = offpolicy_buffers.ReplayMemory(self.args)

        self.logger = logger

    @abc.abstractmethod
    def update(self):
        """
        Implement the update rule.
        """

    def run(self):
        # initialise workers for training
        training_agents = [Worker(args=self.args, env_args=self.env_args, mode='training', worker_id=i+self.training_agent_id_offset)
                           for i in range(self.n_training_workers)]

        # initialise workers for testing after each update step
        testing_agents = [Worker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i+self.testing_agent_id_offset)
                          for i in range(self.n_testing_workers)]

        # start learning
        rollout, self.completed_interactions = 0, 0
        while self.completed_interactions < self.total_interactions:  # steps * n_workers * epochs. 3000 is just a large number
            tstart = time.perf_counter()
            for i in range(self.args.n_training_workers):  # run training workers to collect data

                # TODO: handle buffers better
                if self.agent_type == "OnPolicy":
                    self.training_agents[i].rollout(policy=self.policy, buffer=self.buffer.Rollout, logger=self.logger.logWorker)
                    self.buffer.save_rollout(training_agent_index=i)
                else:
                    self.training_agents[i].rollout(policy=self.policy, buffer=self.buffer, logger=self.logger.logWorker)

            logs = self.update()  # update the models
            self.logger.save_rollout(logs)  # logging
            self.policy.save(rollout)  # save model weights as checkpoints

            # testing: run testing workers on the validation scenario
            with torch.no_grad():
                counter_list = []
                normo_list = []
                for i in range(self.n_testing_workers):
                    counter, normo = testing_agents[i].rollout(policy=self.policy, buffer=None)  # these logs will be saved by the worker.
                    counter_list.append(counter)
                    normo_list.append(normo)
                
            counter_mean = sum(counter_list) / len(counter_list)
            normo_mean = sum(normo_list)/ len(normo_list)
            self.current = counter_mean
            self.current_normo =  normo_mean
            if(counter_mean >= self.best):
                self.best = counter_mean
                self.best_normo = normo_mean
                self.best_params = get_flat_params_from(self.policy.Actor)

            randnum = random.random()
            current_params = get_flat_params_from(self.policy.Actor)
            print('randnum: {}, temperature: {}, avg_t: {}, best_avg_t: {}, avg_normo: {}, best_avg_normo: {}.'.format(randnum, self.temperature, self.current, self.best, self.current_normo, self.best_normo))
            # SRPO Rollback 
            if(self.completed_interactions > 400000  and self.current <= self.best and self.best_params != None and not torch.equal(self.best_params, current_params)):
                self.temperature *= 0.95
                if(randnum > self.temperature):
                    print('Early stop => randnum: {}, temperature: {}, avg_t: {}, best_avg_t: {}, avg_normo: {}, best_avg_normo: {}.'.format(randnum, self.temperature, self.current, self.best, self.current_normo, self.best_normo))
                    set_flat_params_to(self.policy.Actor, self.best_params)

            # update the total number of completed interactions.
            self.completed_interactions += (self.args.n_step * self.n_training_workers)
            rollout += 1
            # print('completed interactions', self.completed_interactions)
            gc.collect()  # garbage collector to clean unused objects.

            # decay lr and set entropy coeff to zero to stabilise the policy towards the end.
            if self.completed_interactions == self.n_interactions_lr_decay:
                self.decay_lr()

            experiment_done = True if self.completed_interactions > self.total_interactions else False

            # logging
            print('\n---------------------------------------------------------')
            print('Training Progress: {:.2f}%, Elapsed time: {:.4f} minutes.'.format(min(100.00, (self.completed_interactions/self.total_interactions)*100),
                                                                                     (time.perf_counter() - tstart)/60))
            print('---------------------------------------------------------')

            # when training complete conduct final validation: typically n=500.
            if experiment_done:
                set_flat_params_to(self.policy.Actor, self.best_params)
                self.evaluate()

    def evaluate(self):  # TODO: refactor below
        print('\n---------------------------------------------------------')
        print('===> Starting Validation Trials ....')
        
        with torch.no_grad():
            for i in range(self.args.n_val_trials):
                self.validation_agents[i].rollout(policy=self.policy, buffer=None, logger=self.logger.logWorker)

            # calculate the final metrics.
            cohort_res, summary_stats = [], []
            secondary_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']
            data = []
            FOLDER_PATH = self.args.experiment_folder+'/testing/'
            for i in range(0, self.args.n_val_trials):
                test_i = 'worker_episode_'+str(self.args.validation_agent_id_offset+i)+'.csv'
                df = pd.read_csv(FOLDER_PATH+ '/'+test_i)
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'])
                reward_val = df['rew'].sum()*(100/288)
                e = [[i, df.shape[0], reward_val, normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper, 0, 0]]
                dataframe = pd.DataFrame(e, columns=secondary_columns)
                data.append(dataframe)
            res = pd.concat(data)
            res['PatientID'] = self.args.patient_id
            res.rename(columns={'sev_hypo':'S_hypo', 'sev_hyper':'S_hyper'}, inplace=True)
            summary_stats.append(res)
            metric=['mean', 'std', 'min', 'max']
            print(calc_stats(res, metric=metric, sim_len=288))

            print('\nAlgorithm Training/Validation Completed Successfully.')
            print('---------------------------------------------------------')
            exit()

    def decay_lr(self):
        self.entropy_coef = 0  # self.entropy_coef / 100
        self.pi_lr = self.pi_lr / 10
        self.vf_lr = self.vf_lr / 10
        for param_group in self.optimizer_Actor.param_groups:
            param_group['lr'] = self.pi_lr
        for param_group in self.optimizer_Critic.param_groups:
            param_group['lr'] = self.vf_lr