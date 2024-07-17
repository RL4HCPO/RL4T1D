import gc
import abc
import time
import torch

from metrics.metrics import time_in_range
from metrics.statistics import calc_stats


from decouple import config
MAIN_PATH = config('MAIN_PATH')

import pandas as pd
from omegaconf import OmegaConf, open_dict


class Agent:
    def __init__(self, args, env_args, type="None"):
        self.args = args
        self.env_args = env_args
        self.agent_type = type

        with open_dict(self.args):  # TODO: the interface between env - agent, improve?
            self.args.n_features = len(env_args.obs_features)
            self.args.obs_window = env_args.obs_window
            self.args.glucose_max = env_args.glucose_max  # Note the selected sensors range would affect this
            self.args.glucose_min = env_args.glucose_min
            self.args.n_action = env_args.n_actions
            self.args.patient_id = env_args.patient_id

        self.policy = None
        self.buffer = None

        # workers run the simulations. For each worker an env is created, and the worker ID should be unique.
        self.n_training_workers = args.n_training_workers
        self.n_testing_workers = args.n_testing_workers

        # The offset params above are for visual convenience of raw logs when going through worker logs which are saved as:
        # e.g., worker_10.csv, worker_5000.csv, workers with 5000+ are testing; workers with 6000+ are validation
        self.training_agent_id_offset = 5  # 5, 6, 7, ... (5+n_training_workers)
        self.testing_agent_id_offset = 5000  # 5000, 5001, 5002, ... (5000+n_testing_workers)
        self.validation_agent_id_offset = 6000  # 6000, 6001, 6002, ... (6000+n_val_trials)

        if args.debug:  # TODO: a better way to approach the debug case?
            self.n_testing_workers = 2
            self.n_training_workers = 2
            self.total_interactions = 4000
            self.n_interactions_lr_decay = 2000
            self.n_val_trials = 3

        # initialise workers
        if type == "OnPolicy":
            from utils.worker import OnPolicyWorker as Worker
            self.training_agents = [Worker(args=self.args, env_args=self.env_args, mode='training',
                                           worker_id=i+self.training_agent_id_offset) for i in range(self.n_training_workers)]
            self.testing_agents = [Worker(args=self.args, env_args=self.env_args, mode='testing',
                                          worker_id=i+self.testing_agent_id_offset) for i in range(self.n_testing_workers)]
            self.validation_agents = [Worker(args=self.args, env_args=self.env_args, mode='testing',
                                             worker_id=i + self.validation_agent_id_offset) for i in range(self.n_val_trials)]
        elif type == "OffPolicy":
            from utils.worker import OffPolicyWorker as Worker
            self.training_agents = [Worker(args=self.args, env_args=self.env_args, mode='training',
                                           worker_id=i+self.training_agent_id_offset) for i in range(self.n_training_workers)]
            self.testing_agents = [Worker(args=self.args, env_args=self.env_args, mode='testing',
                                          worker_id=i+self.testing_agent_id_offset) for i in range(self.n_testing_workers)]
            self.validation_agents = [Worker(args=self.args, env_args=self.env_args, mode='testing',
                                             worker_id=i + self.validation_agent_id_offset) for i in range(self.n_val_trials)]
        else:
            print("specify agent type")
            exit()

    @abc.abstractmethod
    def update(self):
        """
        Implement the update rule.
        """

    def run(self):
        # learning
        rollout, completed_interactions = 0, 0
        while completed_interactions < self.total_interactions:  # steps * n_workers * epochs.
            tstart = time.perf_counter()
            for i in range(self.n_training_workers):  # run training workers to collect data

                # TODO: handle buffers better
                if self.agent_type == "OnPolicy":
                    self.training_agents[i].rollout(policy=self.policy, buffer=self.buffer.RolloutWorker)
                    self.buffer.save_rollout(training_agent_index=i)
                else:
                    self.training_agents[i].rollout(policy=self.policy, buffer=self.buffer)

            self.update()  # update the models
            self.policy.save(rollout)  # save model weights as checkpoints

            # testing: run testing workers on the validation scenario
            with torch.no_grad():
                for i in range(self.n_testing_workers):
                    self.testing_agents[i].rollout(policy=self.policy, buffer=None)  # these logs will be saved by the worker.

            # update the total number of completed interactions.
            completed_interactions += (self.args.n_step * self.n_training_workers)
            rollout += 1
            gc.collect()  # garbage collector to clean unused objects.

            # decay lr and set entropy coeff to zero to stabilise the policy towards the end.
            if completed_interactions > self.n_interactions_lr_decay:
                self.decay_lr()

            experiment_done = True if completed_interactions > self.total_interactions else False

            # logging
            print('\n---------------------------------------------------------')
            print('Training Progress: {:.2f}%, Elapsed time: {:.4f} minutes.'.format(min(100.00, (completed_interactions/self.total_interactions)*100),
                                                                                     (time.perf_counter() - tstart)/60))
            print('---------------------------------------------------------')

            # print('Rollout Time (seconds): {}, update: {}, testing: {}'.format((t2 - t1), (t4 - t2), (t6 - t4))) if self.args.verbose else None
            # self.LogExperiment.save(log_name='/experiment_summary', data=[[experiment_done, rollout, (t2 - t1), (t4 - t2), (t6 - t4)]])

            # when training complete conduct final validation: typically n=500.
            if experiment_done:
                self.evaluate()

    def evaluate(self):
        print('\n---------------------------------------------------------')
        print('===> Starting Validation Trials ....')
        
        with torch.no_grad():
            for i in range(self.n_val_trials):
                self.validation_agents[i].rollout(policy=self.policy, buffer=None)

            # calculate the final metrics.
            cohort_res, summary_stats = [], []
            secondary_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']
            data = []
            FOLDER_PATH = '/results/'+self.args.experiment_folder+'/testing/data'
            for i in range(0, self.n_val_trials):
                test_i = 'logs_worker_'+str(self.validation_agent_id_offset+i)+'.csv'
                df = pd.read_csv(MAIN_PATH +FOLDER_PATH+ '/'+test_i)
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
        return
        # self.entropy_coef = 0  # self.entropy_coef / 100
        # self.pi_lr = self.pi_lr / 10
        # self.vf_lr = self.vf_lr / 10
        # for param_group in self.optimizer_Actor.param_groups:
        #     param_group['lr'] = self.pi_lr
        # for param_group in self.optimizer_Critic.param_groups:
        #     param_group['lr'] = self.vf_lr
