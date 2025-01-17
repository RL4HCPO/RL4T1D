import gc
import csv
import time
import torch
import torch.nn as nn


from utils.worker import OffPolicyWorker as Worker
from utils.offpolicy_buffers import ReplayMemory, Transition
from agents.models.actor_critic_sac import ActorCritic


class SAC:
    def __init__(self, args, load_model, actor_path, critic_path):
        self.args = args
        self.gamma = args.gamma
        self.device = args.device
        self.n_step = args.n_step

        self.grad_clip = args.grad_clip

        self.n_training_workers = args.n_training_workers
        self.n_testing_workers = args.n_testing_workers
        self.training_agent_id_offset = 5  # 5, 6, 7, ... (5+n_training_workers)
        self.testing_agent_id_offset = 5000  # 5000, 5001, 5002, ... (5000+n_testing_workers)
        self.validation_agent_id_offset = 6000  # 6000, 6001, 6002, ... (6000+n_val_trials)
        # The offset params above are for visual convenience of raw logs when going through worker logs which are saved as:
        # e.g., worker_10.csv, worker_5000.csv, workers with 5000+ are testing; workers with 6000+ are validation

        self.replay_buffer_size = args.replay_buffer_size
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size

        self.target_update_interval = 1  # 100
        self.n_updates = 0

        self.soft_tau = args.soft_tau
        self.train_pi_iters = args.n_pi_epochs

        self.soft_q_lr = args.vf_lr
        self.policy_lr = args.pi_lr
        self.grad_clip = args.grad_clip

        self.shuffle_rollout = args.shuffle_rollout

        self.entropy_coef = args.entropy_coef
        self.target_entropy = args.target_entropy
        self.entropy_lr = args.entropy_lr
        self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * self.entropy_coef).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.entropy_lr)

        self.sac_v2 = args.sac_v2
        self.weight_decay = args.weight_decay

        ### SAC networks:
        self.sac = ActorCritic(args, load_model, actor_path, critic_path, args.device).to(self.device)
        print('Policy Parameters: {}'.format(sum(p.numel() for p in self.sac.policy_net.parameters() if p.requires_grad)))
        print('Q1 Parameters: {}'.format(sum(p.numel() for p in self.sac.soft_q_net1.parameters() if p.requires_grad)))
        print('Q2 Parameters: {}'.format(sum(p.numel() for p in self.sac.soft_q_net2.parameters() if p.requires_grad)))
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.soft_q_optimizer1 = torch.optim.Adam(self.sac.soft_q_net1.parameters(), lr=self.soft_q_lr, weight_decay=self.weight_decay)
        self.soft_q_optimizer2 = torch.optim.Adam(self.sac.soft_q_net2.parameters(), lr=self.soft_q_lr, weight_decay=self.weight_decay)
        self.policy_optimizer = torch.optim.Adam(self.sac.policy_net.parameters(), lr=self.policy_lr, weight_decay=self.weight_decay)

        if self.sac_v2:
            for target_param, param in zip(self.sac.target_q_net1.parameters(), self.sac.soft_q_net1.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.sac.target_q_net2.parameters(), self.sac.soft_q_net2.parameters()):
                target_param.data.copy_(param.data)
            for p in self.sac.target_q_net1.parameters():
                p.requires_grad = False
            for p in self.sac.target_q_net2.parameters():
                p.requires_grad = False
        else:
            self.value_criterion = nn.MSELoss()
            self.value_optimizer = torch.optim.Adam(self.sac.value_net.parameters(), lr=self.soft_q_lr, weight_decay=self.weight_decay)
            for target_param, param in zip(self.sac.value_net_target.parameters(), self.sac.value_net.parameters()):
                target_param.data.copy_(param.data)
            for p in self.sac.value_net_target.parameters():
                p.requires_grad = False

        self.replay_memory = ReplayMemory(self.replay_buffer_size)

        self.save_log([['coeff_loss', 'policy_loss', 'q1_loss', 'q2_loss', 'ent_coeff', 'pi_grad', 'q1_grad', 'g2_grad', 'coeff_grad']], '/model_log')
        self.model_logs = torch.zeros(9, device=self.device)
        self.save_log([['ri', 'alive_steps', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi', 'hgbi',
                        'sev_hyper', 'rollout', 'trial']], '/evaluation_log')
        self.save_log([['status', 'rollout', 't_rollout', 't_update', 't_test']], '/experiment_summary')
        self.save_log([[1, 0, 0, 0, 0]], '/experiment_summary')
        self.completed_interactions = 0

    def save_log(self, log_name, file_name):
        with open(self.args.experiment_dir + file_name + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()

    def update(self):
        if len(self.replay_memory) < self.sample_size * 10:
            return

        print('Running network update...')
        cl, pl, ql1, ql2, count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
                                  torch.zeros(1,device=self.device), \
                                  torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        pi_grad, q1_grad, q2_grad, coeff_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
                                                torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        for i in range(self.train_pi_iters):
            # sample from buffer

            transitions = self.replay_memory.sample(self.sample_size)
            batch = Transition(*zip(*transitions))

            cur_state_batch = torch.cat(batch.old_states)
            actions_batch = torch.cat(batch.action).unsqueeze(1)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)
            next_state_batch = torch.cat(batch.next_state)
            done_batch = torch.cat(batch.done).unsqueeze(1)

            actions_pi, log_prob = self.sac.evaluate_policy(cur_state_batch)
            self.entropy_coef = torch.exp(self.log_ent_coef.detach()) if self.sac_v2 else 0.001

            # value network update
            if not self.sac_v2:
                self.value_optimizer.zero_grad()
                with torch.no_grad():
                    min_qf_val = torch.min(self.sac.soft_q_net1(cur_state_batch, actions_pi),
                                           self.sac.soft_q_net2(cur_state_batch, actions_pi))
                predicted_value = self.sac.value_net(cur_state_batch)
                value_func_estimate = min_qf_val - (self.entropy_coef * log_prob)  # todo the temperature paramter
                value_loss = 0.5 * self.value_criterion(predicted_value, value_func_estimate.detach())
                value_loss.backward()
                coeff_grad += torch.nn.utils.clip_grad_norm_(self.sac.value_net.parameters(), self.grad_clip)
                self.value_optimizer.step()
                cl += value_loss.detach()

            # q network update
            self.soft_q_optimizer1.zero_grad()
            self.soft_q_optimizer2.zero_grad()
            with torch.no_grad():  # calculate the target q vals here.
                if self.sac_v2:
                    new_action, next_log_prob = self.sac.evaluate_policy(next_state_batch)
                    next_q_values = torch.min(self.sac.target_q_net1(next_state_batch, new_action),
                                              self.sac.target_q_net2(next_state_batch, new_action))
                    next_q_values = next_q_values - self.entropy_coef * next_log_prob
                    target_q_values = (reward_batch + (self.gamma * (1 - done_batch) * next_q_values))
                else:
                    target_value = self.sac.value_net_target(next_state_batch)
                    target_q_values = (reward_batch + self.gamma * (1 - done_batch) * target_value)

            predicted_q_value1 = self.sac.soft_q_net1(cur_state_batch, actions_batch)
            predicted_q_value2 = self.sac.soft_q_net2(cur_state_batch, actions_batch)
            q_value_loss1 = 0.5 * self.soft_q_criterion1(predicted_q_value1, target_q_values)
            q_value_loss2 = 0.5 * self.soft_q_criterion2(predicted_q_value2, target_q_values)
            q_value_loss1.backward()
            q1_grad += torch.nn.utils.clip_grad_norm_(self.sac.soft_q_net1.parameters(), self.grad_clip)
            self.soft_q_optimizer1.step()
            q_value_loss2.backward()
            q2_grad += torch.nn.utils.clip_grad_norm_(self.sac.soft_q_net2.parameters(), self.grad_clip)
            self.soft_q_optimizer2.step()

            # actor update : next q values
            # freeze q networks save compute: ref: openai:
            for p in self.sac.soft_q_net1.parameters():
                p.requires_grad = False
            for p in self.sac.soft_q_net2.parameters():
                p.requires_grad = False

            self.policy_optimizer.zero_grad()
            min_qf_pi = torch.min(self.sac.soft_q_net1(cur_state_batch, actions_pi),
                                  self.sac.soft_q_net2(cur_state_batch, actions_pi))

            policy_loss = (self.entropy_coef * log_prob - min_qf_pi).mean()
            policy_loss.backward()
            pi_grad += torch.nn.utils.clip_grad_norm_(self.sac.policy_net.parameters(), 10)
            self.policy_optimizer.step()

            # save compute: ref: openai:
            for p in self.sac.soft_q_net1.parameters():
                p.requires_grad = True
            for p in self.sac.soft_q_net2.parameters():
                p.requires_grad = True

            # entropy coeff update
            if self.sac_v2:
                self.ent_coef_optimizer.zero_grad()
                _, log_prob = self.sac.evaluate_policy(cur_state_batch)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_loss.backward()
                coeff_grad += torch.nn.utils.clip_grad_norm_([self.log_ent_coef], self.grad_clip)
                self.ent_coef_optimizer.step()
                cl += ent_coef_loss.detach()

            self.n_updates += 1

            if self.n_updates % self.target_update_interval == 0:
                with torch.no_grad():
                    print("################updated target")
                    if self.sac_v2:
                        for param, target_param in zip(self.sac.soft_q_net1.parameters(), self.sac.target_q_net1.parameters()):
                            target_param.data.mul_((1 - self.soft_tau))
                            target_param.data.add_(self.soft_tau * param.data)
                        for param, target_param in zip(self.sac.soft_q_net2.parameters(), self.sac.target_q_net2.parameters()):
                            target_param.data.mul_((1 - self.soft_tau))
                            target_param.data.add_(self.soft_tau * param.data)
                    else:
                        for param, target_param in zip(self.sac.value_net.parameters(), self.sac.value_net_target.parameters()):
                            target_param.data.mul_((1 - self.soft_tau))
                            target_param.data.add_(self.soft_tau * param.data)

            pl += policy_loss.detach()
            ql1 += q_value_loss1.detach()
            ql2 += q_value_loss2.detach()

        self.model_logs[0] = cl  # value loss or coeff loss
        self.model_logs[1] = pl
        self.model_logs[2] = ql1
        self.model_logs[3] = ql2
        self.model_logs[4] = self.entropy_coef
        self.model_logs[5] = pi_grad
        self.model_logs[6] = q1_grad
        self.model_logs[7] = q2_grad
        self.model_logs[8] = coeff_grad  # value loss grad or coeff loss grad
        self.save_log([self.model_logs.detach().cpu().flatten().numpy()], '/model_log')
        print('success')

    def decay_lr(self):
        self.soft_q_lr = self.soft_q_lr / 10
        self.policy_lr = self.policy_lr / 10
        self.entropy_lr = self.entropy_lr / 10
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = self.policy_lr
        for param_group in self.soft_q_optimizer1.param_groups:
            param_group['lr'] = self.soft_q_lr
        for param_group in self.soft_q_optimizer2.param_groups:
            param_group['lr'] = self.soft_q_lr

    def run(self):
        # initialise workers for training
        training_agents = [Worker(args=self.args, mode='training', worker_id=i+self.training_agent_id_offset)
                           for i in range(self.n_training_workers)]

        # initialise workers for testing after each update step
        testing_agents = [Worker(args=self.args, mode='testing', worker_id=i+self.testing_agent_id_offset)
                          for i in range(self.n_testing_workers)]

        # start sac learning
        for rollout in range(0, 30000):  # steps * n_workers * epochs
            print('\nExperiment: {}, Rollout {}: '.format(self.args.experiment_folder, rollout)) if self.args.verbose else None
            t1 = time.time()
            for i in range(self.n_training_workers):
                training_agents[i].rollout(policy=self.sac, buffer=self.replay_memory)
                self.update()  # maybe less frequent updates ? move outside the for loop
            t2 = time.time()
            self.sac.save(rollout)
            t5 = time.time()

            # testing
            self.sac.is_testing_worker = True
            for i in range(self.n_testing_workers):
                testing_agents[i].rollout(policy=self.sac, buffer=None)
            t6 = time.time()
            self.sac.is_testing_worker = False

            # update the total number of completed interactions.
            self.completed_interactions += (self.n_step * self.n_training_workers)
            gc.collect()

            # decay lr
            if self.completed_interactions > self.args.n_interactions_lr_decay:
                self.decay_lr()

            experiment_done = True if self.completed_interactions > self.args.total_interactions else False

            # logging and termination
            if self.args.verbose:
                print('\nExperiment: {}, Rollout {}: Time for rollout: {}, update: {}, '
                      'testing: {}'.format(self.args.experiment_folder, rollout, (t2 - t1), 0, (t6 - t5)))
            self.save_log([[experiment_done, rollout, (t2 - t1), 0, (t6 - t5)]], '/experiment_summary')

            if experiment_done:
                print('\n################## starting the validation trials #######################')
                validation_agents = [Worker(args=self.args, mode='testing', worker_id=i+self.validation_agent_id_offset)
                                     for i in range(self.args.n_val_trials)]
                for i in range(self.args.n_val_trials):
                    validation_agents[i].rollout(policy=self.sac, buffer=None)
                print('Algorithm Training/Validation Completed Successfully.')
                exit()
