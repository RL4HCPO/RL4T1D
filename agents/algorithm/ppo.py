import torch
import torch.nn as nn

from agents.algorithm.agent import Agent
from agents.models.actor_critic import ActorCritic


class PPO(Agent):
    def __init__(self, args, env_args, logger, load_model, actor_path, critic_path):
        super(PPO, self).__init__(args, env_args=env_args, logger=logger, type="OnPolicy")
        self.device = args.device
        self.completed_interactions = 0

        # training params
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.batch_size = args.batch_size
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr

        # load models and setup optimiser.
        self.policy = ActorCritic(self.args, load_model, actor_path, critic_path).to(self.device)
        if args.verbose:
            print('PolicyNet Params: {}'.format(sum(p.numel() for p in self.policy.Actor.parameters() if p.requires_grad)))
            print('ValueNet Params: {}'.format(sum(p.numel() for p in self.policy.Critic.parameters() if p.requires_grad)))
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.vf_lr)
        self.value_criterion = nn.MSELoss()

        # ppo params
        self.grad_clip = args.grad_clip
        self.entropy_coef = args.entropy_coef
        self.eps_clip = args.eps_clip
        self.target_kl = args.target_kl

    def train_pi(self):
        print('Running Policy Update...')
        temp_loss_log = torch.zeros(1, device=self.device)
        policy_grad, pol_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        # Cost constraint variables
        lambda_c = self.args.lambda_c  # Lagrange multiplier for cost
        cost_threshold = self.args.cost_limit  # Maximum allowable cost
        apply_cost_line_search = self.args.cost_line_search  # Cost-constrained line search flag

        continue_pi_training, buffer_len = True, self.rollout_buffer['len']
        
        for i in range(self.train_pi_iters):
            start_idx, n_batch = 0, 0
            while start_idx < buffer_len:
                n_batch += 1
                end_idx = min(start_idx + self.batch_size, buffer_len)

                states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
                actions_batch = self.rollout_buffer['action'][start_idx:end_idx, :]
                logprobs_batch = self.rollout_buffer['log_prob_action'][start_idx:end_idx, :]
                advantages_batch = self.rollout_buffer['advantage'][start_idx:end_idx]
                advantage_cost_batch = self.rollout_buffer['advantage_cost'][start_idx:end_idx]

                # Normalize the advantage to stabilize training
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)
                advantage_cost_batch = (advantage_cost_batch - advantage_cost_batch.mean()) / (advantage_cost_batch.std() + 1e-5)

                self.optimizer_Actor.zero_grad()

                # Compute log probabilities under new policy
                logprobs_prediction, dist_entropy = self.policy.evaluate_actor(states_batch, actions_batch)

                # Compute policy ratios
                ratios = torch.exp(logprobs_prediction - logprobs_batch).squeeze()

                # Compute PPO clipped policy loss
                r_theta = ratios * advantages_batch
                r_theta_clip = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                policy_loss = -torch.min(r_theta, r_theta_clip).mean()

                # Compute SCPO cost loss
                cost_loss = (ratios * advantage_cost_batch).mean()

                # Compute final SCPO loss with Lagrange multiplier
                total_loss = policy_loss + lambda_c * cost_loss - self.entropy_coef * dist_entropy.mean()

                # Compute KL divergence for Trust Region Constraint
                log_ratio = logprobs_prediction - logprobs_batch
                approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach().cpu().numpy()

                # Enforce trust region constraints
                if approx_kl > 1.5 * self.target_kl:
                    if self.args.verbose:
                        print(f'Early stop => Epoch {i}, Batch {n_batch}, KL: {approx_kl}')
                    continue_pi_training = False
                    break

                # Apply cost-constrained line search (if cost exceeds threshold)
                if apply_cost_line_search and cost_loss.item() > cost_threshold:
                    print(f"Cost {cost_loss.item()} exceeded threshold {cost_threshold}. Adjusting λ_c...")
                    lambda_c *= 1.05  # Increase penalty
                else:
                    lambda_c *= 0.95  # Decrease penalty

                # Prevent NaN values
                if torch.isnan(total_loss):
                    print(f'Policy loss NaN detected! Exiting...')
                    exit()

                # Backpropagation
                temp_loss_log += total_loss.detach()
                total_loss.backward()
                policy_grad += torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
                pol_count += 1
                self.optimizer_Actor.step()
                start_idx += self.batch_size

            if not continue_pi_training:
                break

        mean_pi_grad = policy_grad / pol_count if pol_count != 0 else 0
        print(f'Final Policy Loss: {temp_loss_log}')
        return mean_pi_grad, temp_loss_log

    def train_vf(self):
        print('Running Value Function Update...')

        # Variables for debugging/logging
        val_loss_log, value_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        true_var, explained_var = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)

        lambda_c = self.args.lambda_c  # Lagrange multiplier for cost loss

        for i in range(self.train_v_iters):
            start_idx = 0
            while start_idx < self.rollout_buffer['len']:
                end_idx = min(start_idx + self.batch_size, self.rollout_buffer['len'])

                states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
                value_target = self.rollout_buffer['value_target'][start_idx:end_idx]
                cost_target = self.rollout_buffer['cost_return'][start_idx:end_idx]  # J_Di
                advantage_cost_batch = self.rollout_buffer['advantage_cost'][start_idx:end_idx]

                self.optimizer_Critic.zero_grad()
                value_prediction = self.policy.evaluate_critic(states_batch)

                # Compute Value Loss with Cost-Aware SCPO Adjustment
                value_loss = self.value_criterion(value_prediction, value_target)
                
                # Compute Cost Value Loss (SCPO Constraint)
                cost_loss = self.value_criterion(value_prediction, cost_target)

                # Incorporate cost-aware advantage correction
                advantage_cost_correction = torch.mean((value_prediction - cost_target) * advantage_cost_batch)

                # Final SCPO Value Function Loss
                total_loss = value_loss + lambda_c * cost_loss + lambda_c * advantage_cost_correction

                # Backpropagation
                total_loss.backward()
                value_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.grad_clip)
                self.optimizer_Critic.step()
                val_count += 1
                start_idx += self.batch_size

                # Logging
                val_loss_log += total_loss.detach()
                y_pred = value_prediction.detach().flatten()
                y_true = value_target.flatten()
                var_y = torch.var(y_true)
                true_var += var_y
                explained_var += 1 - torch.var(y_true - y_pred) / (var_y + 1e-5)

        print(f'Final Value Function Loss: {val_loss_log}')
        return value_grad / val_count, val_loss_log, explained_var / val_count, true_var / val_count

    def update(self):
        self.rollout_buffer = self.buffer.get()
        pi_grad, pi_loss = self.train_pi()
        vf_grad, vf_loss, explained_var, true_var = self.train_vf()
        data = dict(policy_grad=pi_grad, policy_loss=pi_loss, value_grad=vf_grad, value_loss=vf_loss,
                    explained_var=explained_var, true_var=true_var)
        return {k: v.detach().cpu().flatten().numpy()[0] for k, v in data.items()}



