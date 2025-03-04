import torch
import numpy as np
from utils import core
from utils.reward_normalizer import RewardNormalizer
from utils.core import linear_scaling


class RolloutBuffer:
    def __init__(self, args):
        self.size = args.n_step
        self.device = args.device

        # Discounted vs. Average reward RL
        self.return_type = args.return_type
        self.gamma = args.gamma if args.return_type == 'discount' else 1
        self.lambda_ = args.lambda_ if args.return_type == 'discount' else 1

        self.n_training_workers = args.n_training_workers
        self.n_step = args.n_step
        self.feature_history = args.obs_window
        self.n_features = args.n_features

        self.Rollout = Rollout(args)
        self.shuffle_rollout = args.shuffle_rollout
        self.normalize_reward = args.normalize_reward
        self.reward_normaliser = RewardNormalizer(
            num_envs=self.n_training_workers, cliprew=10.0, gamma=self.gamma, epsilon=1e-8, per_env=False
        )

        # Storage buffers
        self.states = torch.zeros(self.n_training_workers, self.n_step, self.feature_history, self.n_features, device=self.device)
        self.actions = torch.zeros(self.n_training_workers, self.n_step, device=self.device)
        self.actions_logprobs = torch.zeros(self.n_training_workers, self.n_step, device=self.device)
        self.reward = torch.zeros(self.n_training_workers, self.n_step, device=self.device)
        self.v_targ = torch.zeros(self.n_training_workers, self.n_step, device=self.device)
        self.adv = torch.zeros(self.n_training_workers, self.n_step, device=self.device)
        self.v_pred = torch.zeros(self.n_training_workers, self.n_step + 1, device=self.device)

        # Cost-related buffers for SCPO
        self.direct_cost = torch.zeros(self.n_training_workers, self.n_step, device=self.device)  # Di
        self.cost_return = torch.zeros(self.n_training_workers, self.n_step, device=self.device)  # J_Di
        self.adv_cost = torch.zeros(self.n_training_workers, self.n_step, device=self.device)  # A_Di

        self.first_flag = torch.zeros(self.n_training_workers, self.n_step + 1, device=self.device)

        # Only used for G2P2C
        self.agent_id = args.agent
        self.cgm_target = torch.zeros(self.n_training_workers, self.n_step, device=self.device)

    def save_rollout(self, training_agent_index):
        data = self.Rollout.get()
        self.states[training_agent_index] = data['obs']
        self.actions[training_agent_index] = data['act']
        self.actions_logprobs[training_agent_index] = data['logp']
        self.v_pred[training_agent_index] = data['v_pred']
        self.reward[training_agent_index] = data['reward']
        self.first_flag[training_agent_index] = data['first_flag']
        self.cost_return[training_agent_index] = data['cost_return']
        self.direct_cost[training_agent_index] = data['direct_cost']

        # Only used for G2P2C
        self.cgm_target[training_agent_index] = data['cgm_target']

    def compute_gae(self):
        """
        Compute Generalized Advantage Estimation (GAE) for both reward and cost (SCPO).
        Normalizes cost advantage to prevent large fluctuations.
        """
        orig_device = self.v_pred.device
        assert orig_device == self.reward.device == self.first_flag.device

        vpred, reward, first = (x.cpu() for x in (self.v_pred, self.reward, self.first_flag))
        cost_return = self.cost_return.cpu()  # SCPO cost tracking

        first = first.to(dtype=torch.float32)
        assert first.dim() == 2
        nenv, nstep = reward.shape
        assert vpred.shape == first.shape == (nenv, nstep + 1)

        # Reward Advantage
        adv = torch.zeros(nenv, nstep, dtype=torch.float32)
        lastgaelam = 0
        for t in reversed(range(nstep)):
            notlast = 1.0 - first[:, t + 1]
            nextvalue = vpred[:, t + 1]
            # notlast: whether next timestep is from the same episode
            delta = reward[:, t] + notlast * self.gamma * nextvalue - vpred[:, t]
            adv[:, t] = lastgaelam = delta + notlast * self.gamma * self.lambda_ * lastgaelam
        
        vtarg = vpred[:, :-1] + adv

        # Cost Advantage (Always follows SCPO logic)
        adv_cost = torch.zeros(nenv, nstep, dtype=torch.float32)
        lastgaelam_cost = 0
        for t in reversed(range(nstep)):
            notlast = 1.0 - first[:, t + 1]
            nextvalue_cost = cost_return[:, t]  # Using cumulative cost return (J_Di)

            # Compute cost delta
            delta_cost = cost_return[:, t] + notlast * nextvalue_cost - cost_return[:, t]
            adv_cost[:, t] = lastgaelam_cost = delta_cost + notlast * lastgaelam_cost

        adv_cost = adv_cost / (adv_cost.max() + 1e-5)  # Normalize in [0, 1]

        return adv.to(device=orig_device), vtarg.to(device=orig_device), adv_cost.to(device=orig_device)

    def prepare_rollout_buffer(self, AuxiliaryBuffer=None):
        """
        Prepare the buffer for training with normalized advantages.
        """
        if self.return_type == 'discount':
            if self.normalize_reward:  
                self.reward = self.reward_normaliser(self.reward, self.first_flag)
            self.adv, self.v_targ, self.adv_cost = self.compute_gae()

        if self.return_type == 'average':
            self.reward = self.reward_normaliser(self.reward, self.first_flag, type='average')
            self.adv, self.v_targ, self.adv_cost = self.compute_gae()

        '''Concat data from different workers'''
        s_hist = self.states.view(-1, self.feature_history, self.n_features)
        act = self.actions.view(-1, 1)
        logp = self.actions_logprobs.view(-1, 1)
        v_targ = self.v_targ.view(-1)
        adv = self.adv.view(-1)
        dir_cost = self.direct_cost.view(-1)
        cost_ret = self.cost_return.view(-1)
        adv_cost = self.adv_cost.view(-1)
        first_flag = self.first_flag.view(-1)

        buffer_len = s_hist.shape[0]

        cgm_target = self.cgm_target.view(-1)
        if self.agent_id == "g2p2c":
            AuxiliaryBuffer.update(s_hist, cgm_target, act, first_flag, adv_cost)

        if self.shuffle_rollout:
            rand_perm = torch.randperm(buffer_len)
            s_hist = s_hist[rand_perm, :, :]  # torch.Size([batch, n_steps, features])
            act = act[rand_perm, :]  # torch.Size([batch, 1])
            logp = logp[rand_perm, :]  # torch.Size([batch, 1])
            v_targ = v_targ[rand_perm]  # torch.Size([batch])
            adv = adv[rand_perm]  # torch.Size([batch])
            dir_cost = dir_cost[rand_perm]  # torch.Size([batch])
            cost_ret = cost_ret[rand_perm]  # torch.Size([batch])
            adv_cost = adv_cost[rand_perm]  # torch.Size([batch])

        return dict(
            states=s_hist, 
            action=act, 
            log_prob_action=logp, 
            value_target=v_targ,
            advantage=adv, 
            len=buffer_len,
            cgm_target=cgm_target,
            direct_cost=dir_cost,
            cost_return=cost_ret,
            advantage_cost=adv_cost, 
        )

    def get(self, AuxiliaryBuffer=None):
        return self.prepare_rollout_buffer(AuxiliaryBuffer)

class Rollout:
    def __init__(self, args):
        self.size = args.n_step
        self.device = args.device
        self.args = args

        self.feature_hist = args.obs_window
        self.features = args.n_features

        self.state = np.zeros(core.combined_shape(self.size, (self.feature_hist, self.features)), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.state_values = np.zeros(self.size + 1, dtype=np.float32)
        self.logprobs = np.zeros(self.size, dtype=np.float32)
        self.first_flag = np.zeros(self.size + 1, dtype=np.bool_)
        self.cgm_target = np.zeros(self.size, dtype=np.float32)

        # Cost-related tracking
        self.direct_cost = np.zeros(self.size, dtype=np.float32)  # Stores cost increments (D_i)
        self.cost_return = np.zeros(self.size, dtype=np.float32)  # Stores cumulative cost return (J_Di)
        self.max_cost = 0  # Initialize max state-wise cost (M)

        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size

    def calc_direct_cost(self, cgm_target, is_first):
        """
        Compute cost increment (D_i) based on CGM target.
        Normalize and clip extreme values for stable cost training.
        """
        # Compute raw cost based on CGM deviation from the norm (70-180 mg/dL)
        if 70 <= cgm_target <= 180:
            cost = 0
        elif 54 <= cgm_target < 70:
            cost = (70 - cgm_target) / 8
        elif cgm_target < 54:
            cost = (70 - cgm_target) / 4
        elif 180 < cgm_target <= 250:
            cost = (cgm_target - 180) / 140
        else:
            cost = (cgm_target - 180) / 70

        # **Safe Normalization**: Check if std() is zero
        cost_std = self.direct_cost.std()
        cost_mean = self.direct_cost.mean()

        if cost_std > 1e-5:  # Avoid division by zero
            cost = (cost - cost_mean) / cost_std
        else:
            cost = cost - cost_mean  # Only center without scaling

        # Apply clipping to prevent large cost spikes
        cost = max(0.0, min(cost, 5.0)) # Ensuring cost values stay reasonable

        # Compute direct cost increment (SCPO state-wise constraint)
        if is_first:
            direct_cost = cost
            self.max_cost = cost  # Reset max cost tracking (M)
        else:
            direct_cost = max(cost - self.max_cost, 0)
            self.max_cost = max(self.max_cost, cost)  # Update max state-wise cost (M)

        return direct_cost

    def store(self, obs, act, rew, val, logp, cgm_target, is_first):
        """
        Store transition step details including SCPO cost tracking.
        """
        assert self.ptr < self.max_size

        self.state[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.state_values[self.ptr] = val
        self.logprobs[self.ptr] = logp
        self.first_flag[self.ptr] = is_first
        self.cgm_target[self.ptr] = linear_scaling(x=cgm_target, x_min=self.args.glucose_min, x_max=self.args.glucose_max)

        # Store cost increment (D_i)
        self.direct_cost[self.ptr] = self.calc_direct_cost(cgm_target, is_first)

        # Compute and store cumulative cost return (J_Di)
        self.cost_return[self.ptr] = self.direct_cost[self.ptr] if (self.ptr == 0 or is_first) else self.cost_return[self.ptr - 1] + self.direct_cost[self.ptr]

        self.ptr += 1

    def finish_path(self, final_v):
        self.state_values[self.ptr] = final_v
        self.first_flag[self.ptr] = False

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        data = dict(
            obs=self.state, 
            act=self.actions, 
            v_pred=self.state_values,
            logp=self.logprobs, 
            first_flag=self.first_flag, 
            reward=self.rewards, 
            cgm_target=self.cgm_target,
            direct_cost=self.direct_cost,  # Cost increment (D_i)
            cost_return=self.cost_return,  # Cumulative cost return (J_Di)
        )

        # Reset max cost tracking for the next trajectory
        self.max_cost = 0

        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}

