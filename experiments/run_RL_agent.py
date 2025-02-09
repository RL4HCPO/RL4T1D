import sys
import torch
import random
import os
import warnings
import numpy as np
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
# import wandb

warnings.simplefilter('ignore', Warning)

from utils.logger import setup_folders, copy_folder


def set_agent_parameters(cfg):
    agent = None
    if cfg.agent.agent == 'ppo':
        from agents.algorithm.ppo import PPO
        setup_folders(cfg)
        agent = PPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'cpo':
        from agents.algorithm.cpo import CPO
        setup_folders(cfg)
        agent = CPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'pcpo':
        from agents.algorithm.pcpo import PCPO
        setup_folders(cfg)
        agent = PCPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')
    
    elif cfg.agent.agent == 'combined':
        from agents.algorithm.combined import combined
        setup_folders(cfg)
        agent = combined(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

    # elif args.agent.agent == 'a2c':
    #     from agents.algorithm.a2c import A2C
    #     agent = A2C(args=args, load_model=False, actor_path='', critic_path='')
    #
    # elif args.agent.agent == 'sac':
    #     from agents.algorithm.sac import SAC
    #     agent = SAC(args=args, load_model=False, actor_path='', critic_path='')
    #
    # elif args.agent.agent == 'g2p2c':
    #     from agents.g2p2c.g2p2c import G2P2C
    #     agent = G2P2C(args=args, load_model=False, actor_path='', critic_path='')

    else:
        print('Please select an agent for the experiment. Hint: a2c, sac, ppo, g2p2c')
    return agent


# For checkpoint running
# def set_agent_parameters(cfg, actor_path='',critic_path='', load_model=False):
#     agent = None
#     if cfg.agent.agent == 'ppo':
#         from agents.algorithm.ppo import PPO
#         setup_folders(cfg)
#         agent = PPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

#     elif cfg.agent.agent == 'cpo':
#         from agents.algorithm.cpo import CPO
#         setup_folders(cfg)
#         agent = CPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')


#     elif cfg.agent.agent == 'pcpo':
#         from agents.algorithm.pcpo import PCPO
#         setup_folders(cfg)
#         agent = PCPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

#     # elif args.agent.agent == 'a2c':
#     #     from agents.algorithm.a2c import A2C
#     #     agent = A2C(args=args, load_model=False, actor_path='', critic_path='')
#     #
#     # elif args.agent.agent == 'sac':
#     #     from agents.algorithm.sac import SAC
#     #     agent = SAC(args=args, load_model=False, actor_path='', critic_path='')
#     #
#     # elif args.agent.agent == 'g2p2c':
#     #     from agents.g2p2c.g2p2c import G2P2C
#     #     agent = G2P2C(args=args, load_model=False, actor_path='', critic_path='')

#     else:
#         print('Please select an agent for the experiment. Hint: a2c, sac, ppo, g2p2c')
#     return agent

# def load_latest_checkpoint(experiment_dir):
#     checkpoints_dir = os.path.join(experiment_dir, 'checkpointspcpo')
    
#     checkpoint_files = os.listdir(checkpoints_dir)

#     actor_files = [f for f in checkpoint_files if '_Actor.pth' in f]
#     critic_files = [f for f in checkpoint_files if '_Critic.pth' in f]

#     if not actor_files or not critic_files:
#         raise FileNotFoundError("No Actor or Critic checkpoints found!")
    
#     def get_episode_number(filename):
#         return int(filename.split('_')[1])
    
#     actor_files.sort(key=get_episode_number, reverse=True)
#     critic_files.sort(key=get_episode_number, reverse=True)

#     latest_actor_file = os.path.join(checkpoints_dir, actor_files[0])
#     latest_critic_file = os.path.join(checkpoints_dir, critic_files[0])

#     print(f"Loaded Actor from: {latest_actor_file}")
#     print(f"Loaded Critic from: {latest_critic_file}")

#     return latest_actor_file, latest_critic_file

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    agent = set_agent_parameters(cfg)  # load agent - used for normal running


    #run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    # wandb.config = OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # )
    #wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)



    #pcpo running checkpoints

    # experiment_dir = "../results/test"
    # experiment_dir_new = "../results/pcpo_new"
    # copy_folder(os.path.join(experiment_dir, 'checkpoints'),os.path.join(experiment_dir_new, 'checkpointspcpo'))
    # actor, critic = load_latest_checkpoint(experiment_dir_new)

    # agent = set_agent_parameters(cfg, actor, critic, True)
    if cfg.experiment.verbose:
        print('\nExperiment Starting...')
        print("\nOptions =================>")
        print(vars(cfg))
        print('\nDevice which the program run on:', cfg.experiment.device)

    #exit()

    torch.manual_seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    agent.run()


if __name__ == '__main__':
    main()


#python run_RL_agent.py experiment.folder=test4 agent.debug=True hydra/job_logging=disabled