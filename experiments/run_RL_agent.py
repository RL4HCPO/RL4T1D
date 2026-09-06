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

    elif cfg.agent.agent == 'srpo':
        from agents.algorithm.srpo import SRPO
        setup_folders(cfg)
        agent = SRPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'combined':
        from agents.algorithm.combined import combined
        setup_folders(cfg)
        agent = combined(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

    else:
        print('Please select an agent for the experiment. Hint: a2c, sac, ppo, g2p2c')
    return agent


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    agent = set_agent_parameters(cfg)  # load agent - used for normal running

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