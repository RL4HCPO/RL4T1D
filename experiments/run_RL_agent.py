import sys
import torch
import random
import warnings
import numpy as np
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf


warnings.simplefilter('ignore', Warning)


from utils.logger import setup_folders


def set_agent_parameters(cfg):
    agent = None
    setup_folders(cfg)
    if cfg.agent.agent == 'ppo':
        from agents.algorithm.ppo import PPO
        agent = PPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'a2c':
        from agents.algorithm.a2c import A2C
        agent = A2C(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'sac':
        from agents.algorithm.sac import SAC
        agent = SAC(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')
    #
    # elif args.agent.agent == 'g2p2c':
    #     from agents.g2p2c.g2p2c import G2P2C
    #     agent = G2P2C(args=args, load_model=False, actor_path='', critic_path='')

    else:
        print('Please select an agent for the experiment. Hint: a2c, sac, ppo, g2p2c')
    return agent


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.experiment.verbose:
        print('\nExperiment Starting...')
        print("\nOptions =================>")
        print(vars(cfg))
        print('\nDevice which the program run on:', cfg.experiment.device)

    agent = set_agent_parameters(cfg)  # load agent

    torch.manual_seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    if cfg.mlflow.track:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.experiment.name)
        experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
        run_name = cfg.experiment.run_name if cfg.experiment.run_name is not None else cfg.agent.agent

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
            mlflow.log_params(cfg)
            agent.run()
    else:
        agent.run()


if __name__ == '__main__':
    main()

