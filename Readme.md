This project is an extension of the project https://github.com/RL4H/RL4T1D. 

Using the project
--

<h4>Installation</h4>

* Create and activate a Python3.10.12 virtual environment.
* Clone the repository
* Go to the project folder (RL4T1D): <code>cd RL4T1D</code>
* Install the required Python libraries <code>requirements.txt</code>. 
* Create an environment file <code>.env</code> at the root of the project folder with <code>MAIN_PATH=path-to-this-project</code> (<code>echo "MAIN_PATH=$(pwd)">.env</code>).<br>

<h4>Running SRPO training loop</h4>

```
cd experiments 
python run_RL_agent.py experiment.device=cpu agent=srpo agent.debug=false hydra/job_logging=disabled
```


### Citing
```
@article{hettiarachchi2024g2p2c,
  title={G2P2C—A modular reinforcement learning algorithm for glucose control by glucose prediction and planning in Type 1 Diabetes},
  author={Hettiarachchi, Chirath and Malagutti, Nicolo and Nolan, Christopher J and Suominen, Hanna and Daskalaki, Elena},
  journal={Biomedical Signal Processing and Control},
  volume={90},
  pages={105839},
  year={2024},
  publisher={Elsevier}
}
``` 


