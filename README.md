# Doomination

## Installation
To install Doomination, simply clone or download the repository and run:
```bash
conda create -n doomicrl python=3.10
conda activate doomicrl
pip install --upgrade pip
pip install -e .   # -e flag installs the package in editable mode
```


## Train an agent

To train an agent, run:
```bash
python sample_factory/doom/train_vizdoom.py
```

### Continual Learning
```
# debugging on 1 worker
python sample_factory/doom/train_vizdoom.py --algo PPO --envs volcanic_venture remedy_rush armament_burden detonators_dilemma --seed 123 --num_envs_per_worker 1 --num_workers 1 --worker_num_splits 1 --train_for_env_steps 100000 --with_wandb False --wandb_user tu-e --wandb_project doomination --wandb_tags LOCAL --log_heatmap False

# testing env switching
python sample_factory/doom/train_vizdoom.py --algo PPO --train_for_env_steps 100_000 --env_steps_per_env 10_000 --envs volcanic_venture remedy_rush armament_burden detonators_dilemma --seed 123 --with_wandb True --log_heatmap False

```
