""" Optimize the hyperparameters of a DQN agent
training on a Minigrid environment.
"""
from typing import Any
from typing import Dict
import os
import json

import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
# from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor # record episode statistics
import torch
import torch.nn as nn
import numpy as np
import time

import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from wrappers import ImgObsWrapper

from agents import MFECAgent, NECAgent
from memory import ExperienceReplay
from test import test
from argparse import Namespace


N_TRIALS = 10
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(1e6)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

ENV_ID = 'MiniGrid-DoorKey-6x6-v0'


def sample_dqn_params(trial: optuna.Trial):
    """Sampler for DQN hyperparameters."""
    key_size = 2 ** trial.suggest_int('exponent_key_size', 6, 9)
    memory_capacity = 10 ** trial.suggest_int('exponent_memory_capacity', 3, 5) # replay buffer size
    replay_frequency = trial.suggest_int('replay_frequency', 1, 4)
    epsilon_initial = trial.suggest_float("epsilon_initial", 0.99, 1, log=False)
    epsilon_final = trial.suggest_float("epsilon_final", 0.01, 0.05, log=False)
    epsilon_anneal_start = trial.suggest_int('epsilon_anneal_start', 5000, 10_000)
    epsilon_anneal_end = trial.suggest_int('epsilon_anneal_end', 1_000_000, 5_000_000)
    discount = trial.suggest_float("discount", 0.99, 1., log=False)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)

    args = Namespace(id='minigrid-hp',
                 seed=42,
                 T_max=int(10e6),
                 max_episode_length=605,
                 key_size=key_size,
                 model=None,
                 memory_capacity=memory_capacity,
                 replay_frequency=replay_frequency,
                 episodic_multi_step=100,
                 epsilon_initial=epsilon_initial,
                 epsilon_final=epsilon_final,
                 epsilon_anneal_start=epsilon_anneal_start,
                 epsilon_anneal_end=epsilon_anneal_end,
                 discount=discount,
                 learning_rate=learning_rate,
                 rmsprop_decay=0.95,
                 rmsprop_epsilon=0.01,
                 rmsprop_momentum=0,
                 kernel='mean',
                 kernel_delta=1e-3,
                 batch_size=32,
                 learn_start=100,
                 evaluation_interval=1,
                 evaluation_size=100,
                 evaluation_episodes=10,
                 evaluation_epsilon=0.,
                 checkpoint_interval=0,
                 render=False)
    return args

# class TrialEvalCallback(EvalCallback):
#     """Callback used for evaluating and reporting a trial."""

#     def __init__(
#         self,
#         eval_env: gym.Env,
#         trial: optuna.Trial,
#         n_eval_episodes: int = 5,
#         eval_freq: int = 10000,
#         deterministic: bool = True,
#         verbose: int = 0,
#     ):
#         super().__init__(
#             eval_env=eval_env,
#             n_eval_episodes=n_eval_episodes,
#             eval_freq=eval_freq,
#             deterministic=deterministic,
#             verbose=verbose,
#         )
#         self.trial = trial
#         self.eval_idx = 0
#         self.is_pruned = False

#     def _on_step(self) -> bool:
#         if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#             super()._on_step()
#             self.eval_idx += 1
#             self.trial.report(self.last_mean_reward, self.eval_idx)
#             # Prune trial if need.
#             if self.trial.should_prune():
#                 self.is_pruned = True
#                 return False
#         return True


def objective(trial: optuna.Trial) -> float:
    # Sample hyperparameters.
    config = sample_dqn_params(trial)
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(f'results_{timestr}', config.id)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create env used for training the agent
    env = ImgObsWrapper(RGBImgObsWrapper(gym.make(ENV_ID)), config.device)
    hash_size = len(np.concatenate([env.grid.encode(),
                env.agent_pos, env.agent_dir], axis=None).ravel()) + 1
    # Create the RL model.
    agent = NECAgent(config, env.observation_space.shape, env.action_space.n,
                     hash_size, use_memory=False)
    mem = ExperienceReplay(config.memory_capacity, env.observation_space.shape, config.device)

    # Construct validation memory
    val_mem = ExperienceReplay(config.evaluation_size, env.observation_space.shape, config.device)
    T, done, states = 0, True, []  # Store transition data in episodic buffers
    while T < config.evaluation_size:
        if done:
            state, info = env.reset()
            done = False
        states.append(state.detach().cpu())  # Append transition data to episodic buffers
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated | truncated
        T += 1
    val_mem.append_batch(np.stack(states), np.zeros((config.evaluation_size, ), dtype=np.int64), np.zeros((config.evaluation_size, ), dtype=np.float32))

    agent.train()
    # agent.learn(N_TIMESTEPS, callback=eval_callback)
    done, epsilon, test_rewards = True, config.epsilon_initial, 0
    agent.set_epsilon(epsilon)
    for T in range(int(500_000)):
        if done:
            state, info = env.reset()
            done = False
            states, actions, rewards, keys, values, hashes = [], [], [], [], [], []  # Store transition data in episodic buffers

        # Linearly anneal ε over set interval
        if T > config.epsilon_anneal_start and T <= config.epsilon_anneal_end:
            epsilon -= (config.epsilon_initial - config.epsilon_final) / (config.epsilon_anneal_end - config.epsilon_anneal_start)
            agent.set_epsilon(epsilon)
    
        # Append transition data to episodic buffers (1/2)
        states.append(state.detach().cpu())
        state_hash = np.concatenate([env.grid.encode(),
                    env.agent_pos, env.agent_dir], axis=None).ravel()
        state_hash = state_hash / 0xFF
        state_hash = state_hash.astype(np.float32)
        hashes.append(state_hash)
    
        # Choose an action according to the policy
        action, value = agent.act(state, return_value=True, return_key_value=False)
        state, reward, terminated, truncated, info = env.step(action)  # Step
        done = terminated | truncated


        # Append transition data to episodic buffers (2/2)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        # Calculate returns at episode to batch memory updates
        if done:
            episode_T = len(rewards)
            returns, multistep_returns = [None] * episode_T, [None] * episode_T
            returns.append(0)
            for i in range(episode_T - 1, -1, -1):  # Calculate return-to-go in reverse
                returns[i] = rewards[i] + config.discount * returns[i + 1]
                if episode_T - i > config.episodic_multi_step:  # Calculate multi-step returns (originally only for NEC)
                    multistep_returns[i] = returns[i] + config.discount ** config.episodic_multi_step * (values[i + config.episodic_multi_step] - returns[i + config.episodic_multi_step])
                else:  # Calculate Monte Carlo returns (originally only for MFEC)
                    multistep_returns[i] = returns[i]
            states, actions, returns, keys, hashes = np.stack(states), np.asarray(actions, dtype=np.int64), np.asarray(multistep_returns, dtype=np.float32), np.stack(keys), np.stack(hashes)
            mem.append_batch(states, actions, returns)
        
        if T >= config.learn_start:
            if T % config.replay_frequency == 0:
                agent.learn(mem)  # Train network

        if T % config.evaluation_interval == 0:
            agent.eval()  # Set agent to evaluation mode
            test_rewards, test_Qs = test(config, T, agent, val_mem, results_dir)  # Test
            agent.train()  # Set agent back to training mode
            
    env.close()


    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # return eval_callback.last_mean_reward
    print(type(test_rewards), "\n", test_rewards)
    return np.mean(test_rewards)


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    # torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(storage="sqlite:///mfec.sqlite3",
                                study_name="DQN DoorKey 6x6",
                                sampler=sampler,
                                pruner=pruner,
                                direction="maximize",
                                load_if_exists=True,)
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))