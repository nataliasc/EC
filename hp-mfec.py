""" Optimize the hyperparameters of a MFEC agent
training on a Minigrid environment.
"""
from typing import Any
from typing import Dict
import os

import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
# from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor # record episode statistics
import torch
import torch.nn as nn
import numpy as np

import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from wrappers import ImgObsWrapper

from agents import MFECAgent, NECAgent
from memory import ExperienceReplay
from test import test
from argparse import Namespace


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(3e6)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

ENV_ID = 'MiniGrid-DoorKey-6x6-v0'

DEFAULT_HYPERPARAMS = {
    "env": ENV_ID,
}


def sample_nec_params(trial: optuna.Trial):
    """Sampler for NEC hyperparameters."""
    key_size = trial.suggest_int('key_size', 64, 512)
    num_neighbours = trial.suggest_int('num_neighbours', 11, 50)
    memory_capacity = trial.suggest_int('memory_capacity', 100, 10_000)
    dictionary_capacity = trial.suggest_int('dictionary_capacity', 500, 50_000, log=True)
    replay_frequency = trial.suggest_int('replay_frequency', 1, 4)
    epsilon_initial = trial.suggest_float("epsilon_initial", 0.99, 1, log=False)
    epsilon_final = trial.suggest_float("epsilon_final", 0.01, 0.05, log=False)
    epsilon_anneal_start = trial.suggest_int('epsilon_anneal_start', 5000, 10_000)
    epsilon_anneal_end = trial.suggest_int('epsilon_anneal_end', 1_000_000, 5_000_000)
    discount = trial.suggest_float("discount", 0.99, 1., log=False)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    dictionary_learning_rate = trial.suggest_float("dict_lr", 1e-5, 0.1, log=True)
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values.
    # trial.set_user_attr("n_steps", n_steps)
    trial.set_user_attr("key_size", 64)

    # net_arch = [
    #     {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    # ]

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    args = Namespace(id='minigrid-hp',
                 seed=123,
                 T_max=int(10e4),
                 max_episode_length=605,
                 key_size=key_size,
                 num_neighbours=num_neighbours,
                #  model="./model.pth",
                 memory_capacity=memory_capacity,
                 dictionary_capacity=dictionary_capacity,
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
                 dictionary_learning_rate=dictionary_learning_rate,
                 kernel='mean',
                 kernel_delta=1e-3,
                 batch_size=32,
                 learn_start=10_000,
                 evaluation_interval=1000,
                 evaluation_episodes=10,
                 evaluation_size=300,
                 evaluation_epsilon=0.,
                 checkpoint_interval=0,
                 render=False)
    return args

    # return {
    #     "n_steps": n_steps,
    #     "key_size": key_size,
    #     "num_neighbours": num_neighbours,
    #     "memory_capacity": memory_capacity,
    #     "dictionary_capacity": dictionary_capacity,
    #     "replay_frequency": replay_frequency,
    #     "epsilon_initial": epsilon_initial,
    #     "epsilon_final": epsilon_final,
    #     "epsilon_anneal_start": epsilon_anneal_start,
    #     "epsilon_anneal_end": epsilon_anneal_end,
    #     "discount": discount,
    #     "learning_rate": learning_rate,
    #     "evaluation_interval": 1000,
    #     "episodic_multistep": 1e6,
    #     "max_episode_length": 605,
    #     "rmsprop_decay": 0.95,
    #     "rmsprop_epsilon": 0.01,
    #     "rmsprop_momentum": 0,
    #     "dictionary_learning_rate": 0.01,
    #     "kernel": "mean",
    #     "kernel_delta": 1e-3,
    #     "batch_size": 32,
    #     "evaluation_episodes": 10,
    #     "evaluation_size": 100,
    #     "evaluation_epsilon": 0.01,
    #     "checkpoint_interval": 0
    # }


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
    # kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs = sample_nec_params(trial)
    results_dir = os.path.join('results', kwargs.id)
    os.makedirs(results_dir, exist_ok=True)
    kwargs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create env used for training the agent
    env = ImgObsWrapper(RGBImgObsWrapper(gym.make(ENV_ID)), kwargs.device)
    hash_size = len(np.concatenate([env.grid.encode(),
                env.agent_pos, env.agent_dir], axis=None).ravel()) + 1
    # Create the RL model.
    agent = MFECAgent(kwargs, env.observation_space.shape, env.action_space.n, hash_size)
    # mem = ExperienceReplay(kwargs.memory_capacity, env.observation_space.shape, device)
    # Create env used for evaluation.
    # eval_env = Monitor(ImgObsWrapper(RGBImgObsWrapper(gym.make(ENV_ID))))
    # Create the callback that will periodically evaluate and report the performance.
    # eval_callback = TrialEvalCallback(
    #     eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    # )

    nan_encountered = False
    try:
        agent.train()
        # agent.learn(N_TIMESTEPS, callback=eval_callback)
        done, epsilon = True, kwargs.epsilon_initial
        agent.set_epsilon(epsilon)
        for T in range(1, int(3e6) + 1):
           if done:
            state, info = env.reset()
            done = False
            states, actions, rewards, keys, values, hashes = [], [], [], [], [], []  # Store transition data in episodic buffers

            # Linearly anneal ε over set interval
            if T > kwargs.epsilon_anneal_start and T <= kwargs.epsilon_anneal_end:
                epsilon -= (kwargs.epsilon_initial - kwargs.epsilon_final) / (kwargs.epsilon_anneal_end - kwargs.epsilon_anneal_start)
                agent.set_epsilon(epsilon)
        
            # Append transition data to episodic buffers (1/2)
            states.append(state.detach().cpu())
            state_hash = np.concatenate([env.grid.encode(),
                        env.agent_pos, env.agent_dir], axis=None).ravel()
            state_hash = state_hash / 0xFF
            state_hash = state_hash.astype(np.float32)
            hashes.append(state_hash)
        
            # Choose an action according to the policy
            action, key, value = agent.act(state, return_key_value=True)
            state, reward, terminated, truncated, info = env.step(action)  # Step
            done = terminated | truncated


            # Append transition data to episodic buffers (2/2); note that original NEC implementation does not recalculate keys/values at the end of the episode
            actions.append(action)
            rewards.append(reward)
            keys.append(key.cpu().numpy())
            values.append(value)

            # Calculate returns at episode to batch memory updates
            if done:
                episode_T = len(rewards)
                returns, multistep_returns = [None] * episode_T, [None] * episode_T
                returns.append(0)
                for i in range(episode_T - 1, -1, -1):  # Calculate return-to-go in reverse
                    returns[i] = rewards[i] + kwargs.discount * returns[i + 1]
                    if episode_T - i > kwargs.episodic_multi_step:  # Calculate multi-step returns (originally only for NEC)
                        multistep_returns[i] = returns[i] + kwargs.discount ** kwargs.episodic_multi_step * (values[i + kwargs.episodic_multi_step] - returns[i + kwargs.episodic_multi_step])
                    else:  # Calculate Monte Carlo returns (originally only for MFEC)
                        multistep_returns[i] = returns[i]
                states, actions, returns, keys, hashes = np.stack(states), np.asarray(actions, dtype=np.int64), np.asarray(multistep_returns, dtype=np.float32), np.stack(keys), np.stack(hashes)
                unique_actions, unique_action_reverse_idxs = np.unique(actions, return_inverse=True)  # Find every unique action taken and indices
                for i, a in enumerate(unique_actions):
                    a_idxs = (unique_action_reverse_idxs == i).nonzero()[0]
                    agent.update_memory_batch(a.item(), keys[a_idxs], returns[a_idxs][:, np.newaxis], hashes[a_idxs])  # Append transition to DND of action in batch

            # results_dir = os.path.join('results', kwargs.id)
            test_rewards = []
            if T % kwargs.evaluation_interval == 0:
                agent.eval()  # Set agent to evaluation mode
                test_rewards = test(kwargs, T, agent, results_dir)  # Test
                agent.train()  # Set agent back to training mode
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        env.close()
        # eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # return eval_callback.last_mean_reward
    return sum(test_rewards)


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    # torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(storage="sqlite:///mfec.sqlite3",
                                study_name="MFEC-18",
                                sampler=sampler,
                                pruner=pruner,
                                direction="maximize")
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