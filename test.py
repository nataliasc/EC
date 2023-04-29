import os
import numpy as np
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import ipdb
import plotly.express as px
import time
import imageio

import gymnasium as gym
from wrappers import ImgObsWrapper
from minigrid.wrappers import RGBImgObsWrapper


# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10


# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='episode'):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
    ys = np.asarray(ys_population, dtype=np.float32)
    ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
    trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
    data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
  else:
    data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
  plotly.offline.plot({
    'data': data,
    'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
    
# Test DQN
def test(args, T, agent, val_mem, results_dir, evaluate=False):
  global Ts, rewards, Qs, best_avg_reward
  # Environment
  env = gym.make('MiniGrid-DoorKey-6x6-v0')
  env = RGBImgObsWrapper(env) # Get pixel observations
  env = ImgObsWrapper(env, args.device) # Get rid of the 'mission' field
  Ts.append(T)
  T_rewards, T_Qs, frames = [], [], []

  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, info = env.reset()
        reward_sum = 0
        done = False

      action = agent.act(state)  # Choose an action ε-greedily (default for eval mode)

      state, reward, terminated, truncated, _ = env.step(action)  # Step
        
      done = terminated | truncated
      reward_sum += reward
      frame = env.get_frame(tile_size=30)
      frames.append(frame)

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()
  timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
  imageio.mimsave(f'test_{timestr}.gif', frames, fps=4)

  # Test Q-values over validation memory
  for state in val_mem:  # Iterate over valid states
    T_Qs.append(agent.evaluate_q(state))

  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  if not evaluate:
    # Append to results
    rewards.append(T_rewards)
    Qs.append(T_Qs)

    # Plot
    lineplot(Ts, rewards, 'Reward', path=results_dir, xaxis='Step')
    lineplot(Ts, Qs, 'Q', path=results_dir, xaxis='Step')

    # Save model parameters if improved
    if avg_reward > best_avg_reward:
      best_avg_reward = avg_reward
      agent.save(results_dir)

  # Return rewards and Q-values
  return T_rewards, T_Qs
