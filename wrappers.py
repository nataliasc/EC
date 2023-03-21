import numpy as np
import torch
from gymnasium.core import ObservationWrapper
from gym import spaces

class ImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = ImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (7, 7, 3)
    """

    def __init__(self, env, device):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]
        height, width, channels = self.observation_space.shape
        new_shape = (channels, height, width)
        self.observation_space = spaces.Box(low=0, high=255, shape=new_shape, dtype=self.observation_space.dtype)
        self.device = device

    def observation(self, obs):
        obs = obs["image"] / 0xFF
        obs = np.swapaxes(obs, 0, -1) # re-order channels from HxWxC to CxHxW for PyTorch convolutional layers
        return torch.from_numpy(obs.astype(np.float32)).to(self.device)