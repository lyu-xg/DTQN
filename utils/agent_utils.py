import gym
import torch
import numpy as np
from dtqn.agents.dqn import DqnAgent
from dtqn.agents.adrqn import AdrqnAgent
from dtqn.agents.drqn import DrqnAgent
from dtqn.agents.dtqn import DtqnAgent
from dtqn.networks.adrqn import ADRQN
from dtqn.networks.drqn import DRQN
from dtqn.networks.darqn import DARQN
from dtqn.networks.dqn import DQN
from dtqn.networks.dtqn import DTQN
from utils import env_processing

from utils.team import Team

MODEL_MAP = {
    "DTQN": DTQN,
    "ADRQN": ADRQN,
    "DRQN": DRQN,
    "DARQN": DARQN,
    "DQN": DQN,
}

AGENT_MAP = {
    "DTQN": DtqnAgent,
    "ADRQN": AdrqnAgent,
    "DRQN": DrqnAgent,
    "DARQN": DrqnAgent,
    "DQN": DqnAgent,
}


def get_agent(
        model_str: str,
        env: gym.Env,
        embed_per_obs_dim: int,
        inner_embed: int,
        buffer_size: int,
        device: torch.device,
        learning_rate: float,
        batch_size: int,
        context_len: int,
        history: bool,
        target_update_frequency: int,
        gamma: float,
        num_heads: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        identity: bool = True,
        gate: str = "res",
        pos: int = 1,
):
    is_multi_agent = hasattr(env, 'n_agents') and env.n_agents > 1
    env_obs_length = env_processing.get_env_obs_length(env.observation_space)
    env_obs_mask = env_processing.get_env_obs_mask(env.observation_space)
    if isinstance(env_obs_mask, np.ndarray):
        obs_vocab_size = env_obs_mask.max() + 1
    else:
        obs_vocab_size = env_obs_mask + 1
    is_discrete_env = isinstance(
        env.observation_space,
        (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary),
    )
    action_space_length = env.action_space.n

    if model_str == 'DQN': context_len = 1

    def make_model(network_cls):
        return lambda: network_cls(
            env_obs_length,
            action_space_length,
            embed_per_obs_dim,
            inner_embed,
            is_discrete_env,
            obs_vocab_size=obs_vocab_size,
            batch_size=batch_size,
        ).to(device)

    def make_dtqn(network_cls):
        return lambda: network_cls(
            env_obs_length,
            action_space_length,
            embed_per_obs_dim,
            inner_embed,
            num_heads,
            num_layers,
            context_len,
            dropout=dropout,
            gate=gate,
            identity=identity,
            pos=pos,
            discrete=is_discrete_env,
            vocab_sizes=obs_vocab_size,
            target_update_frequency=target_update_frequency,
        ).to(device)

    if "DTQN" not in model_str:
        network_factory = make_model(MODEL_MAP[model_str])
    else:
        network_factory = make_dtqn(MODEL_MAP[model_str])

    agent_factory = lambda i: AGENT_MAP[model_str](
        network_factory,
        buffer_size,
        device,
        env_obs_length,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        context_len=context_len,
        embed_size=inner_embed,
        history=history,
        target_update_frequency=target_update_frequency,
        obs_mask=env_obs_mask,
        num_actions=action_space_length,
        agent_index=i+1
    )

    if is_multi_agent:
        return Team([agent_factory(i) for i in range(env.n_agents)])
    else:
        return agent_factory(1)
