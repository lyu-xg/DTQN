import wandb

from dtqn.agents.dqn import DqnAgent
from utils.logging_utils import RunningAverage


class Team:
    # meta agent, act between env and individual agents, aggregate and distribute interaction information
    def __init__(self, agents: [DqnAgent]):
        self.agents = agents
        self.episode_rewards = RunningAverage(10)
        self.episode_lengths = RunningAverage(10)

    def get_action(self, obs) -> [int]:
        return [a.get_action(o) for a, o in zip(self.agents, obs)]

    def observe(self, cur_obs, obs, action, reward, done, timestep, store=True):
        # distribute cur_obs, obs, action to the agents, singularity reward since we have dec-pomdp
        for cur_o, o, u, a in zip(cur_obs, obs, action, self.agents):
            a.observe(cur_o, o, u, reward, done, timestep, store=store)

    def train(self) -> None:
        for a in self.agents: a.train()

    def context_reset(self):
        for a in self.agents: a.context_reset()

    def eval_on(self):
        for a in self.agents: a.eval_on()

    def eval_off(self):
        for a in self.agents: a.eval_off()

    def log(self, ep_reward, ep_len):
        self.episode_rewards.add(ep_reward)
        self.episode_lengths.add(ep_len)

    def sample_random_action(self):
        return [a.sample_random_action() for a in self.agents]

    def report(self, t):
        for a in self.agents: a.report(t)
        wandb.log(
            {
                f"results/Eval_Return": self.episode_rewards.mean(),
                f"results/Eval_Episode_Length": self.episode_lengths.mean(),
            },
            t,
        )