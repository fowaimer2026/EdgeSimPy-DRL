"""Deep Reinforcement Learning for computation offloading in EdgeSimPy.

This is an *opt-in* subpackage with a hard dependency on PyTorch, kept separate from the
core framework so that importing ``edge_sim_py`` never requires torch. Import it
explicitly::

    from edge_sim_py.drl import OffloadingEnv, DQNAgent, PPOAgent, A2CAgent

It provides a self-contained service-placement RL environment whose reward is the
negative weighted offloading cost (reusing ``WeightedCostModel``), three agents
(DQN with Double/Dueling, PPO, A2C/A3C), synthetic scenario generation, evaluation
baselines, and a policy adapter that runs a trained agent inside a live simulation.
"""

__version__ = "1.1.0"

from .offloading_env import OffloadingEnv
from .scenario import random_scenario, reset_component_registries
from .evaluation import evaluate_agent, evaluate_greedy, evaluate_random
from .agents import DQNAgent, PPOAgent, A2CAgent
from .policy import drl_offloading_policy
