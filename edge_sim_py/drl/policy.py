"""Adapter that runs a trained DRL agent as an EdgeSimPy resource-management policy.

This lets a policy learned in :class:`OffloadingEnv` drive a *live* simulation: at each
step it encodes every pending service against the current edge-server state, asks the
agent for a (greedy) placement, and provisions the service through the regular
EdgeSimPy :meth:`Service.provision` API -- so capacity reservation, layer pulling and
migration all behave exactly as they would for the hand-written policies.
"""

# EdgeSimPy components
from edge_sim_py.components.service import Service
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.drl.offloading_env import OffloadingEnv


def drl_offloading_policy(parameters: dict):
    """Resource-management algorithm backed by a trained DRL agent.

    Required ``parameters`` keys:
        * ``agent``: a trained agent exposing ``act(state, mask, greedy=True)``. Its action
          space size must match ``len(EdgeServer.all())``.

    Optional keys:
        * ``cost_model_parameters`` (dict): forwarded to the environment's cost models so
          the state features match those seen during training.

    Args:
        parameters (dict): Resource-management algorithm parameters.
    """
    agent = parameters.get("agent")
    if agent is None:
        raise ValueError("drl_offloading_policy requires a trained 'agent' in its parameters.")

    cost_parameters = parameters.get("cost_model_parameters", {})
    servers = EdgeServer.all()
    pending = [service for service in Service.all() if service.server is None and not service.being_provisioned]
    if not pending or not servers:
        return

    # The environment is used purely as a (read-only) state encoder over the live servers;
    # placement itself goes through the real provisioning API below.
    env = OffloadingEnv(services=pending, servers=servers, cost_parameters=cost_parameters)

    for service in pending:
        if service.server is not None or service.being_provisioned:
            continue

        state, mask = env.observe(service)
        action = agent.act(state, mask, greedy=True)
        target_server = servers[action]

        if env._has_capacity(target_server, service):
            service.provision(target_server=target_server)
