"""Contains a computation-offloading latency cost model definition."""

# EdgeSimPy components
from edge_sim_py.components.cost_models.base_cost_model import (
    resolve_input_data_size,
    transmission_delay,
    network_path_delay,
    processing_delay,
)


class LatencyCostModel:
    """Latency cost of offloading a service to a candidate edge server.

    The total latency follows the classic three-term decomposition used in the
    mobile-edge-computing offloading literature [1]:

        latency = transmission_delay + propagation_delay + processing_delay

    where the transmission delay accounts for pushing the service's input data
    over the access network, the propagation delay accumulates the link delays
    along the user <-> server path (plus wireless access), and the processing
    delay reflects how long the (possibly loaded) server takes to run the task.

    [1] Y. Mao, C. You, J. Zhang, K. Huang, and K. B. Letaief, "A Survey on Mobile
        Edge Computing: The Communication Perspective", IEEE Communications Surveys
        & Tutorials, vol. 19, no. 4, pp. 2322-2358, 2017.
    """

    # Default parameters. Override per call via the ``parameters`` argument.
    DEFAULT_PARAMETERS = {
        "input_data_size": None,  # Mbit to offload (falls back to service.state, then 1.0)
        "access_bandwidth": 100.0,  # Mbps available between the user and its base station
        "cpu_cycles_per_unit": 1.0,  # workload scaling applied to the service CPU demand
    }

    @classmethod
    def get_cost(cls, service: object, server: object, user: object = None, parameters: dict = None) -> float:
        """Computes the latency cost of hosting ``service`` on ``server`` for ``user``.

        Args:
            service (object): Service to be offloaded.
            server (object): Candidate edge server.
            user (object, optional): User accessing the service. Defaults to None.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            cost (float): Estimated end-to-end latency.
        """
        return cls.get_cost_breakdown(service, server, user, parameters)["total"]

    @classmethod
    def get_cost_breakdown(cls, service: object, server: object, user: object = None, parameters: dict = None) -> dict:
        """Computes the latency cost together with its individual terms.

        Args:
            service (object): Service to be offloaded.
            server (object): Candidate edge server.
            user (object, optional): User accessing the service. Defaults to None.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            breakdown (dict): Dictionary with the transmission, propagation, processing
                terms and the resulting total latency.
        """
        params = {**cls.DEFAULT_PARAMETERS, **(parameters or {})}

        data_size = resolve_input_data_size(service=service, parameters=params)

        transmission = transmission_delay(data_size=data_size, bandwidth=params["access_bandwidth"])
        propagation = network_path_delay(user=user, server=server)
        processing = processing_delay(service=service, server=server, parameters=params)

        return {
            "transmission": transmission,
            "propagation": propagation,
            "processing": processing,
            "total": transmission + propagation + processing,
        }
