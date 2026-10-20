"""Contains a computation-offloading monetary cost model definition."""

# EdgeSimPy components
from edge_sim_py.components.cost_models.base_cost_model import (
    resolve_input_data_size,
    processing_delay,
)


class MonetaryCostModel:
    """Monetary (billing) cost of offloading a service to a candidate edge server.

    Following common pay-as-you-go edge/cloud pricing schemes, the charge combines
    a compute term, a memory term and a data-transfer term:

        * compute cost  -- price per CPU-unit per time x CPU demand x processing time;
        * memory cost   -- price per memory-unit per time x memory demand x processing time;
        * transfer cost -- price per GB x offloaded data converted from Mbit to GB.

    Per-server pricing can be expressed by setting ``server.cost_model_parameters``,
    which is merged on top of the defaults so heterogeneous infrastructures (e.g.
    cheaper community servers vs. premium operator servers) can be modeled.
    """

    # Default prices (arbitrary currency units). Override per call/per server.
    DEFAULT_PARAMETERS = {
        "input_data_size": None,  # Mbit to offload (falls back to service.state, then 1.0)
        "cpu_cycles_per_unit": 1.0,  # workload scaling applied to the service CPU demand
        "price_per_cpu_unit_time": 0.05,  # charge per unit of CPU demand per unit of time
        "price_per_memory_unit_time": 0.01,  # charge per unit of memory demand per unit of time
        "price_per_gb_transferred": 0.10,  # charge per GB of offloaded data
    }

    @classmethod
    def get_cost(cls, service: object, server: object, user: object = None, parameters: dict = None) -> float:
        """Computes the monetary cost of hosting ``service`` on ``server``.

        Args:
            service (object): Service to be offloaded.
            server (object): Candidate edge server.
            user (object, optional): Unused; kept for a uniform cost-model interface.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            cost (float): Estimated monetary charge.
        """
        return cls.get_cost_breakdown(service, server, user, parameters)["total"]

    @classmethod
    def get_cost_breakdown(cls, service: object, server: object, user: object = None, parameters: dict = None) -> dict:
        """Computes the monetary cost together with its individual terms.

        Args:
            service (object): Service to be offloaded.
            server (object): Candidate edge server.
            user (object, optional): Unused; kept for a uniform cost-model interface.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            breakdown (dict): Dictionary with the compute, memory and transfer cost
                terms and the resulting total charge.
        """
        # Per-server pricing (if any) overrides the call parameters, which override the defaults.
        server_pricing = getattr(server, "cost_model_parameters", None) or {}
        params = {**cls.DEFAULT_PARAMETERS, **(parameters or {}), **server_pricing}

        data_size = resolve_input_data_size(service=service, parameters=params)
        processing = processing_delay(service=service, server=server, parameters=params)

        compute_cost = params["price_per_cpu_unit_time"] * service.cpu_demand * processing
        memory_cost = params["price_per_memory_unit_time"] * service.memory_demand * processing

        # Convert the offloaded payload from Mbit to GB (1 GB = 8192 Mbit).
        data_gb = data_size / 8192.0
        transfer_cost = params["price_per_gb_transferred"] * data_gb

        return {
            "compute": compute_cost,
            "memory": memory_cost,
            "transfer": transfer_cost,
            "total": compute_cost + memory_cost + transfer_cost,
        }
