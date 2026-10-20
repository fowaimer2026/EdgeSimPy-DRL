"""Contains a computation-offloading energy cost model definition."""

# EdgeSimPy components
from edge_sim_py.components.cost_models.base_cost_model import (
    resolve_input_data_size,
    transmission_delay,
    network_path_delay,
    processing_delay,
)


class EnergyCostModel:
    """Energy cost of offloading a service to a candidate edge server.

    The model captures both the energy spent on the end device while the task is
    offloaded and the energy spent by the edge server executing it [1]:

        * device transmit energy -- radio power x transmission delay;
        * device idle energy     -- idle power x (propagation + processing), i.e.
          the device waiting for the remote result;
        * server compute energy  -- the server's power draw (reusing EdgeSimPy's
          existing power model when available) x processing delay.

    [1] W. Zhang et al., "Energy-Optimal Mobile Cloud Computing under Stochastic
        Wireless Channel", IEEE Transactions on Wireless Communications, 2013.
    """

    # Default parameters. Powers are expressed in Watts (energy = power x time).
    DEFAULT_PARAMETERS = {
        "input_data_size": None,  # Mbit to offload (falls back to service.state, then 1.0)
        "access_bandwidth": 100.0,  # Mbps available between the user and its base station
        "cpu_cycles_per_unit": 1.0,  # workload scaling applied to the service CPU demand
        "device_transmit_power": 1.5,  # device radio power while transmitting
        "device_idle_power": 0.1,  # device power while waiting for the remote result
        "server_idle_power": 100.0,  # fallback server power when no power model is set
    }

    @classmethod
    def get_cost(cls, service: object, server: object, user: object = None, parameters: dict = None) -> float:
        """Computes the energy cost of hosting ``service`` on ``server`` for ``user``.

        Args:
            service (object): Service to be offloaded.
            server (object): Candidate edge server.
            user (object, optional): User accessing the service. Defaults to None.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            cost (float): Estimated energy consumption.
        """
        return cls.get_cost_breakdown(service, server, user, parameters)["total"]

    @classmethod
    def get_cost_breakdown(cls, service: object, server: object, user: object = None, parameters: dict = None) -> dict:
        """Computes the energy cost together with its individual terms.

        Args:
            service (object): Service to be offloaded.
            server (object): Candidate edge server.
            user (object, optional): User accessing the service. Defaults to None.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            breakdown (dict): Dictionary with the device-transmit, device-idle and
                server-compute energy terms and the resulting total energy.
        """
        params = {**cls.DEFAULT_PARAMETERS, **(parameters or {})}

        data_size = resolve_input_data_size(service=service, parameters=params)

        transmission = transmission_delay(data_size=data_size, bandwidth=params["access_bandwidth"])
        propagation = network_path_delay(user=user, server=server)
        processing = processing_delay(service=service, server=server, parameters=params)

        # Energy spent by the end device: transmitting the payload, then idling while
        # the request travels the network and is processed remotely.
        device_transmit_energy = params["device_transmit_power"] * transmission
        device_idle_energy = params["device_idle_power"] * (propagation + processing)

        # Energy spent by the edge server: reuse the configured power model so the
        # estimate is consistent with the rest of the simulation. The power model
        # reports an instantaneous draw (W), which we integrate over the processing time.
        server_power = params["server_idle_power"]
        if getattr(server, "power_model", None) is not None:
            try:
                server_power = server.get_power_consumption()
            except Exception:
                server_power = params["server_idle_power"]
        server_compute_energy = server_power * processing

        return {
            "device_transmit": device_transmit_energy,
            "device_idle": device_idle_energy,
            "server_compute": server_compute_energy,
            "total": device_transmit_energy + device_idle_energy + server_compute_energy,
        }
