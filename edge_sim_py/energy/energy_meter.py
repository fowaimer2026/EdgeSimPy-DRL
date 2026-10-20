"""Cumulative energy accounting over time for EdgeSimPy devices.

EdgeSimPy's power models report an *instantaneous* draw (Watts) via
``get_power_consumption()``, but nothing integrates that draw into energy consumed over
time. ``EnergyMeter`` closes that gap: sampled once per simulation tick, it accumulates
``power x tick_duration`` per device and in total, turning the existing power models into
energy/SLA-grade metrics (e.g. Joules or Wh, depending on the units used).

It works with any device exposing ``get_power_consumption()`` -- edge servers and
network switches -- and can be driven either by the standalone
:class:`~edge_sim_py.workload.execution_engine.ExecutionEngine` or as a monitor hook on
a live :class:`~edge_sim_py.simulator.Simulator`.
"""


class EnergyMeter:
    """Accumulates energy consumed by a set of devices across simulation ticks."""

    def __init__(self, tick_duration: float = 1.0):
        """Creates an energy meter.

        Args:
            tick_duration (float, optional): Duration of one simulation tick, used to turn
                instantaneous power into energy (energy += power * tick_duration). Defaults to 1.0.
        """
        self.tick_duration = tick_duration
        self.total_energy = 0.0
        self.samples = 0
        # Per-device cumulative energy, keyed by ``id(device)`` because some devices
        # (e.g. NetworkLink) are unhashable; ``_labels`` keeps a readable name alongside.
        self._per_device = {}
        self._labels = {}

    def sample(self, devices: list) -> float:
        """Accumulates one tick of energy for every device, returning the tick's energy.

        Args:
            devices (list): Devices exposing ``get_power_consumption()``.

        Returns:
            tick_energy (float): Total energy consumed by the devices during this tick.
        """
        tick_energy = 0.0
        for device in devices:
            power = device.get_power_consumption()
            energy = power * self.tick_duration
            key = id(device)
            self._per_device[key] = self._per_device.get(key, 0.0) + energy
            self._labels[key] = str(device)
            tick_energy += energy

        self.total_energy += tick_energy
        self.samples += 1
        return tick_energy

    def energy_of(self, device: object) -> float:
        """Returns the cumulative energy consumed by a single device."""
        return self._per_device.get(id(device), 0.0)

    def average_power(self) -> float:
        """Returns the mean total power across all sampled ticks."""
        if self.samples == 0:
            return 0.0
        return self.total_energy / (self.samples * self.tick_duration)

    def reset(self) -> None:
        """Clears all accumulated energy."""
        self.total_energy = 0.0
        self.samples = 0
        self._per_device = {}
        self._labels = {}

    def metrics(self) -> dict:
        """Returns a summary of the accumulated energy.

        Returns:
            metrics (dict): Total energy, number of ticks sampled, average power and a
                per-device breakdown keyed by device label.
        """
        return {
            "total_energy": self.total_energy,
            "ticks_sampled": self.samples,
            "average_power": self.average_power(),
            "per_device": {self._labels[key]: energy for key, energy in self._per_device.items()},
        }
