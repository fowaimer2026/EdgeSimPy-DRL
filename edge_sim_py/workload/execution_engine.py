"""A discrete-time execution engine modeling task queueing, compute time and energy.

``ExecutionEngine`` simulates what happens *after* a placement decision: tasks assigned
to edge servers queue for CPU, execute for a duration set by their workload and the CPU
they receive, and finish (or miss their deadline). It optionally accumulates energy via an
:class:`~edge_sim_py.energy.energy_meter.EnergyMeter` and injects failures via a
:class:`~edge_sim_py.reliability.failure_model.FailureModel`, so a single run yields the
ground-truth metrics -- response time, queueing delay, deadline violations, energy and
reliability impact -- that the cost models and DRL agents can be validated against.

CPU model: a server runs as many queued tasks as fit within its CPU capacity (the sum of
running tasks' ``cpu_demand`` may not exceed ``server.cpu``); the rest wait (FIFO by
arrival). A running task advances ``cpu_demand * processing_rate`` cycles per tick, so its
execution time is ``cpu_cycles / (cpu_demand * processing_rate)`` and contention manifests
as queueing delay. The server's ``cpu_demand``/``memory_demand`` are updated to reflect the
running tasks, so the power models (and thus the energy meter) track the real load.
"""

# EdgeSimPy components
from edge_sim_py.workload.task import Task, WAITING, RUNNING, FINISHED, FAILED


class ExecutionEngine:
    """Steps assigned tasks through queueing and execution on a set of edge servers."""

    def __init__(
        self,
        servers: list,
        tasks: list,
        processing_rate: float = 1.0,
        tick_duration: float = 1.0,
        energy_meter: object = None,
        failure_model: object = None,
        requeue_on_failure: bool = True,
    ):
        """Creates an execution engine.

        Args:
            servers (list): Edge servers that execute tasks.
            tasks (list): Tasks to run; each must have its ``server`` assigned.
            processing_rate (float, optional): Cycles advanced per CPU unit per tick. Defaults to 1.0.
            tick_duration (float, optional): Duration of one tick (for energy). Defaults to 1.0.
            energy_meter (EnergyMeter, optional): If given, energy is sampled each tick. Defaults to None.
            failure_model (FailureModel, optional): If given, failures are injected each tick. Defaults to None.
            requeue_on_failure (bool, optional): If True, tasks running on a failing server
                return to its queue; if False, they are marked FAILED. Defaults to True.
        """
        self.servers = servers
        self.tasks = tasks
        self.processing_rate = processing_rate
        self.tick_duration = tick_duration
        self.energy_meter = energy_meter
        self.failure_model = failure_model
        self.requeue_on_failure = requeue_on_failure

        self.current_step = 0
        self.events = []
        # Per-server queues, keyed by id(server) (servers are hashable, but keep it uniform).
        self._waiting = {}
        self._running = {}
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        """Resets task/server state so the engine can run from scratch."""
        self.current_step = 0
        self.events = []
        self._waiting = {id(server): [] for server in self.servers}
        self._running = {id(server): [] for server in self.servers}

        for server in self.servers:
            server.cpu_demand = 0
            server.memory_demand = 0

        for task in self.tasks:
            task.status = WAITING
            task.remaining_cycles = task.cpu_cycles
            task.start_step = None
            task.end_step = None
            if task.server is not None:
                self._waiting[id(task.server)].append(task)

        # Schedule each queue by arrival order (FIFO among arrived tasks).
        for queue in self._waiting.values():
            queue.sort(key=lambda task: task.arrival_step)

    # ------------------------------------------------------------------ per-tick logic

    def _handle_failures(self) -> None:
        """Applies the failure model and re-queues (or fails) tasks on failed servers."""
        if self.failure_model is None:
            return

        events = self.failure_model.step(self.servers, self.current_step)
        self.events.extend(events)

        for event in events:
            if event["event"] != "failure":
                continue
            server = event["server"]
            for task in self._running[id(server)]:
                if self.requeue_on_failure:
                    task.status = WAITING
                    task.remaining_cycles = task.cpu_cycles  # restart the interrupted task
                    task.start_step = None
                    self._waiting[id(server)].append(task)
                else:
                    task.status = FAILED
                    task.end_step = self.current_step
            self._running[id(server)] = []
            server.cpu_demand = 0
            server.memory_demand = 0

    def _admit_tasks(self) -> None:
        """Admits queued tasks onto each available server while CPU capacity allows."""
        for server in self.servers:
            if not server.available:
                continue

            queue = self._waiting[id(server)]
            running = self._running[id(server)]

            # FIFO admission limited by remaining CPU capacity.
            remaining_queue = []
            for task in queue:
                used_cpu = sum(running_task.cpu_demand for running_task in running)
                if task.arrival_step <= self.current_step and used_cpu + task.cpu_demand <= server.cpu:
                    task.status = RUNNING
                    task.start_step = self.current_step
                    running.append(task)
                else:
                    remaining_queue.append(task)
            self._waiting[id(server)] = remaining_queue

            # Update the server's reserved demand to reflect the running tasks.
            server.cpu_demand = sum(running_task.cpu_demand for running_task in running)
            server.memory_demand = sum(running_task.memory_demand for running_task in running)

    def _progress_tasks(self) -> None:
        """Advances running tasks and finalizes the ones that complete this tick."""
        for server in self.servers:
            if not server.available:
                continue

            still_running = []
            for task in self._running[id(server)]:
                task.remaining_cycles -= task.cpu_demand * self.processing_rate
                if task.remaining_cycles <= 0:
                    task.remaining_cycles = 0
                    task.status = FINISHED
                    task.end_step = self.current_step
                else:
                    still_running.append(task)
            self._running[id(server)] = still_running
            server.cpu_demand = sum(running_task.cpu_demand for running_task in still_running)
            server.memory_demand = sum(running_task.memory_demand for running_task in still_running)

    def step(self) -> None:
        """Advances the simulation by one tick.

        The step counter is incremented at the *end* of the tick, so the first tick runs at
        step 0; a task with ``arrival_step == 0`` can therefore start immediately (zero
        waiting time). Tasks occupy the inclusive step interval ``[start_step, end_step]``.
        """
        self._handle_failures()
        self._admit_tasks()
        self._progress_tasks()
        if self.energy_meter is not None:
            self.energy_meter.sample(self.servers)
        self.current_step += 1

    # ------------------------------------------------------------------ drivers

    def all_settled(self) -> bool:
        """Whether every task has finished or failed."""
        return all(task.status in (FINISHED, FAILED) for task in self.tasks)

    def run(self, max_steps: int = 100_000) -> int:
        """Runs until all tasks settle or ``max_steps`` is reached.

        Args:
            max_steps (int, optional): Safety bound on the number of ticks. Defaults to 100000.

        Returns:
            steps (int): Number of ticks executed.
        """
        while not self.all_settled() and self.current_step < max_steps:
            self.step()
        return self.current_step
