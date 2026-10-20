"""Contains the Task component: a unit of computation executed on an edge server.

A ``Task`` is the missing *workload* primitive in EdgeSimPy. Where a ``Service`` models a
long-lived container reserved on a server, a ``Task`` models a discrete computation that
arrives at a point in time, waits in the server's CPU queue, executes for a duration
determined by its workload and the CPU it receives, and finishes -- optionally subject to
a deadline. This is what lets the simulator produce real response times, queueing delays
and deadline-violation rates instead of pure network-latency estimates.
"""

# EdgeSimPy components
from edge_sim_py.component_manager import ComponentManager

# Task lifecycle states.
WAITING = "waiting"
RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"


class Task(ComponentManager):
    """A discrete computation executed on an edge server."""

    _instances = []
    _object_count = 0

    def __init__(
        self,
        obj_id: int = None,
        label: str = "",
        cpu_cycles: float = 1.0,
        cpu_demand: int = 1,
        memory_demand: int = 0,
        input_size: float = 0.0,
        arrival_step: int = 0,
        deadline: int = None,
        server: object = None,
    ):
        """Creates a Task.

        Args:
            obj_id (int, optional): Object identifier. Defaults to None.
            label (str, optional): Human-readable label. Defaults to "".
            cpu_cycles (float, optional): Total compute work to execute. Defaults to 1.0.
            cpu_demand (int, optional): CPU units the task occupies while running (its
                degree of parallelism / processing rate). Defaults to 1.
            memory_demand (int, optional): Memory the task occupies while running. Defaults to 0.
            input_size (float, optional): Offloaded input data size (e.g. Mbit). Defaults to 0.0.
            arrival_step (int, optional): Step at which the task becomes schedulable. Defaults to 0.
            deadline (int, optional): Absolute step by which the task must finish; ``None``
                means no deadline. Defaults to None.
            server (object, optional): Edge server the task is assigned to. Defaults to None.
        """
        self.__class__._instances.append(self)
        self.__class__._object_count += 1
        if obj_id is None:
            obj_id = self.__class__._object_count
        self.id = obj_id

        self.label = label
        self.cpu_cycles = cpu_cycles
        self.remaining_cycles = cpu_cycles
        self.cpu_demand = cpu_demand
        self.memory_demand = memory_demand
        self.input_size = input_size
        self.arrival_step = arrival_step
        self.deadline = deadline
        self.server = server

        self.status = WAITING
        self.start_step = None
        self.end_step = None

    # ------------------------------------------------------------------ properties

    @property
    def finished(self) -> bool:
        """Whether the task completed successfully."""
        return self.status == FINISHED

    @property
    def failed(self) -> bool:
        """Whether the task was abandoned (e.g. its server failed and was not re-queued)."""
        return self.status == FAILED

    @property
    def waiting_time(self):
        """Steps spent queued before execution started (``None`` until it starts)."""
        if self.start_step is None:
            return None
        return self.start_step - self.arrival_step

    @property
    def execution_time(self):
        """Steps spent executing, over the inclusive interval (``None`` until finished)."""
        if self.start_step is None or self.end_step is None:
            return None
        return self.end_step - self.start_step + 1

    @property
    def response_time(self):
        """End-to-end steps from arrival to completion, inclusive (``None`` until finished)."""
        if self.end_step is None:
            return None
        return self.end_step - self.arrival_step + 1

    @property
    def met_deadline(self):
        """Whether the task finished within its deadline (``None`` if no deadline/unfinished)."""
        if self.deadline is None or self.end_step is None:
            return None
        return self.end_step <= self.deadline

    def collect(self) -> dict:
        """Returns a metrics snapshot for the task."""
        return {
            "Instance ID": self.id,
            "Status": self.status,
            "Server": self.server.id if self.server else None,
            "Arrival": self.arrival_step,
            "Start": self.start_step,
            "End": self.end_step,
            "Waiting Time": self.waiting_time,
            "Execution Time": self.execution_time,
            "Response Time": self.response_time,
            "Met Deadline": self.met_deadline,
        }
