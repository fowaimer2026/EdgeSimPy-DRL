"""Contains a programmable-switch (SDN/P4-style) abstraction for the routing layer.

This module models the *programmability* of edge network switches called out in the
EdgeSimPy future-work agenda. It adds a match-action flow table to switches and a
lightweight controller that, given a routing policy, computes a path and installs the
corresponding forwarding rules on every switch along it. Data-plane forwarding can
then be reconstructed by following the installed rules hop by hop -- exactly the
control-plane / data-plane separation of software-defined networking.
"""

# EdgeSimPy components
from edge_sim_py.components.routing.shortest_path_routing import ShortestPathRouting


class FlowRule:
    """A match-action forwarding rule installed on a programmable switch.

    A packet/flow is described by a small metadata dict (e.g. ``{"flow_id": 7}`` or
    ``{"app_id": 3}``). A rule *matches* when every key in its ``match`` dict equals the
    corresponding value in the packet's metadata; an empty ``match`` is a wildcard
    (table-miss / default) rule. The ``action`` dict describes what to do -- typically
    ``{"next_hop": <NetworkSwitch>}`` to forward toward the next switch on the path.
    """

    def __init__(self, match: dict = None, action: dict = None, priority: int = 0):
        """Creates a FlowRule.

        Args:
            match (dict, optional): Header fields to match (empty = wildcard). Defaults to {}.
            action (dict, optional): Action to apply on a match. Defaults to {}.
            priority (int, optional): Higher-priority rules win ties. Defaults to 0.
        """
        self.match = match or {}
        self.action = action or {}
        self.priority = priority

    def matches(self, packet: dict) -> bool:
        """Returns whether this rule matches a packet's metadata."""
        return all(packet.get(key) == value for key, value in self.match.items())

    def __repr__(self) -> str:
        return f"FlowRule(match={self.match}, action={self.action}, priority={self.priority})"


def install_flow_rule(switch: object, rule: FlowRule) -> None:
    """Installs a flow rule on a switch, keeping the table ordered by descending priority.

    Args:
        switch (object): NetworkSwitch to program.
        rule (FlowRule): Rule to install.
    """
    table = getattr(switch, "flow_table", None)
    if table is None:
        table = []
        switch.flow_table = table
    table.append(rule)
    table.sort(key=lambda installed_rule: installed_rule.priority, reverse=True)


def lookup_flow_rule(switch: object, packet: dict) -> FlowRule:
    """Returns the highest-priority rule on a switch matching a packet (or ``None``).

    Args:
        switch (object): NetworkSwitch to query.
        packet (dict): Packet/flow metadata.

    Returns:
        rule (FlowRule): Matching rule, or ``None`` on a table miss.
    """
    for rule in getattr(switch, "flow_table", []):
        if rule.matches(packet):
            return rule
    return None


def clear_flow_table(switch: object) -> None:
    """Removes every rule installed on a switch."""
    switch.flow_table = []


class RoutingController:
    """A minimal SDN controller that programs switches according to a routing policy.

    The controller separates the *control plane* (deciding a path with a routing
    policy) from the *data plane* (the per-switch flow tables that actually forward
    traffic), letting experiments compare policies purely by the rules they install.
    """

    @classmethod
    def install_path(
        cls,
        topology: object,
        source: object,
        target: object,
        policy=ShortestPathRouting,
        flow: object = None,
        packet: dict = None,
        priority: int = 1,
        parameters: dict = None,
    ) -> list:
        """Computes a path with ``policy`` and installs forwarding rules along it.

        On each non-terminal switch of the path a rule is installed whose action
        forwards matching packets to the next switch. The terminal switch receives a
        ``deliver`` action.

        Args:
            topology (object): Network topology graph.
            source (object): Source network switch.
            target (object): Target network switch.
            policy: Routing policy class exposing ``get_path``. Defaults to ShortestPathRouting.
            flow (object, optional): Flow being routed (forwarded to the policy and used
                as the default match key). Defaults to None.
            packet (dict, optional): Match key for the installed rules. Defaults to a
                ``{"flow_id": flow.id}`` match, or a wildcard rule when no flow is given.
            priority (int, optional): Priority of the installed rules. Defaults to 1.
            parameters (dict, optional): Policy parameters. Defaults to None.

        Returns:
            path (list): The path the rules were installed along (``[]`` if unreachable).
        """
        path = policy.get_path(topology=topology, source=source, target=target, flow=flow, parameters=parameters)
        if len(path) < 1:
            return []

        # Default match: everything (wildcard) unless we can key on the flow id.
        if packet is None:
            packet = {"flow_id": flow.id} if flow is not None and getattr(flow, "id", None) is not None else {}

        for index, switch in enumerate(path):
            if index < len(path) - 1:
                action = {"next_hop": path[index + 1]}
            else:
                action = {"deliver": True}
            install_flow_rule(switch=switch, rule=FlowRule(match=dict(packet), action=action, priority=priority))

        return path

    @classmethod
    def forward(cls, source: object, packet: dict, max_hops: int = 1000) -> list:
        """Reconstructs the data-plane path a packet takes by following installed rules.

        Starting at ``source``, the controller repeatedly looks up the matching rule and
        follows its ``next_hop`` action until a ``deliver`` action (or a table miss) is
        reached, mirroring how a packet traverses a chain of programmable switches.

        Args:
            source (object): Switch where forwarding begins.
            packet (dict): Packet/flow metadata used for rule lookups.
            max_hops (int, optional): Safety bound against forwarding loops. Defaults to 1000.

        Returns:
            path (list): The switches the packet visits, ending at the delivery switch.
                Empty if the source has no matching rule.
        """
        path = []
        current = source

        for _ in range(max_hops):
            rule = lookup_flow_rule(switch=current, packet=packet)
            if rule is None:
                # Table miss: forwarding stops (drop) unless we already started a path.
                return path
            path.append(current)
            if rule.action.get("deliver"):
                return path
            next_hop = rule.action.get("next_hop")
            if next_hop is None:
                return path
            current = next_hop

        return path
