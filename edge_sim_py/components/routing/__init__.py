"""Automatic Python configuration file."""

__version__ = "1.1.0"

# Pluggable routing policies
from .shortest_path_routing import ShortestPathRouting
from .least_congested_routing import LeastCongestedRouting
from .latency_aware_routing import LatencyAwareRouting
from .ecmp_routing import ECMPRouting

# Programmable-switch (SDN) abstraction
from .programmable_switch import (
    FlowRule,
    RoutingController,
    install_flow_rule,
    lookup_flow_rule,
    clear_flow_table,
)
