#!/usr/bin/env python3

"""Configuration facade for the online frontier explorer.

The public mixin preserves the old configuration entry point while sibling
modules own each independent configuration domain. Keeping the order here
makes the small number of cross-domain dependencies explicit.
"""

from global_frontier_config_exploration import (
    GlobalFrontierExplorationConfigurationMixin,
)
from global_frontier_config_navigation import (
    GlobalFrontierNavigationConfigurationMixin,
)
from global_frontier_config_topics import GlobalFrontierTopicConfigurationMixin


class GlobalFrontierConfigurationMixin(
    GlobalFrontierExplorationConfigurationMixin,
    GlobalFrontierNavigationConfigurationMixin,
    GlobalFrontierTopicConfigurationMixin,
):
    """Resolve static configuration without mixing unrelated policy domains."""

    def _load_parameters(self, gp):
        """Load groups in dependency order while preserving ROS parameter names."""
        self._load_topic_parameters(gp)
        self._load_navigation_parameters(gp)
        self._load_exploration_parameters(gp)
