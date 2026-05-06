"""P5: X-Agent-ID — وسم agent_id للنتائج وتمييزها في بيئات Swarm.

Adds agent identification to memory results, enabling:
1. Tracking which agent created/owns a memory
2. Filtering search results by agent_id
3. Cross-agent memory sharing with attribution
4. Swarm environment coordination

Usage:
    from unified.agent_id import AgentIdMixin

    # In UnifiedMemoryProvider:
    class UnifiedMemoryProvider(MemoryProvider, AgentIdMixin):
        pass

    # Set agent_id during initialization
    provider.initialize(session_id="abc", agent_id="agent-5")

    # Search with agent filter
    results = provider._search_graph_cached(query, agent_id="agent-5")
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ============================================================
# Agent ID configuration
# ============================================================

DEFAULT_AGENT_ID = "default"
AGENT_ID_KEY = "agent_id"


class AgentIdMixin:
    """P5: Adds agent_id tracking and filtering to memory operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent_id = DEFAULT_AGENT_ID

    def _get_agent_id(self) -> str:
        """Get the current agent ID."""
        return self._agent_id

    def _set_agent_id(self, agent_id: str):
        """Set the agent ID for this provider instance."""
        if agent_id and agent_id.strip():
            self._agent_id = agent_id.strip()
            logger.info("Agent ID set to: %s", self._agent_id)
        else:
            self._agent_id = DEFAULT_AGENT_ID

    def _annotate_with_agent_id(self, result: Dict) -> Dict:
        """Add agent_id to a search result."""
        if result is not None and isinstance(result, dict):
            result[AGENT_ID_KEY] = self._agent_id
        return result

    def _annotate_results_with_agent_id(self, results: List[Dict]) -> List[Dict]:
        """Add agent_id to all search results."""
        for r in results:
            self._annotate_with_agent_id(r)
        return results

    def _filter_by_agent_id(
        self,
        results: List[Dict],
        agent_id: Optional[str] = None,
    ) -> List[Dict]:
        """Filter results by agent_id.

        If agent_id is None or 'all', returns all results.
        Otherwise returns only results matching the agent_id.
        """
        if agent_id is None or agent_id.lower() in ('all', '*'):
            return results

        return [
            r for r in results
            if r.get(AGENT_ID_KEY) == agent_id
        ]

    def _get_agent_id_from_args(self, args: Dict) -> Optional[str]:
        """Extract agent_id from tool call arguments."""
        return args.get(AGENT_ID_KEY)

    def get_agent_id_schema(self) -> Dict[str, Any]:
        """Get the agent_id parameter schema for tool calls."""
        return {
            AGENT_ID_KEY: {
                "type": "string",
                "description": (
                    "P5: Filter by agent ID. "
                    "Use 'all' or omit for all agents. "
                    "Examples: 'agent-1', 'hermes-swarm-3'"
                ),
            },
        }

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about the current agent's memory."""
        return {
            AGENT_ID_KEY: self._agent_id,
            "status": "active",
        }
