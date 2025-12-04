"""Metta Spider - Alignment League submission."""

from mettagrid.policy.policy import MultiAgentPolicy, AgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class MettaSpiderPolicy(MultiAgentPolicy):
    """Multi-agent policy wrapper."""

    short_names = ["metta_spider", "spider"]

    def __init__(self, policy_env_info: PolicyEnvInterface, **kwargs):
        super().__init__(policy_env_info, **kwargs)
        self._agent_policies: dict[int, AgentPolicy] = {}

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = SpiderAgentPolicy(self._policy_env_info, agent_id)
        return self._agent_policies[agent_id]


class SpiderAgentPolicy(AgentPolicy):
    """Per-agent policy that decides actions from observations."""

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        self.agent_id = agent_id

    def step(self, obs: AgentObservation) -> Action:
        # TODO: Replace with your logic
        # obs.tokens contains visible game state
        # Return Action(name="move_up"), Action(name="use"), etc.
        return Action(name="noop")
