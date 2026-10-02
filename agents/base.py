"""agents/base.py - shared helpers: AgentSpec (role / work / rules), builder, runner.

Every agent file defines an AgentSpec and a build() function.
create_react_agent gives each agent the ReAct loop:  Think -> Act (call tool) -> Observe -> ... -> Answer.
"""
from dataclasses import dataclass, field

from langgraph.prebuilt import create_react_agent

import config
from llm import strip_think

COMMON_RULES = [
    "LANGUAGE (very important): look ONLY at the patient's latest message. If it is English, reply ONLY in English. "
    "If it is Urdu script, reply in Urdu script. If it is Roman Urdu (Urdu in English letters, like 'mujhe bukhar hai'), "
    "reply in Roman Urdu. 'hello' or 'hello doctor' is English. Never switch language on your own, "
    "never mix scripts, never use emojis or special symbols.",
]


@dataclass
class AgentSpec:
    name: str                     # unique id, also used as the tool name
    description: str              # one line: what this agent does (the doctor agent reads this)
    role: str                     # who the agent is
    work: list = field(default_factory=list)      # what it must do, step by step
    rules: list = field(default_factory=list)     # agent specific rules

    def system_prompt(self) -> str:
        work = "\n".join(f"{i}. {w}" for i, w in enumerate(self.work, 1))
        rules = "\n".join(f"- {r}" for r in COMMON_RULES + self.rules)
        return (f"ROLE:\n{self.role}\n\nYOUR WORK:\n{work}\n\nRULES:\n{rules}\n\n"
                "Think step by step, use tools when you need facts, then give the final answer.")


def build_agent(spec: AgentSpec, llm, tools, checkpointer=None):
    return create_react_agent(model=llm, tools=tools, prompt=spec.system_prompt(),
                              checkpointer=checkpointer, name=spec.name)


def last_text(result) -> str:
    return strip_think(result["messages"][-1].content)


def show_steps(label: str, messages):
    """DEBUG=1 -> print which tools the agent called (its ReAct 'acting' steps)."""
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            print(f"   [{label}] tool -> {call['name']}({call['args']})")


def run_agent(agent, label: str, text: str) -> str:
    """Call a sub-agent and return its final answer as text."""
    result = agent.invoke({"messages": [("user", text)]})
    if config.DEBUG:
        show_steps(label, result["messages"])
    return last_text(result)
