"""
LangGraph Orchestration
-----------------------------------
Wires five agents into a state graph:

  START -> planner -> query -> retrieve -> forecast -> insight -> critic
                                                            ^          |
                                                            |__________|
                                                    (loops back ONCE if
                                                     critic finds an issue,
                                                     with the flagged issue
                                                     passed to Insight)

query/retrieve/forecast nodes only do work if the planner's chosen plan
calls for them; otherwise they pass state through unchanged.

retrieve_node pulls 3 chunks instead of 2 for better coverage on
questions needing a specific fact from a less-obvious chunk.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.agents import planner_agent, query_agent, insight_agent, critic_agent, forecast_agent
from src.tools import retrieve_context


class AgentState(TypedDict):
    question: str
    plan: str
    sql: str
    sql_result: str
    context: str
    forecast_result: str
    answer: str
    verdict: str
    issue: str
    revision_count: int


def planner_node(state: AgentState) -> AgentState:
    plan = planner_agent(state["question"])
    print(f"[Planner] plan = {plan}")
    return {**state, "plan": plan}


def query_node(state: AgentState) -> AgentState:
    if state["plan"] in ("sql", "both"):
        result = query_agent(state["question"])
        print(f"[Query Agent] SQL: {result['sql']}")
        return {**state, "sql": result["sql"], "sql_result": result["result"]}
    return state


def retrieve_node(state: AgentState) -> AgentState:
    if state["plan"] in ("context", "both"):
        context = retrieve_context(state["question"], n_results=3)
        print(f"[RAG] retrieved {len(context)} chars of context")
        return {**state, "context": context}
    return state


def forecast_node(state: AgentState) -> AgentState:
    if state["plan"] == "forecast":
        result = forecast_agent(state["question"])
        print(f"[Forecast Agent] {result['forecast_result']}")
        return {**state, "forecast_result": result["forecast_result"]}
    return state


def insight_node(state: AgentState) -> AgentState:
    is_revision = state.get("revision_count", 0) > 0
    answer = insight_agent(
        state["question"],
        state.get("sql_result", ""),
        state.get("context", ""),
        state.get("forecast_result", ""),
        prior_issue=state.get("issue", "") if is_revision else "",
    )
    print(f"[Insight]{' (revision)' if is_revision else ''} {answer}")
    return {**state, "answer": answer}


def critic_node(state: AgentState) -> AgentState:
    critique = critic_agent(
        state["question"],
        state["answer"],
        state.get("sql_result", ""),
        state.get("context", ""),
        state.get("forecast_result", ""),
    )
    print(f"[Critic] verdict = {critique['verdict']}" + (f" | issue: {critique['issue']}" if critique['verdict'] == "NEEDS_REVISION" else ""))
    return {
        **state,
        "verdict": critique["verdict"],
        "issue": critique["issue"],
        "revision_count": state.get("revision_count", 0) + 1,
    }


def route_after_critic(state: AgentState) -> str:
    """If approved, we're done. Allows exactly one real revision attempt:
    on the first Critic pass (revision_count == 1), a NEEDS_REVISION
    verdict routes back to Insight with the flagged issue attached. On
    the second pass (revision_count == 2), the answer is accepted either
    way, bounding total API calls per question."""
    if state["verdict"] == "APPROVED" or state["revision_count"] >= 2:
        return "end"
    return "revise"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("query", query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("insight", insight_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "query")
    graph.add_edge("query", "retrieve")
    graph.add_edge("retrieve", "forecast")
    graph.add_edge("forecast", "insight")
    graph.add_edge("insight", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"end": END, "revise": "insight"},
    )

    return graph.compile()


def ask(question: str) -> dict:
    app = build_graph()
    initial_state: AgentState = {
        "question": question,
        "plan": "",
        "sql": "",
        "sql_result": "",
        "context": "",
        "forecast_result": "",
        "answer": "",
        "verdict": "",
        "issue": "",
        "revision_count": 0,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    import time

    test_questions = [
        "How many store type 'b' stores are in the dataset?",
        "What will store 1's sales be on 2015-08-20?",
    ]

    for i, q in enumerate(test_questions):
        if i > 0:
            print("\n[pausing 20s]")
            time.sleep(20)
        print("\n" + "=" * 60)
        print(f"QUESTION: {q}")
        print("=" * 60)
        result = ask(q)
        print(f"\nFINAL ANSWER: {result['answer']}")
        print(f"REVISIONS: {result['revision_count'] - 1}")