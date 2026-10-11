import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.knowledge_agent import knowledge_node
from agents.sales_agent import lead_node
from agents.orchestrator import next_after, orchestrator_node, route_after_orchestrator
from agents.support_agent import support_node
from db.repo import get_lead, init_db
from agents.state import AgentState

FALLBACK_REPLY = "Sorry, I'm having a little trouble right now. Please try again in a moment."


def respond_node(state: AgentState):
    """Store the final reply in the chat history so the next turn can see it."""
    return {"messages": [AIMessage(content=state.get("reply") or "")]}


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("knowledge", knowledge_node)
    g.add_node("lead", lead_node)
    g.add_node("support", support_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges(
        "orchestrator", route_after_orchestrator,
        {"knowledge": "knowledge", "lead": "lead", "support": "support", END: "respond"},
    )
    g.add_conditional_edges("knowledge", next_after("knowledge"),
                            {"lead": "lead", "support": "support", END: "respond"})
    g.add_conditional_edges("lead", next_after("lead"), {"support": "support", END: "respond"})
    g.add_conditional_edges("support", next_after("support"), {END: "respond"})
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


def make_checkpointer(path: str = "data/checkpoints.db"):
    """SQLite-backed memory so conversations survive restarts."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    except ImportError:
        print("langgraph-checkpoint-sqlite not installed: using in-memory sessions")
        return MemorySaver()


init_db()
graph = build_graph(make_checkpointer())


def run_turn(session_id: str, message: str, known_fields: dict | None = None) -> dict:
    """One customer message in, one reply out. Never raises."""
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"session_id": session_id, "messages": [HumanMessage(content=message)]}
    if known_fields:                      # e.g. the phone number a WhatsApp message came from
        inputs["lead_fields"] = known_fields
    try:
        out = graph.invoke(inputs, config)
    except Exception as exc:
        print("run_turn error:", repr(exc))
        return {"reply": FALLBACK_REPLY, "intents": [], "lead": None, "error": True}
    lead = get_lead(out["lead_id"]) if out.get("lead_id") else None
    return {
        "reply": out.get("reply") or FALLBACK_REPLY,
        "intents": out.get("intents", []),
        "lead": lead,
    }


if __name__ == "__main__":
    sid = "cli-test"
    print("Vantis assistant (type 'quit' to exit)")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        result = run_turn(sid, text)
        print(f"Bot: {result['reply']}")
        if result["lead"]:
            lead = result["lead"]
            print(f"[lead #{lead['id']}: {lead['lead_status']} {lead['lead_score']}]")