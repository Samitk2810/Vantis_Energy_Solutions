from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from rag.retriever import get_context
from typing import Optional

from pydantic import BaseModel, Field
from agents.state import AgentState
from tools.service_area import check_service_area, find_supported_city
load_dotenv()

SYSTEM_PROMPT = """You are the Vantis Energy Solutions assistant, helping residential and small commercial customers with rooftop solar.

Answer using ONLY the knowledge base context below. Treat it as the single source of truth.

Style:
- Be conversational, concise and friendly. Do not act like an aggressive salesperson.
- Answer the question first. Then, if useful, ask ONE relevant follow-up question.
- Never mention "context", "knowledge base", "sections" or "documents" to the customer.

Hard rules. Never invent or state:
- an exact price or price range for any system or package
- exact savings amounts or percentages
- subsidy amounts or guaranteed subsidy eligibility
- warranty durations
- installation dates or guaranteed timelines
- financing approval or EMI amounts
- equipment specifications that are not in the context
- service availability for a city that is not listed in the context
- appointments or actions by a human representative that did not actually happen

How to handle these:
- Price: explain there is no single fixed price and that a consultation is needed for a project-specific cost.
- Savings or subsidy: explain they depend on the project and need verification.
- Warranty: explain it depends on the component and manufacturer terms.
- Installation date: explain scheduling depends on project, site and equipment factors.
- City not listed: say availability is not confirmed in your current information and offer to collect details so the sales team can verify.
- System size: do not recommend a size without the customer's consumption or bill and property type. Ask for them first.

If the answer is not in the context, say you don't have that information and offer to connect them with the sales team. Do not guess or use general solar knowledge for Vantis-specific facts.

Context:
{context}"""

REWRITE_PROMPT = (
    "Rewrite the customer's last message as a standalone search query for a solar "
    "company knowledge base. Use the chat history to resolve references like 'that' "
    "or 'the 10 kW one'. Output only the query."
)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

def rewrite_query(question: str, history: list) -> str:
    if not history:
        return question
    convo = "\n".join(f"{role}: {text}" for role, text in history[-6:])
    msg = f"{REWRITE_PROMPT}\n\nChat history:\n{convo}\n\nLast message: {question}"
    return llm.invoke(msg).content.strip()


def answer(question: str, history: list | None = None, note: str = "") -> str:
    """history = list of (role, text) tuples, role is 'user' or 'assistant'."""
    history = history or []
    query = rewrite_query(question, history)
    context = get_context(query)

    system = SYSTEM_PROMPT.format(context=context)
    if note:
        system += f"\n\nSystem note for this reply: {note}"
    messages = [SystemMessage(content=system)]
    for role, text in history[-6:]:
        messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
    messages.append(HumanMessage(content=question))
    return llm.invoke(messages).content

# ---------------- graph node ----------------

class CityMention(BaseModel):
    city: Optional[str] = Field(
        default=None,
        description="The city the customer gives as their location or where they want installation. null if none.",
    )


city_extractor = llm.with_structured_output(CityMention)

LISTED_NOTE = "{city} is a listed Vantis service location."
UNVERIFIED_NOTE = (
    "{city} is NOT in Vantis's listed service locations. Say service availability is not confirmed "
    "in your current information, and offer to collect the customer's exact location and contact "
    "details so the sales team can verify. Do not say Vantis does or does not operate there."
)


def detect_city(question: str):
    """Listed cities are matched in code. Other cities are picked up by the LLM."""
    listed = find_supported_city(question)
    if listed:
        return listed
    try:
        return city_extractor.invoke(question).city
    except Exception:
        return None


def knowledge_node(state: AgentState):
    question = state["messages"][-1].content
    history = [
        ("user" if m.type == "human" else "assistant", m.content)
        for m in state["messages"][:-1]
    ]

    update = {}
    note = ""
    city = detect_city(question)
    if city:
        status, city = check_service_area(city)
        update["lead_fields"] = {"city": city}
        update["location_status"] = status
        if status == "supported":
            note = LISTED_NOTE.format(city=city)
        else:
            note = UNVERIFIED_NOTE.format(city=city)
            update["awaiting"] = "lead_details"

    reply = answer(question, history, note)
    update["reply"] = (state.get("reply") or "") + reply
    return update


if __name__ == "__main__":
    tests = [
        "Do you install in Kota?",
        "What about Pune?",
        "How much does a 5 kW system cost?",
        "What warranty do the panels have?",
        "How much will I save per month?",
        "Tell me about HomePro 5",
    ]
    for q in tests:
        print(f"\nQ: {q}\nA: {answer(q)}")