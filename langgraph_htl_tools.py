import os
from typing import TypedDict, Literal, List, Annotated
import operator
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

# -------------------------------------------------------------------------
# 1. Define Tools
# -------------------------------------------------------------------------
@tool
def check_system_logs(environment: str) -> str:
    """Fetch diagnostic log snippets for a specific environment (dev or test)."""
    if "dev" in environment.lower():
        return "Log snippet: [ERROR] Dev DB connection timeout at 10:14 AM."
    return "Log snippet: [FAIL] Test runner exited with code 1 in pytest integration suite."

tools = [check_system_logs]

# -------------------------------------------------------------------------
# 2. Define Shared State (Using message history for tool compatibility)
# -------------------------------------------------------------------------
class TicketState(TypedDict):
    ticket_input: str
    messages: Annotated[List[BaseMessage], operator.add]
    classification: str
    resolution: str
    human_approved: bool

# -------------------------------------------------------------------------
# 3. Define LLM & Nodes
# -------------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
).bind_tools(tools)

def route_ticket(state: TicketState) -> Literal["dev_node", "test_node"]:
    """Classifies incoming input strictly into 'test' or 'dev'."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an automated IT support router. Classify the ticket strictly into 'test' or 'dev'."),
        ("user", "Ticket description: {input}")
    ])
    chain = prompt | llm
    response = chain.invoke({"input": state["ticket_input"]})
    result = response.content.lower()

    if "test" in result:
        return "test_node"
    return "dev_node"

def process_dev_ticket(state: TicketState):
    """Dev node using Tool Calling logic."""
    print("\n--- Executing DEV node logic (LLM + Tools) ---")
    
    # Prompt instructing LLM to fetch logs using the tool first
    messages = [
        ("system", "You are a DevOps engineer. Check the system logs for 'dev' environment using the provided tool, then summarize the issue."),
        ("user", f"Ticket: {state['ticket_input']}")
    ]
    response = llm.invoke(messages)
    return {
        "classification": "dev",
        "messages": [response],
        "resolution": "Proposed Dev Action: Database server restart required."
    }

def process_test_ticket(state: TicketState):
    """Test node logic."""
    print("\n--- Executing TEST node logic ---")
    return {
        "classification": "test",
        "resolution": "Proposed Test Action: Reset test suite cache and rerun pipelines."
    }

def human_review_node(state: TicketState):
    """HITL Gate: Pauses graph execution using interrupt() for approval."""
    print("\n--- [HITL Gate]: Pausing execution for Human Verification ---")
    
    # Pause execution and pass state information to human reviewer
    review_request = {
        "action_required": "Review ticket resolution plan",
        "proposed_resolution": state["resolution"]
    }
    
    # interrupt() saves state and halts until caller passes Command(resume=...)
    human_response = interrupt(review_request)
    
    # Check decision received from human input
    approved = human_response.get("approved", False)
    return {
        "human_approved": approved,
        "resolution": state["resolution"] if approved else "REJECTED by Human Administrator."
    }

# -------------------------------------------------------------------------
# 4. Construct Workflow
# -------------------------------------------------------------------------
workflow = StateGraph(TicketState)

# Add Nodes
workflow.add_node("dev_node", process_dev_ticket)
workflow.add_node("test_node", process_test_ticket)
workflow.add_node("tools", ToolNode(tools)) # Prebuilt LangGraph Tool Executor
workflow.add_node("human_review", human_review_node)

# Conditional initial router
workflow.add_conditional_edges(
    START,
    route_ticket,
    {
        "dev_node": "dev_node",
        "test_node": "test_node"
    }
)

# Route dev_node output to tool execution if tool_calls exist, else to human review
workflow.add_conditional_edges("dev_node", tools_condition, {"tools": "tools", END: "human_review"})
workflow.add_edge("tools", "human_review")
workflow.add_edge("test_node", "human_review")
workflow.add_edge("human_review", END)

# Compile with persistent Memory Saver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -------------------------------------------------------------------------
# 5. Execution Test Run
# -------------------------------------------------------------------------
config = {"configurable": {"thread_id": "thread-101"}}
initial_state = {"ticket_input": "Dev database connection timed out during API execution"}

print("=== Step 1: Initial Workflow Invocation ===")
for event in app.stream(initial_state, config=config):
    print("Stream event:", event)

# Check graph snapshot status
state_snapshot = app.get_state(config)
print("\nWorkflow is now PAUSED at node:", state_snapshot.next)

print("\n=== Step 2: Human Resumes Execution ===")
# Resume paused graph by supplying the human's decision via Command(resume=...)
final_output = app.invoke(Command(resume={"approved": True}), config=config)

print(f"\nFinal Ticket Resolution: {final_output['resolution']}")
print(f"Human Approved: {final_output.get('human_approved')}")