import os
from typing import TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv


'''
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY")
)
'''
# Load the environment varialbe form local .env file
load_dotenv()
# 1. Define the shared state structure
class TicketState(TypedDict):
    ticket_input: str
    classification: str
    resolution: str

# 2. Define the router function (LLM acts as decision engine)
def route_ticket(state: TicketState) -> Literal["dev_node", "test_node"]:
    # 3. define LLM using low temperation for strict deterministic classificaion
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 4. define prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an automated IT support router. Classify the incoming ticket input strictly into one of the two categories: 'test' or 'dev'."),
            ("user", "Ticket description: {input}")
        ]
    )

    # 5. Chain the prompt and invoke LLM
    chain = prompt | llm
    response = chain.invoke({"input": state["ticket_input"]})

    # 6. Normalize the output string
    result = response.content.lower()

    # 7. if/else routing
    if "test" in result:
        return "test_node"
    else:
        return "dev_node"
    
# 8. Define the exeuction nodes

def process_dev_ticket(state: TicketState):
    print("\n--- Executing DEV node logic ---\n")
    return {"resolution": "Dev team action: Please check the environment version upgrades and patch logs."}

def process_test_ticket(state: TicketState):
    print("\n--- Executing TEST node logic ---\n")
    return {"resolution": "Test team action: Please check the test suite pipeline."}

# 9. Construct automation graph workflow

workflow = StateGraph(TicketState)
# Add our execution nodes to graph topology
workflow.add_node("dev_node", process_dev_ticket)
workflow.add_node("test_node", process_test_ticket)

# Add conditional routing
workflow.add_conditional_edges(
    START,
    route_ticket,
    {
        "dev_node": "dev_node",
        "test_node": "test_node"
    }
)

# Connect the processing node directly to the termination point
workflow.add_edge("dev_node", END)
workflow.add_edge("test_node", END)

# 10. Compile the graph with in memory persistent check pointer

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Test Case 1: Dev environment issue
config = {"configurable": {"thread_id": "1"}}
initial_state_1 = {"ticket_input": "I have an issue with dev environment spinning up"}

output_1 = app.invoke(initial_state_1, config=config)
print(f"Final system resolution: {output_1["resolution"]}")

print("-" *50)

# Test case 2: Test environment issue
config2 = {"configurable": {"thread_id": "2"}}
initial_state_2 = {"ticket_input": " I have an issue with test automation suite"}

output_2 = app.invoke(initial_state_2, config=config2)
print(f"Finaly system resolution: {output_2["resolution"]}")

print("-" *50)