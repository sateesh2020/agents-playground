import os
import operator
import sqlite3
from typing import TypedDict, Annotated, List, Union
from dotenv import load_dotenv
from IPython.display import Image, display

from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver # To save conversation state
from langgraph.prebuilt import ToolNode # Import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from constants import AgentState
from agents import (
    CustomerInteractionAgent,
    BillingAgent,
    OutageAgent,
    TechSupportAgent
)
from routing import AgentRouter, routing_tools
from utils.graph_utils import print_graph
from tools import customer_lookup_tool

# --- Environment Setup ---
load_dotenv()

# Check if the API key is loaded (optional but good practice)
# if not os.getenv("GOOGLE_API_KEY"):
#     print("⚠️ Google API Key not found. Please set it in the .env file.")
    # exit() # You might want to exit if the key is essential

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ Groq API Key not found. Please set GROQ_API_KEY in the .env file.")
    # exit()

# --- LLM Initialization ---
# Use a Gemini model capable of function calling/tool use later
# gemini-pro is good for chat, gemini-1.5-pro-latest might be better for complex reasoning/tool use
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
#print("Gemini LLM Initialized.")

chosen_model = "llama3-8b-8192"
print(f"Initializing Groq LLM with model: {chosen_model}")

# Base LLM for standard responses
llm = ChatGroq(
    model=chosen_model,
    temperature=0.2, # Lower temp for more predictable responses
    groq_api_key=GROQ_API_KEY # Pass the key explicitly
)


agent_router = AgentRouter(llm)

# --- Agent Nodes (Functions) ---
# We'll define each agent as a function that operates on the AgentState.
customer_interaction_agent = CustomerInteractionAgent(llm)
billing_agent = BillingAgent(llm)
outage_agent = OutageAgent(llm)
tech_support_agent = TechSupportAgent(llm)


# --- ToolNode Instantiation ---
# Instantiate ToolNode with the list of tools it should be able to execute
# In this case, only the customer_lookup_tool
customer_tool_node = ToolNode([customer_lookup_tool])

# --- Graph Definition ---
# Use SqliteSaver for persistence - conversations can be resumed.
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn) # Use a file like "conversation_memory.sqlite" for persistence

workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("customer_interaction_agent", customer_interaction_agent.interact)
# Use the ToolNode instance directly for the node definition
workflow.add_node("execute_tools", customer_tool_node)
workflow.add_node("billing_agent", billing_agent.interact)
workflow.add_node("tech_support_agent", tech_support_agent.interact)
workflow.add_node("outage_agent", outage_agent.interact)

# Define the entry point
workflow.set_entry_point("customer_interaction_agent")

# --- Edge Logic ---
def decide_after_interaction(state: AgentState) -> str:
    """Decides where to go after the interaction node."""
    messages = state['messages']
    last_message = state['messages'][-1] if state['messages'] else None
    user_info = state.get('user_info')

    # --- Priority 1: Tool Execution ---
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Tool execution needed
        return "execute_tools"
    
    # --- Priority 2: Routing Trigger ---
    # Check if user is verified, there are at least 2 messages,
    # the user spoke last turn (msg[-2]), and the AI just responded (msg[-1]).
    if (
        user_info and
        len(messages) >= 2 and
        isinstance(messages[-2], HumanMessage) and
        isinstance(last_message, AIMessage) and
        not last_message.tool_calls # Ensure the AI didn't just call another tool
    ):
        # User is verified AND just provided input -> go to router
        ai_content_lower = last_message.content.lower()
        is_waiting_prompt = any(phrase in ai_content_lower for phrase in [
            "account id", "account number", "need your", "clarify", "what is",
            "how can i help you today?" # Also wait if AI just asked this after verify
        ])
        print(ai_content_lower)
        print(f"Is Waiting Prompt:{is_waiting_prompt}")
        if not is_waiting_prompt:
            # If the AI just gave a standard response/acknowledgement to the verified user's query
            print("--- Edge: Verified user spoke, AI acknowledged -> Routing ---")
            return "route_request"
        else:
            # If the AI *did* ask a waiting question even though user is verified (unlikely but possible)
            print(f"--- Edge: AI asked '{last_message.content[:50]}...' requiring wait, ending turn ---")
            return END
    # --- Priority 3: Explicit Waiting Conditions ---
    if isinstance(last_message, AIMessage):
        ai_content_lower = last_message.content.lower()
        # Waiting for ID (user not verified)
        if not user_info and any(phrase in ai_content_lower for phrase in ["account id", "account number", "provide", "verify"]):
             print("--- Edge: AI asked for ID, ending turn to wait ---")
             return END
        # Waiting after successful verification + "how can I help?" prompt
        if user_info and "how can i help you today?" in ai_content_lower:
             print("--- Edge: AI confirmed verification and asked how to help, ending turn to wait ---")
             return END
        # General AI response without tool call often means wait (e.g., "Hello!")
        if not last_message.tool_calls:
             print(f"--- Edge: General AI response ('{last_message.content[:50]}...'), ending turn to wait ---")
             return END
    
    # Default case / unexpected state -> Wait for user
    print("--- Edge: Defaulting to END to wait ---")
    return END

# 1. After interaction agent, check if a tool needs to be executed
workflow.add_conditional_edges(
    "customer_interaction_agent",
    # Function to check if the last message contains tool calls
    decide_after_interaction,
    {
        "execute_tools": "execute_tools", # If tool call present, go to executor
        "route_request": "route_request",  # Otherwise, proceed to routing
        END: END
    }
)

# 2. After executing a tool, always go back to the interaction agent to process the result
workflow.add_edge("execute_tools", "customer_interaction_agent")

# 3. The main routing logic (now separated) - runs *after* interaction agent if no tool was called,
# or *after* interaction agent processes the tool result.
# It uses the LLM-based router
workflow.add_node("route_request", agent_router.route_request) # Add router as explicit node

# Define conditional edges *from* the new route_request node
workflow.add_conditional_edges(
    "route_request", # Source node is now the explicit router
    lambda state: state['next_node'], # The router function *itself* should return the next node name
    {
        # Mapping: Router's output -> Target node name
        "billing_agent": "billing_agent",
        "tech_support_agent": "tech_support_agent",
        "outage_agent": "outage_agent",
        "customer_interaction_agent": "customer_interaction_agent", # Loop back for clarification/follow-up
        END: END
    }
)

# Define what happens after the specialist agents run.
# For now, they just provide a message and we can end,
# or ideally, loop back to the interaction agent to see if the user needs more help.
# Let's loop back for now to allow follow-up questions.
workflow.add_edge("billing_agent", "customer_interaction_agent")
workflow.add_edge("tech_support_agent", "customer_interaction_agent")
workflow.add_edge("outage_agent", "customer_interaction_agent")


# Compile the graph into a runnable LangChain object
# Checkpoints allow the graph to be paused and resumed
memory = MemorySaver()
app = workflow.compile(memory)
print("LangGraph Compiled.")

graph = app.get_graph()

print(graph)
print_graph(graph)


# --- Running the Graph ---
def run_conversation():
    thread_id = f"isp-verify-groq-thread-{os.urandom(4).hex()}"
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- Starting Conversation (Thread: {thread_id}) ---")
    print("AI: Hello! I'm Zoey, Ziply Fiber's AI assistant. How can I help you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("AI: Goodbye!")
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}
        try:
            response = app.invoke(inputs, config=config)
            final_ai_message = ""
            if response and response.get('messages'):
                 for msg in reversed(response['messages']):
                     if isinstance(msg, AIMessage):
                          final_ai_message = msg.content
                          break
            if final_ai_message:
                 print(f"AI: {final_ai_message}")
            else:
                 print("AI: (Waiting for next input or process ended)")

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            break


# Start the interactive conversation
if __name__ == "__main__":
    
    run_conversation()