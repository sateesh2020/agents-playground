import os
import operator
from typing import TypedDict, Annotated, List, Union
from dotenv import load_dotenv

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
# --- CHANGE 1: Import ChatGroq ---
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

# --- Environment Setup ---
load_dotenv()
# --- CHANGE 2: Get Groq API Key ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ Groq API Key not found. Please set GROQ_API_KEY in the .env file.")
    # exit()

# --- Mock Customer Data & Internal Function (Keep as is) ---
MOCK_CUSTOMER_DB = {
    "12345": {"account_id": "12345", "name": "Alice Wonderland", "address": "123 Rabbit Hole Lane", "service_plan": "FiberOptic 500Mbps", "modem_mac": "AA:BB:CC:00:11:22", "status": "Active"},
    "67890": {"account_id": "67890", "name": "Bob The Builder", "address": "456 Construction Way", "service_plan": "Cable 100Mbps", "modem_mac": "DD:EE:FF:33:44:55", "status": "Active"},
    "55555": {"account_id": "55555", "name": "Charlie Chaplin", "address": "789 Silent Film Ave", "service_plan": "DSL 50Mbps", "modem_mac": "GG:HH:II:66:77:88", "status": "Suspended (Payment)"}
}

def get_customer_info(account_id: str) -> dict | None:
    print(f"--- INTERNAL: Fetching info for Account ID: {account_id} ---")
    return MOCK_CUSTOMER_DB.get(account_id)

# --- LangChain Tool Definition (Keep as is) ---
@tool
def customer_lookup_tool(account_id: str) -> str:
    """
    Looks up customer information based on their account ID.
    Returns a summary string confirming found details or stating 'not found'.
    Use this tool ONLY when the user explicitly provides an account ID.
    """
    print(f"--- TOOL: Running customer_lookup_tool for ID: {account_id} ---")
    customer_data = get_customer_info(account_id)
    if customer_data:
        return f"Successfully found customer: Name: {customer_data.get('name', 'N/A')}, Plan: {customer_data.get('service_plan', 'N/A')}, Status: {customer_data.get('status', 'N/A')}."
    else:
        return f"Customer account ID '{account_id}' not found."

# --- LLM Initialization ---
# --- CHANGE 3: Initialize ChatGroq ---
# Select a Groq model, e.g., 'llama3-8b-8192', 'mixtral-8x7b-32768'
# Ensure the model supports tool calling (most common ones do)
chosen_model = "llama3-8b-8192"
print(f"Initializing Groq LLM with model: {chosen_model}")

# Base LLM for standard responses
llm = ChatGroq(
    model=chosen_model,
    temperature=0.2, # Lower temp for more predictable responses
    groq_api_key=GROQ_API_KEY # Pass the key explicitly
)
# LLM bound with the lookup tool
llm_with_customer_tool = llm.bind_tools([customer_lookup_tool])
print("Groq LLMs Initialized.")


# --- Agent State Definition (Keep as is) ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    user_info: dict | None

# --- Agent Node: Customer Interaction & Verification (Keep as is) ---
# The logic inside this node doesn't need to change as it interacts
# with the LLM instance via standard LangChain methods (invoke, bind_tools).
def customer_interaction_node(state: AgentState) -> dict:
    """
    Handles user interaction, asks for ID if needed, processes tool results,
    and confirms customer info. Stops after confirmation/failure or asking for ID.
    """
    print("--- Calling Customer Interaction Node ---")
    current_messages = state['messages']
    user_info = state.get('user_info')
    last_message = current_messages[-1]

    # --- Scenario 1: Process Tool Result ---
    if isinstance(last_message, ToolMessage) and last_message.name == 'customer_lookup_tool':
        print("--- Processing Customer Lookup Tool Result ---")
        tool_result_content = last_message.content
        account_id = None
        for msg in reversed(current_messages[:-1]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                 matching_call = next((tc for tc in msg.tool_calls if tc.get('id') == last_message.tool_call_id and tc['name'] == 'customer_lookup_tool'), None)
                 if matching_call:
                    account_id = matching_call['args'].get('account_id')
                    break
        retrieved_data = get_customer_info(account_id) if account_id else None

        if retrieved_data:
            print(f"--- Storing User Info in State: {retrieved_data['name']} ---")
            confirmation_prompt = f"""The customer lookup tool was just run for ID '{account_id}'.
Tool Result: {tool_result_content}
Confirm this to the user in a single sentence, stating their name and service plan. Example: "Okay Alice, I've found your account for the FiberOptic 500Mbps plan."
Do not ask further questions.
"""
            ai_response = llm.invoke(confirmation_prompt)
            return {"user_info": retrieved_data, "messages": [ai_response]}
        else:
            print("--- Customer Lookup Failed (post-tool execution) ---")
            failure_prompt = f"""The customer lookup tool was just run for ID '{account_id or 'provided'}', but it failed.
Tool Result: {tool_result_content}
Inform the user the account ID was not found and ask them to provide a valid one. Example: "Sorry, I couldn't find an account with ID '{account_id or 'provided'}'. Please provide a valid account ID."
"""
            ai_response = llm.invoke(failure_prompt)
            return {"user_info": None, "messages": [ai_response]}

    # --- Scenario 2: Handle Human Input / Standard Interaction ---
    elif isinstance(last_message, HumanMessage):
        if user_info:
            print("--- User Info already known. Confirming. ---")
            confirmation_message = f"Okay {user_info.get('name', 'there')}, I already have your details confirmed for the {user_info.get('service_plan', 'your')} plan."
            return {"messages": [AIMessage(content=confirmation_message)]}
        else:
            print("--- User Info unknown. Analyzing last message. ---")
            analysis_prompt = f"""You are an ISP customer support AI. Your current goal is ONLY to verify the customer using their Account ID.
You DO NOT have the customer's info yet.
Analyze the latest user message: "{last_message.content}"

1.  If the user explicitly provided something that looks like an Account ID (e.g., numbers, maybe letters/numbers mix), call the `customer_lookup_tool` with that ID. Only call the tool if an ID seems present.
2.  If the user asked a question that requires verification (e.g., about their bill, specific connection status) BUT DID NOT provide an ID, ask them politely JUST for their Account ID. Example: "To help with that, I'll need your Account ID first. Could you please provide it?"
3.  If the user makes a general statement or asks a question not requiring specific account info (e.g., "what are your hours?", "hi"), just give a brief, relevant reply. Example: "Hello! How can I help today?" or "Our support hours are..."

Respond conversationally OR call the tool based on the analysis. Be concise.
"""
            ai_response_or_tool_call = llm_with_customer_tool.invoke(analysis_prompt)
            return {"messages": [ai_response_or_tool_call]}

    # --- Fallback / Other Scenarios ---
    else:
         print("--- No specific action triggered (e.g., initial state or unexpected AI message). Passing turn. ---")
         return {"messages": []}


# --- ToolNode Instantiation (Keep as is) ---
tool_node = ToolNode([customer_lookup_tool])

# --- Graph Definition (Keep as is) ---
memory = SqliteSaver.from_conn_string(":memory:")
workflow = StateGraph(AgentState)
workflow.add_node("interaction", customer_interaction_node)
workflow.add_node("execute_tool", tool_node)
workflow.set_entry_point("interaction")

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1] if state['messages'] else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_tool"
    else:
        # Stop the run to wait for next human input
        return END

workflow.add_conditional_edges("interaction", should_continue, {"execute_tool": "execute_tool", END: END})
workflow.add_edge("execute_tool", "interaction")

app = workflow.compile()
print("LangGraph Compiled (Simplified Verification Agent using Groq).")


# --- Running the Graph (Keep as is) ---
def run_conversation():
    thread_id = f"isp-verify-groq-thread-{os.urandom(4).hex()}"
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- Starting Conversation (Thread: {thread_id}) ---")
    print("AI: Hello! I'm the Horizon Broadband verification assistant (using Groq). How can I help you today? If you need account specific help, please provide your Account ID.")

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

if __name__ == "__main__":
    run_conversation()