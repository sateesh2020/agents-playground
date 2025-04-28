import os
import operator
from typing import TypedDict, Annotated, List, Union
from dotenv import load_dotenv
import json # Keep for potential future use, though less critical now

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode # Import ToolNode
from pydantic import BaseModel, Field # Keep for routing tools
from langgraph.checkpoint.memory import MemorySaver

# --- Environment Setup ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ Groq API Key not found. Please set GROQ_API_KEY in the .env file.")
# --- Mock Customer Data & Tool ---
MOCK_CUSTOMER_DB = {
    "12345": {"account_id": "12345", "name": "Alice Wonderland", "address": "123 Rabbit Hole Lane", "service_plan": "FiberOptic 500Mbps", "modem_mac": "AA:BB:CC:00:11:22", "status": "Active"},
    "67890": {"account_id": "67890", "name": "Bob The Builder", "address": "456 Construction Way", "service_plan": "Cable 100Mbps", "modem_mac": "DD:EE:FF:33:44:55", "status": "Active"},
    "55555": {"account_id": "55555", "name": "Charlie Chaplin", "address": "789 Silent Film Ave", "service_plan": "DSL 50Mbps", "modem_mac": "GG:HH:II:66:77:88", "status": "Suspended (Payment)"}
}

def get_customer_info(account_id: str) -> dict | None:
    print(f"--- INTERNAL: Fetching info for Account ID: {account_id} ---")
    customer_data = MOCK_CUSTOMER_DB.get(account_id)
    if customer_data:
        print(f"--- INTERNAL: Found customer data: {customer_data['name']} ---")
        return customer_data
    else:
        print(f"--- INTERNAL: Account ID {account_id} not found. ---")
        return None

@tool
def customer_lookup_tool(account_id: str) -> str:
    """Looks up customer information based on their account ID. Returns a summary string."""
    print(f"--- TOOL: Running customer_lookup_tool for ID: {account_id} ---")
    customer_data = get_customer_info(account_id)
    if customer_data:
        return f"Successfully found customer: Name: {customer_data['name']}, Plan: {customer_data['service_plan']}, Status: {customer_data['status']}."
    else:
        return f"Customer account ID '{account_id}' not found in the system."

# --- Routing Tools Definition (Keep as before) ---
class RouteToBilling(BaseModel):
    """Routes the conversation to the Billing Agent."""
    reason: str = Field(description="Brief reason for routing to billing.")
class RouteToTechSupport(BaseModel):
    """Routes the conversation to the Technical Support Agent."""
    reason: str = Field(description="Brief reason for routing to tech support.")
class RouteToOutageCheck(BaseModel):
    """Routes the conversation to the Outage Check Agent."""
    reason: str = Field(description="Brief reason for routing to outage check.")
class RouteToGeneralInteraction(BaseModel):
    """Routes the conversation back to the general Customer Interaction Agent."""
    reason: str = Field(description="Brief reason for continuing with general interaction.")
class RouteToEnd(BaseModel):
    """Ends the conversation."""
    reason: str = Field(description="Brief reason for ending.")

routing_tools = [RouteToBilling, RouteToTechSupport, RouteToOutageCheck, RouteToGeneralInteraction, RouteToEnd]


# --- LLM Initialization ---
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
chosen_model = "llama3-8b-8192"
print(f"Initializing Groq LLM with model: {chosen_model}")

# Base LLM for standard responses
llm = ChatGroq(
    model=chosen_model,
    temperature=0.2, # Lower temp for more predictable responses
    groq_api_key=GROQ_API_KEY # Pass the key explicitly
)
# Bind customer lookup tool for interaction agent
llm_with_customer_tool = llm.bind_tools([customer_lookup_tool])
# Bind routing tools for the router node
llm_with_routing_tools = llm.bind_tools(routing_tools)
print("Gemini LLMs Initialized with Tools.")


# --- Agent State Definition (Keep as before) ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    user_info: dict | None
    next_node: str | None # Used by the router node

# --- Agent Nodes (Functions) ---

# customer_interaction_agent remains the same as in the previous version
# It handles interacting, asking for ID, processing tool *results*, and deciding if a tool *needs* to be called
def customer_interaction_agent(state: AgentState) -> dict:
    """
    Handles interaction, asks for ID if needed, uses customer_lookup_tool request,
    and processes the ToolMessage result *after* ToolNode runs.
    Updates user_info in the state upon successful lookup.
    """
    print("--- Calling Customer Interaction Agent ---")
    current_messages = state['messages']
    user_info = state.get('user_info') # Get current user info from state

    last_message = current_messages[-1]
    prompt_for_llm = ""

    # >>> Logic to process the ToolMessage result (comes AFTER ToolNode runs) <<<
    if isinstance(last_message, ToolMessage) and last_message.name == 'customer_lookup_tool':
        print("--- Processing Customer Lookup Tool Result ---")
        tool_result_content = last_message.content
        # Find the original tool call to get the account_id argument
        tool_call_msg = None
        account_id = None
        for msg in reversed(current_messages[:-1]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                 matching_call = next((tc for tc in msg.tool_calls if tc['id'] == last_message.tool_call_id and tc['name'] == 'customer_lookup_tool'), None)
                 if matching_call:
                    account_id = matching_call['args'].get('account_id')
                    break # Found the call

        retrieved_data = None
        if account_id:
             # Re-fetch the *actual data dictionary* using the internal function
             retrieved_data = get_customer_info(account_id)

        if retrieved_data:
             print(f"--- Storing User Info in State: {retrieved_data['name']} ---")
             state_update = {"user_info": retrieved_data}
             prompt_for_llm = f"""You just successfully looked up the customer using their account ID '{account_id}'. Their details have been retrieved.
            Tool Result: {tool_result_content}
            Previous messages: {current_messages}

            Acknowledge the customer by name and confirm you have their details (no need to repeat the details unless relevant). Ask how you can specifically help them now that you are verified.
            """
        else:
            print("--- Customer Lookup Failed or ID not Found (post-tool execution) ---")
            state_update = {"user_info": None} # Ensure user_info is None
            prompt_for_llm = f"""The attempt to look up the customer account ID '{account_id or 'provided'}' failed.
            Tool Result: {tool_result_content}
            Previous messages: {current_messages}

            Inform the user that the account ID was not found or there was an issue. Ask them to please provide a valid account ID to proceed, or ask if they need help finding it.
            """
        # Use the base LLM here, no tool needed for this response
        ai_response = llm.invoke(prompt_for_llm)
        state_update["messages"] = [ai_response]
        return state_update

    # --- Standard interaction flow or request tool ---
    else:
        prompt_context = f"""You are a friendly and helpful AI customer service assistant for 'Horizon Broadband'.
        Your goal is to understand the customer's needs. If the request requires customer-specific information (billing details, technical troubleshooting, outage confirmation for their address), you MUST have their account ID.

        Current conversation history:
        {current_messages}
        """
        if user_info:
            prompt_for_llm = f"""{prompt_context}
You ALREADY have the verified customer's information (Name: {user_info.get('name', 'N/A')}).
Based on the latest message, understand the user's request and respond conversationally. You DO NOT need to ask for the account ID again. Determine the user's core issue (e.g., billing, tech support, outage) so the system can route them.
"""
            # Use base LLM as no tool call expected here, just conversation
            ai_response = llm.invoke(prompt_for_llm)
            return {"messages": [ai_response]}

        else: # No user_info yet
            prompt_for_llm = f"""{prompt_context}
You DO NOT have the customer's information yet.
Analyze the latest message:
1. If the user provided something that looks like an account ID, call the 'customer_lookup_tool' with that ID.
2. If the user asks a question that REQUIRES account information (billing, specific tech support, outage check) and DID NOT provide an ID, ask them politely for their account ID so you can proceed with verification.
3. If the user asks a general question that DOES NOT require account info, answer it directly without asking for an ID.
Respond conversationally or call the tool.
"""
            # Use the LLM bound with the customer tool
            ai_response_or_tool_call = llm_with_customer_tool.invoke(prompt_for_llm)
            # If tool call requested, graph handles execution via ToolNode next
            return {"messages": [ai_response_or_tool_call]}


# Placeholder agents remain the same
def billing_agent(state: AgentState) -> dict:
    print("--- Calling Billing Agent (Placeholder) ---")
    user_info = state.get('user_info')
    name = user_info.get('name', 'Customer') if user_info else 'Customer'
    message = AIMessage(content=f"Okay {name}, I see you're on the {user_info.get('service_plan', 'current')} plan. Let's look into that bill. (Billing Agent is under construction)")
    return {"messages": [message]}

def tech_support_agent(state: AgentState) -> dict:
    print("--- Calling Technical Support Agent (Placeholder) ---")
    user_info = state.get('user_info')
    name = user_info.get('name', 'Customer') if user_info else 'Customer'
    message = AIMessage(content=f"Alright {name}, let's check the status for your modem ({user_info.get('modem_mac', 'your modem')}). (Tech Support Agent is under construction)")
    return {"messages": [message]}

def outage_check_agent(state: AgentState) -> dict:
    print("--- Calling Outage Check Agent (Placeholder) ---")
    user_info = state.get('user_info')
    name = user_info.get('name', 'Customer') if user_info else 'Customer'
    address = user_info.get('address', 'your area') if user_info else 'your area'
    message = AIMessage(content=f"Okay {name}, I will check for reported outages near {address}. (Outage Agent is under construction)")
    return {"messages": [message]}

# Router function remains the same - uses LLM with *routing* tools
def route_request(state: AgentState) -> dict:
    """
    Determines the next step using LLM tool calling with routing tools.
    Returns the decision in the state dictionary under the key 'next_node'.
    """
    print("--- Routing Request Node ---")
    messages = state['messages']
    user_info = state.get('user_info') # Router can also use user info for context if needed

    # We need context for the router LLM
    context_messages = messages[-3:] # Use last few messages

    prompt = f"""Analyze the following recent conversation history for an ISP customer support bot.
The user's identity is {'KNOWN (' + user_info['name'] + ')' if user_info else 'UNKNOWN'}.
Determine the most appropriate next step or agent to handle the user's latest request or statement.
Call the *single* most relevant routing tool based on the user's need.

Available Routes:
- RouteToBilling: For bills, charges, payments.
- RouteToTechSupport: For internet speed, connectivity, modem issues.
- RouteToOutageCheck: For suspected service outages.
- RouteToGeneralInteraction: If unclear, needs clarification, general chat, or follow-up.
- RouteToEnd: If finished (e.g., user says thanks, bye).

Conversation History:
{context_messages}

Based *specifically* on the last message in the context of the conversation, which routing tool should be called? Call only one.
"""
    try:
        # Use the LLM bound with *routing* tools
        ai_msg_with_tool = llm_with_routing_tools.invoke(prompt)

        if not hasattr(ai_msg_with_tool, 'tool_calls') or not ai_msg_with_tool.tool_calls:
             print("LLM did not call routing tool. Defaulting route.")
             # Fallback Logic
             if isinstance(state['messages'][-1], HumanMessage):
                 next_node = "customer_interaction_agent" # Ask for clarification
             else: # AI was last (e.g., after successful verification)
                  ai_content = state['messages'][-1].content.lower()
                  # If AI asked "how can I help", loop back for user answer
                  if any(phrase in ai_content for phrase in ["how can i help", "what can i do for you", "how may i assist"]):
                      next_node = "customer_interaction_agent" # Wait for user response
                  # If user just said thanks after verification, maybe end? Or wait? Let's wait.
                  elif any(phrase in state['messages'][-2].content.lower() for phrase in ["thank", "ok", "got it"]) and isinstance(state['messages'][-2], HumanMessage):
                       next_node = "customer_interaction_agent" # Wait for next actual request
                  else:
                      next_node = END # Default to END if AI seems done

             return {"next_node": next_node}


        tool_call = ai_msg_with_tool.tool_calls[0]
        tool_name = tool_call['name']
        print(f"LLM recommended route: {tool_name}, Reason: {tool_call.get('args', {}).get('reason', 'N/A')}")

        next_node_decision = "customer_interaction_agent" # Default fallback
        if tool_name == "RouteToBilling": next_node_decision = "billing_agent"
        elif tool_name == "RouteToTechSupport": next_node_decision = "tech_support_agent"
        elif tool_name == "RouteToOutageCheck": next_node_decision = "outage_check_agent"
        elif tool_name == "RouteToGeneralInteraction": next_node_decision = "customer_interaction_agent"
        elif tool_name == "RouteToEnd": next_node_decision = END
        else: print(f"Warning: Unknown routing tool called: {tool_name}. Defaulting.")

        return {"next_node": next_node_decision}

    except Exception as e:
        print(f"Error during LLM routing: {e}")
        return {"next_node": "customer_interaction_agent"} # Fallback


# --- ToolNode Instantiation ---
# Instantiate ToolNode with the list of tools it should be able to execute
# In this case, only the customer_lookup_tool
customer_tool_node = ToolNode([customer_lookup_tool])


# --- Graph Definition ---
# memory = SqliteSaver.from_conn_string(":memory:")
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("customer_interaction_agent", customer_interaction_agent)
# Use the ToolNode instance directly for the node definition
workflow.add_node("execute_customer_tool", customer_tool_node)
workflow.add_node("route_request", route_request) # Router node
workflow.add_node("billing_agent", billing_agent) # Placeholder
workflow.add_node("tech_support_agent", tech_support_agent) # Placeholder
workflow.add_node("outage_check_agent", outage_check_agent) # Placeholder

# Define the entry point
workflow.set_entry_point("customer_interaction_agent")

# Define routing logic

# 1. After interaction agent, check if a tool was requested
workflow.add_conditional_edges(
    "customer_interaction_agent",
    # Function to check the *last* message added by customer_interaction_agent
    lambda state: "execute_customer_tool" if isinstance(state['messages'][-1], AIMessage) and state['messages'][-1].tool_calls else "route_request",
    {
        "execute_customer_tool": "execute_customer_tool", # If tool call present, go to ToolNode
        "route_request": "route_request"                # Otherwise, go to the router node
    }
)

# 2. After ToolNode executes the tool, always go back to the interaction agent to process the result
workflow.add_edge("execute_customer_tool", "customer_interaction_agent")

# 3. The router node decides the next specialist agent or end
workflow.add_conditional_edges(
    "route_request",
    lambda state: state['next_node'], # Read the decision from the state
    {
        "billing_agent": "billing_agent",
        "tech_support_agent": "tech_support_agent",
        "outage_check_agent": "outage_check_agent",
        "customer_interaction_agent": "customer_interaction_agent", # Loop back
        END: END
    }
)

# 4. Edges from specialist agents back to interaction agent for follow-up
workflow.add_edge("billing_agent", "customer_interaction_agent")
workflow.add_edge("tech_support_agent", "customer_interaction_agent")
workflow.add_edge("outage_check_agent", "customer_interaction_agent")


memory = MemorySaver()
app = workflow.compile(memory)
# Compile the graph
app = workflow.compile(checkpointer=memory)
print("LangGraph Compiled with ToolNode for Customer Lookup.")


# --- Running the Graph (Keep the run_conversation function as before) ---
def run_conversation():
    config = {"configurable": {"thread_id": "isp-thread-toolnode-1"}}
    print("\n--- Starting Conversation ---")
    # Initial greeting - add it to the state directly? Or let the agent do it?
    # Let's simulate the first interaction setting the stage.
    initial_state = {"messages": [AIMessage(content="Hello! I'm Horizon Broadband's AI assistant. To help with specific account issues, I may need your Account ID.")]}
    current_config = config

    # Optional: Load initial state if you want the AI to start with a greeting
    # try:
    #     app.update_state(current_config, initial_state)
    #     print(f"AI: {initial_state['messages'][0].content}")
    # except Exception as e:
    #      print(f"Error setting initial state: {e}")
    #      # Fallback if state update isn't supported or fails
    #      print("AI: Hello! I'm Horizon Broadband's AI assistant. How can I help you today?")


    print("AI: Hello! I'm Horizon Broadband's AI assistant. How can I help you today? For account issues, please provide your Account ID.")


    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("AI: Thank you for contacting Horizon Broadband. Goodbye!")
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
            traceback.print_exc() # Print detailed traceback for debugging
            break


if __name__ == "__main__":
    run_conversation()