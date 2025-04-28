from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage

from constants import AgentState
from tools import customer_lookup_tool, get_customer_info
from routing import routing_tools

class CustomerInteractionAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def interact(self, state: AgentState) -> dict:
        """
        Handles the primary interaction with the user.
        Determines intent and prepares for next steps.
        """
        print("--- Calling Customer Interaction Agent ---")
        current_messages = state['messages']
        user_info = state.get('user_info') # Get current user info from state
        llm_with_tools = self.llm.bind_tools([customer_lookup_tool])
        #llm_with_routing = self.llm.bind_tools(routing_tools)
        # Determine if we need to prompt for ID or if we just received a tool result
        last_message = current_messages[-1]
        prompt_for_llm = ""
        requires_llm_call = True
        # Get the node decision from the immediately preceding step (if available, set by router)
        # Note: This relies on the router setting 'next_node'. If the entry point goes directly here, it will be None.
        previous_decision = state.get('next_node')
        print(previous_decision)
        # Check if the router explicitly sent us back here to wait for the user
        # This happens if the router's output was 'customer_interaction_agent'
        # AND the last message wasn't a ToolMessage (we always process tool results).
        if previous_decision == "customer_interaction_agent" and not isinstance(last_message, ToolMessage):
            # Further check: Was the last message from the AI asking for input?
            if isinstance(last_message, AIMessage):
                ai_content_lower = last_message.content.lower()
                is_waiting_message = any(phrase in ai_content_lower for phrase in ["account id", "account number", "could you please provide", "waiting for", "need your", "clarify", "what is", "tell me"])
                if is_waiting_message:
                    print("--- Customer Interaction Agent: Router directed back to wait. Passing turn. ---")
                    # Return empty dict: NO new messages, let the loop wait for input()
                    # Clear the next_node decision so it doesn't persist incorrectly
                    return {"next_node": None}
        print("Last Message", last_message)
        # Check if the last message was a ToolMessage from our lookup tool
        if isinstance(last_message, ToolMessage) and last_message.name == 'customer_lookup_tool':
            # We just got the result of the customer lookup tool
            print("--- Processing Customer Lookup Tool Result ---")
            tool_result_content = last_message.content
            # We need the original arguments (account_id) to fetch the full data again
            # Search backwards for the AI message that made the tool call
            tool_call_msg = None
            for msg in reversed(current_messages[:-1]):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    # Find the specific tool call that matches the ToolMessage id
                    matching_call = next((tc for tc in msg.tool_calls if tc['id'] == last_message.tool_call_id), None)
                    if matching_call and matching_call['name'] == 'customer_lookup_tool':
                        tool_call_msg = matching_call
                        break

            retrieved_data = None
            ai_response = None
            if tool_call_msg:
                account_id = tool_call_msg['args'].get('account_id')
                if account_id:
                    # Re-fetch the *actual data dictionary* using the internal function
                    # We do this because the tool returns only a summary string to the LLM
                    retrieved_data = get_customer_info(account_id) # Call our internal function

            if retrieved_data:
                print(f"--- Storing User Info in State: {retrieved_data['name']} ---")
                # Prepare state update for user_info
                state_update = {"user_info": retrieved_data}
                # Now, let the LLM generate a response acknowledging the successful lookup
                prompt_for_llm = f"""You just successfully looked up the customer using their account ID. Their details have been retrieved.
                Tool Result: {tool_result_content}
                Previous messages: {current_messages}

                Acknowledge the customer by name and confirm you have their details (no need to repeat the details unless relevant). 
                Ask how you can specifically help them now that you are verified.
                """
                ai_response = self.llm.invoke(prompt_for_llm)
            else:
                # Tool failed or ID wasn't found
                print("--- Customer Lookup Failed or ID not Found ---")
                state_update = {"user_info": None} # Ensure user_info is None
                prompt_for_llm = f"""The attempt to look up the customer account ID failed.
                Tool Result: {tool_result_content}
                Previous messages: {current_messages}

                Inform the user that the account ID was not found and ask them to please provide a valid account ID to proceed, or ask if they need help finding it.
                """
                # Execute LLM call to formulate the response to the user
                ai_response = self.llm.invoke(prompt_for_llm) # Use the base LLM here, no tool needed now
            state_update["messages"] = [ai_response]
            print(f"Updated State: {state_update}")
            return state_update # Return dict containing messages and user_info


        # --- Standard interaction flow (if not handling a tool result) ---
        else:
            prompt_context = f"""You are a friendly and helpful AI customer service assistant for 'Ziply Fiber'.
            Your goal is to understand the customer's needs. If the request requires customer-specific information (billing details, technical troubleshooting, outage confirmation for their address), you MUST have their account ID.

            Current conversation history:
            {current_messages}
            """
            if user_info:
                print("Has Userinfo in state")
                prompt_for_llm = f"""{prompt_context}
                You ALREADY have the customer's information (Name: {user_info.get('name', 'N/A')}, Plan: {user_info.get('service_plan', 'N/A')}).
                Based on the latest message, understand the user's request and respond conversationally. You DO NOT need to ask for the account ID again. Determine the user's core issue (e.g., billing, tech support, outage).
                """
                # Use base LLM as no tool call expected here, just conversation
                ai_response = self.llm.invoke(prompt_for_llm)
                return {"messages": [ai_response]}
            else:
                print("No Userinfo in state")
                prompt_for_llm = f"""{prompt_context}
                You DO NOT have the customer's information yet.
                Carefully Analyze the latest user message: "{last_message.content}"

                Follow these steps IN ORDER:

                1.  **General Greeting/Statement:** If the user message is a simple greeting (like "hi", "hello"), a general statement ("my internet is slow"), or asks a question NOT requiring specific account info ("what are your hours?"), provide a brief, helpful, CONVERSATIONAL response. Do NOT ask for an ID and do NOT call any tools. Example response for "hi": "Hello! How can I help you today?" Example response for "my internet is slow": "Okay, I can help with that. To check your specific connection, I'll need your Account ID. Could you please provide it?"

                2.  **Needs Verification (No ID Provided):** If the user message asks a question that requires verification (e.g., about their specific bill, connection status, outage at their address) AND they DID NOT provide something that looks like an Account ID, ask them politely JUST for their Account ID. Example: "To help with that, I'll need your Account ID first. Could you please provide it?" Do NOT call any tools.

                3.  **Account ID Provided:** If the user message CLEARLY provides what looks like an Account ID (e.g., consists mainly of numbers like '12345', or phrases like 'my id is 67890'), then call the `customer_lookup_tool` with ONLY the extracted ID. Do NOT add conversational text before the tool call.

                Choose ONLY ONE of the above actions. Prioritize step 1, then step 2, then step 3. Be concise in your conversational responses.
                """
                # Invoke the LLM, potentially calling the customer_lookup_tool
                ai_response = llm_with_tools.invoke(prompt_for_llm)

                # If the LLM called the tool, the graph framework handles executing it next
                # We just return the AIMessage containing the tool call request.
                # If it didn't call a tool, it's just a conversational response.
                return {"messages": [ai_response], "next_node": None}