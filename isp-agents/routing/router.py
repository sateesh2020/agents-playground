import json # Import json for potential fallback parsing if needed

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import END

from constants import AgentState
from .models import routing_tools

class AgentRouter:
    
    def __init__(self, llm):
        self.llm = llm

    def route_request(self, state: AgentState) -> str:
        """
        Determines the next step (agent node) based on the conversation history using LLM tool calling.

        Returns:
            str: The name of the next node to execute (e.g., "billing_agent", "tech_support_agent", "END").
        """
        print("--- Routing Request (LLM Tool Calling) ---")

        messages = state['messages']
        user_info = state.get('user_info')
        last_message = messages[-1]
        llm_with_tools = self.llm.bind_tools(routing_tools)
        # Ensure we only route based on the *user's* last message primarily,
        # but provide context. We might refine this logic later.
        # Let's provide the last few messages for better context.
        context_messages = messages[-3:] # Adjust N as needed

        # Construct a prompt for the routing LLM
        prompt = f"""Analyze the following recent conversation history for an ISP customer support bot.
        The user's identity is {'KNOWN (' + user_info['name'] + ')' if user_info else 'UNKNOWN'}.
        Determine the most appropriate next step or agent to handle the user's latest request or statement.
        Call the *single* most relevant routing tool based on the user's need.

        *** CRITICAL RULE ***: If the AI just confirmed verification or handled a previous request, and the LATEST message is the user asking a NEW question (e.g., "check for outages", "why is my bill high?", "internet is slow again"), route them accordingly.
        If the latest message is just an acknowledgement ("ok", "thanks"), use RouteToEnd or RouteToGeneralInteraction if appropriate.
        
        Available Routes:
        - RouteToBilling: Use for questions about bills, charges, payments, invoices.
        - RouteToTechSupport: Use for issues with internet speed, connectivity, modem problems, service not working.
        - RouteToOutageCheck: Use specifically when the user suspects or asks about a outage in their area, or if he is in outage.
        - RouteToGeneralInteraction: Use if the user's request is unclear, requires clarification, is a general comment, a follow-up question after a previous step, or doesn't fit other categories.
        - RouteToEnd: Use if the user indicates the conversation is finished (e.g., says "thank you", "bye", "that's all").

        Conversation History:
        {context_messages}

        Based *specifically* on the last message in the context of the conversation, which routing tool should be called? Call only one.
        **Remember the CRITICAL RULE about waiting for user input after an AI question.**
        """

        try:
            # Invoke the LLM with the routing tools bound
            # We pass the prompt as a HumanMessage to fit the expected format for invoke
            ai_msg_with_tool = llm_with_tools.invoke(prompt)
            next_node = "customer_interaction_agent"
            # Check if the LLM decided to call a tool
            if not hasattr(ai_msg_with_tool, 'tool_calls') or not ai_msg_with_tool.tool_calls:
                # Fallback if the LLM didn't call a tool (e.g., it just chatted)
                print("LLM did not call a specific routing tool. Defaulting to general interaction.")
                # Analyze the AI's response - maybe it asks a clarifying question?
                if isinstance(last_message, HumanMessage):
                    # If user was last, let general agent respond/clarify
                    next_node = "customer_interaction_agent"
                else:
                    # If AI was last and didn't route, perhaps it's a concluding remark?
                    # Let's check common closing phrases in the AI's last message
                    ai_content = ai_msg_with_tool.content.lower()
                    if any(phrase in ai_content for phrase in ["anything else", "how else can i help", "need more help"]):
                        next_node = "customer_interaction_agent" # Keep conversation open
                    else:
                        # Defaulting to END if AI's last message seems conclusive but didn't call RouteToEnd
                        print("AI response seems conclusive, routing to END as fallback.")
                        next_node = END
                return {"next_node": next_node}

            # Extract the called tool information
            tool_call = ai_msg_with_tool.tool_calls[0] # We instructed it to call only one
            tool_name = tool_call['name']
            print(f"LLM recommended route: {tool_name}, Reason: {tool_call.get('args', {}).get('reason', 'N/A')}")

            # --- ADD CHECK: Override LLM if it violates the critical rule ---
            # Check if the last message was AI asking for ID, but LLM routed elsewhere
            if isinstance(last_message, AIMessage) and \
            any(phrase in last_message.content.lower() for phrase in ["account id", "account number", "verify"]) and \
            tool_name not in ["RouteToGeneralInteraction", "RouteToEnd"]: # Allow ending if user refuses, maybe?
                print(f"*** WARNING: LLM violated rule! AI asked for ID, but router chose {tool_name}. Overriding to RouteToGeneralInteraction. ***")
                tool_name = "RouteToGeneralInteraction" # Force wait

            next_node_decision = "customer_interaction_agent" # Default fallback route
            # Map tool name to graph node name
            if tool_name == "RouteToBilling":
                next_node_decision =  "billing_agent"
            elif tool_name == "RouteToTechSupport":
                next_node_decision =  "tech_support_agent"
            elif tool_name == "RouteToOutageCheck":
                next_node_decision =  "outage_agent"
            elif tool_name == "RouteToGeneralInteraction":
                # Route back to the main interaction agent
                next_node_decision =  "customer_interaction_agent"
            elif tool_name == "RouteToEnd":
                next_node_decision =  END
            else:
                # Should not happen if tools are defined correctly
                print(f"Warning: Unknown tool called: {tool_name}. Defaulting to general interaction.")
            
            # Return the decision within the state dictionary
            return {"next_node": next_node_decision}

        except Exception as e:
            print(f"Error during LLM routing: {e}")
            # Fallback in case of error
            return {"next_node": "customer_interaction_agent"} # Fallback
