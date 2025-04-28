from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage

from constants import AgentState

class BillingAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def interact(self, state: AgentState) -> dict:
        """Placeholder for the agent handling billing queries."""
        print("--- Calling Billing Agent (Placeholder) ---")
        user_info = state.get('user_info')
        message_content = None
        if not user_info:
            # Ask for the Account ID if it's missing
            message_content = "To help with your billing query, I need to verify your account. Could you please provide your Account ID?"
            # We don't need to route immediately, just ask and let the flow return to wait for user input
            # via the standard customer_interaction_agent loop.
        else:
            # We have the info, proceed with placeholder logic
            name = user_info.get('name', 'Customer')
            plan = user_info.get('service_plan', 'current')
            message_content = f"Okay {name}, I see you're on the {plan} plan. Let's look into that bill. (Billing Agent is under construction)"

        return {"messages": [AIMessage(content=message_content)]}