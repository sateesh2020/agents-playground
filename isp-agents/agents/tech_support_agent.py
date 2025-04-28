from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage

from constants import AgentState

class TechSupportAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def interact(self, state: AgentState) -> dict:
        """Placeholder for the agent handling technical support."""
        print("--- Calling Technical Support Agent (Placeholder) ---")
        user_info = state.get('user_info')
        message_content = None
        if not user_info:
            # Ask for the Account ID if it's missing
            message_content = "To troubleshoot your internet issue, I need to access your account details. Could you please provide your Account ID?"
        else:
            # We have the info, proceed with placeholder logic
            name = user_info.get('name', 'Customer')
            modem = user_info.get('modem_mac', 'your modem')
            message_content = f"Alright {name}, let's check the status for your modem ({modem}). (Tech Support Agent is under construction)"

        return {"messages": [AIMessage(content=message_content)]}