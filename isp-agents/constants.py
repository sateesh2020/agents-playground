from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import AnyMessage
import operator

# --- Agent State Definition ---
# This TypedDict defines the structure that will be passed between nodes in the graph.
# We'll add more fields as we build more agents.
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add] # Accumulates messages (user, AI, tool)
    user_info: dict | None # <<< This will store the dict from get_customer_info
    issue_type: str | None # e.g., 'technical', 'billing', 'outage', 'general_query'
    requires_tool_call: bool # Flag if the last AI message requested a tool
    next_node: str | None # Could be used for explicit routing control
    # Add more state variables as needed for other agents:
    # diagnostic_results: dict | None
    # outage_status: str | None
    # billing_details: dict | None