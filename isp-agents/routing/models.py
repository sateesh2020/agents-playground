from pydantic import BaseModel, Field

# Define the possible routes as Pydantic models (acting as tools)
class RouteToBilling(BaseModel):
    """Routes the conversation to the Billing Agent."""
    reason: str = Field(description="Brief reason why the conversation is being routed to billing.")

class RouteToTechSupport(BaseModel):
    """Routes the conversation to the Technical Support Agent."""
    reason: str = Field(description="Brief reason why the conversation is being routed to technical support (e.g., slow internet, no connection).")

class RouteToOutageCheck(BaseModel):
    """Routes the conversation to the Outage Check Agent."""
    reason: str = Field(description="Brief reason why the conversation is being routed to outage check.")

class RouteToGeneralInteraction(BaseModel):
    """Routes the conversation back to the general Customer Interaction Agent for clarification or general chat."""
    reason: str = Field(description="Brief reason why the conversation should continue with the general interaction agent (e.g., needs clarification, general inquiry, follow-up).")

class RouteToEnd(BaseModel):
    """Ends the conversation as the user's query is resolved or they indicated completion."""
    reason: str = Field(description="Brief reason for ending the conversation (e.g., user said thank you, issue resolved).")

# Combine all routing tools into a list
routing_tools = [RouteToBilling, RouteToTechSupport, RouteToOutageCheck, RouteToGeneralInteraction, RouteToEnd]