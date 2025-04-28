# --- Tool Definition for LangChain ---
# We need to make this function callable as a tool by the LLM
from langchain_core.tools import tool


# --- Mock Customer Data ---
MOCK_CUSTOMER_DB = {
    "12345": {
        "account_id": "12345",
        "name": "Alice Wonderland",
        "address": "123 Rabbit Hole Lane",
        "service_plan": "FiberOptic 500Mbps",
        "modem_mac": "AA:BB:CC:00:11:22",
        "status": "Active",
        "outage": "Yes"
    },
    "67890": {
        "account_id": "67890",
        "name": "Bob The Builder",
        "address": "456 Construction Way",
        "service_plan": "Cable 100Mbps",
        "modem_mac": "DD:EE:FF:33:44:55",
        "status": "Active",
        "outage": "No"
    },
     "55555": {
        "account_id": "55555",
        "name": "Charlie Chaplin",
        "address": "789 Silent Film Ave",
        "service_plan": "DSL 50Mbps",
        "modem_mac": "GG:HH:II:66:77:88",
        "status": "Suspended (Payment)",
        "outage": "Yes"
    }
}

def get_customer_info(account_id: str) -> dict | None:
    """
    Simulates fetching customer information from a database based on account ID.
    Returns customer data dictionary or None if not found.
    """
    print(f"--- TOOL: Attempting to fetch info for Account ID: {account_id} ---")
    customer_data = MOCK_CUSTOMER_DB.get(account_id)
    if customer_data:
        print(f"--- TOOL: Found customer data: {customer_data['name']} ---")
        return customer_data
    else:
        print(f"--- TOOL: Account ID {account_id} not found. ---")
        return None


@tool
def customer_lookup_tool(account_id: str) -> str:
    """
    Looks up customer information based on their account ID.
    Returns a summary string of the customer data if found, or a 'not found' message.
    Use this tool *only* when the user provides an account ID or when you need customer details to proceed with a specific request (like billing or technical support).
    """
    print("---TOOL: Start")
    customer_data = get_customer_info(account_id)
    if customer_data:
        # Return a string summary for the LLM, we'll store the full dict separately
        return f"Successfully found customer: Name: {customer_data['name']}, Plan: {customer_data['service_plan']}, Status: {customer_data['status']}."
    else:
        return f"Customer account ID '{account_id}' not found in the system."