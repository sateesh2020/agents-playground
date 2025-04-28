# from langgraph.prebuilt import ToolExecutor
# from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage

# from constants import AgentState
# from .customer_info import customer_lookup_tool

# tool_executor = ToolExecutor([customer_lookup_tool]) # Pass the list of tools

# # --- Agent Node to Execute Tools ---
# def execute_tools(state: AgentState) -> dict:
#     """Calls the appropriate tool executor based on the last AI message."""
#     print("--- Executing Tools ---")
#     messages = state['messages']
#     last_message = messages[-1]

#     # Ensure it's an AIMessage with tool_calls
#     if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
#         print("--- No tools to execute ---")
#         # This branch might indicate a logic error or simply no tool was called
#         return {} # No changes to state if no tool call

#     tool_invocation = last_message.tool_calls[0] # Assuming one tool call for now
#     tool_name = tool_invocation['name']

#     print(f"--- Request to execute tool: {tool_name} ---")

#     # We use the ToolExecutor defined above
#     if tool_name == "customer_lookup_tool": # Check if it's the tool we expect here
#         response = tool_executor.invoke(tool_invocation) # response is a ToolMessage
#         print(f"--- Tool Execution Result: {response.content} ---")
#         # The state update returns the ToolMessage to be added to the messages list
#         return {"messages": [response]}
#     else:
#          print(f"--- Warning: Attempted to execute unknown tool: {tool_name} ---")
#          # Return an error message or handle appropriately
#          error_message = ToolMessage(content=f"Error: Unknown tool '{tool_name}' requested.", tool_call_id=tool_invocation['id'])
#          return {"messages": [error_message]}
