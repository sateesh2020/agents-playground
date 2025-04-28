
def print_graph(graph):
    # You can inspect the nodes
    print("Nodes:")
    # Iterate over the values of the nodes dictionary to get the Node objects
    for node_obj in graph.nodes.values():
        print(f"- {node_obj.id}") # Now you can access the 'id' attribute of the Node object

    print("\nEdges:")
    # Iterate directly over the list of Edge objects in graph.edges
    for edge in graph.edges:
        # Edge objects seem to have 'source' and 'target' attributes which are the node IDs (strings)
        if edge.conditional:
            # For conditional edges, the condition logic and target mapping are handled internally
            # The edge object itself just knows it's conditional and points to a potential target
            print(f"- Conditional Edge: From {edge.source} to {edge.target} (Conditional)")
            # Note: The detailed conditional mapping isn't directly printed on the edge object itself
            # in the structure you provided.
        else:
            print(f"- Simple Edge: From {edge.source} to {edge.target}")


    # # You can still print entry and finish points as before
    # print("\nEntry Point:")
    # print(graph.entry_point)

    # print("\nFinish Points:")
    # print(graph.finish_points)