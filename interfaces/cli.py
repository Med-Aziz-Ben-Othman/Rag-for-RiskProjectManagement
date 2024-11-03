# CLI interaction interface
def simple_cli_interaction_interface(query_engine):
    """CLI interface for interacting with the risk manager assistant"""
    try:
        while True:
            query = input("Enter your query: ")
            if query.lower() == 'exit':
                print("Exiting the risk manager assistant.")
                break
            answer = query_engine.query(query)
            print("risk manager assistant:", answer)
    except KeyboardInterrupt:
        print("\nExiting the risk manager assistant.")