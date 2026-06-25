from graph import graph, State

def main():
    compiled_graph = graph.compile()
    print("\n=== User Query Phase ===")
    print("Enter your queries. Type 'exit' to quit.")
    while True:
        q = input("Query (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() == 'exit':
            print("Exiting user phase.")
            break
        state: State = {
            "user_intent": q,
            "thread_id": "1",
            "user_role": "user"
        }
        result = compiled_graph.invoke(state)
        if result.get("alert"):
            print(result["alert"])
        elif result.get("response"):
            print("Response\n", result["response"])

if __name__ == "__main__":
    main()
