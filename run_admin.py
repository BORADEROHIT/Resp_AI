from graph import graph, State

def main():
    compiled_graph = graph.compile()
    print("\n=== Admin Review Phase ===")
    print("Review flagged queries.")
    while True:
        exit_input = input("Type 'exit' to quit or press Enter to continue: ").strip().lower()
        if exit_input == 'exit':
            print("Exiting admin phase.")
            break

        admin_state: State = {
            "user_role": "admin"
        }
        admin_result = compiled_graph.invoke(admin_state)

        response = admin_result.get("response")

        if response == "No queries pending review.":
            print("No more pending queries. Exiting admin phase.")
            break
        if admin_result.get("has_error"):
            print("Error in admin phase:", response)
            break

if __name__ == "__main__":
    main()
