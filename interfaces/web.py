import streamlit as st

def streamlit_interaction_interface(query_engine):
    """Streamlit interface for interacting with the risk manager assistant."""
    st.title("risk manager assistant Chatbot")
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = query_engine
    query = st.text_input("Enter your query:")
    if query:
        st.subheader("User Query :")
        st.write(query)

        st.write("**Chat Bot is typing...**")
        st.subheader("risk manager assistant:")

        if query.lower() == 'exit':
            st.write("Exiting the risk manager assistant.")
        else:
            answer = st.session_state.query_engine.query(query)
            st.write(f"risk manager assistant: {answer}")
