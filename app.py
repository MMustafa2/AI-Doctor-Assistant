import streamlit as st

from Consultation import Consultation, handle

# ---------- page ----------
st.set_page_config(page_title="AI Doctor Assistant", page_icon="🩺")
st.title("AI Doctor Assistant")
st.caption("AI draft, not a licensed doctor.")

if st.sidebar.button("New chat"):
    st.session_state.clear()
    st.rerun()

if "consult" not in st.session_state:
    st.session_state.consult = Consultation()
    st.session_state.chat = [("assistant", "Hello! How can I help you today? Please tell me about your problem.")]

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

if text := st.chat_input("Type your message..."):
    st.session_state.chat.append(("user", text))          # the chat shows the patient's ORIGINAL text
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("Doctor is thinking..."):
            try:
                reply, _ = handle(st.session_state.consult, text)   # Consultation detects the patient's language itself
            except Exception as e:                        # e.g. Groq rate limit
                reply = f"Error: {type(e).__name__}. Please wait a minute and try again.\n\n`{str(e)[:300]}`"
        st.markdown(reply)
    st.session_state.chat.append(("assistant", reply))