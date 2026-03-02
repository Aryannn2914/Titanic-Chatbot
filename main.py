import embedding as emb
import streamlit as st
import matplotlib.pyplot as plt

st.title("Chatbot")

question = st.chat_input("Ask question")
if question:
    st.write(f"{question}:")
    response = emb.agent_reasoning(question)    
    st.text(response)
    if plt.get_fignums():
        fig = plt.gcf()
        st.pyplot(fig)

