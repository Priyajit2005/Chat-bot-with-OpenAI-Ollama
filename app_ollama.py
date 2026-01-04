from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith (optional for Ollama, but OK)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OLLAMA"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, model, temperature):
    llm = Ollama(
        model=model,
        temperature=temperature
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

st.title("Enhanced Q&A Chatbot With Ollama")

model = st.sidebar.selectbox(
    "Select an Ollama model",
    ["gemma:2b"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model, temperature)
    st.write(response)
else:
    st.write("Please provide the query")
