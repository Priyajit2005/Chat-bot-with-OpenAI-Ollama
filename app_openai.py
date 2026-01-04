import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

## LangSmith tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OPENAI"


## Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, model, temperature, max_tokens):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"question": question})


## UI
st.title("Enhanced Q&A Chatbot With OpenAI")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

model = st.sidebar.selectbox(
    "Select an OpenAI model",
    ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
    else:
        response = generate_response(
            user_input,
            api_key,
            model,
            temperature,
            max_tokens
        )
        st.write(response)
else:
    st.write("Please provide the query")
