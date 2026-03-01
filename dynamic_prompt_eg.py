import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Groq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

st.header("Research Paper Summarizer")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Create Prompt Template
template = PromptTemplate(
    input_variables=["paper", "style", "length"],
    template="""
    Explain the research paper "{paper}" in a {style} style.
    The explanation should be {length}.
    """
)

if st.button("Summarize") and paper_input != "Select...":
    
    prompt = template.format(
        paper=paper_input,
        style=style_input,
        length=length_input
    )

    result = model.invoke(prompt)

    st.write(result.content)