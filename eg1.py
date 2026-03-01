from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
llm = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.7
)


#1st prompts for detailed report 
template1 = PromptTemplate(template="write a detailed report on the {topic}",
                           input_variables=['topic'])

#2nd prompt for a 5 line summary 
template2 = PromptTemplate(template="write a 5 line summary on the {text}",
                           input_variables=['text'])

prompt1 = template1.invoke({'topic': "Black hole"})

response = llm.invoke(prompt1)

print(response.content)