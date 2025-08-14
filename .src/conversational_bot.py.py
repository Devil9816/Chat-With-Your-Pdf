from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnableSequence         NOT NEEDED SIMPLY USING | i.e. piping prompt llm together gets the job done
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.5)

# Create a simple prompt
prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
    You are a friendly chatbot. Respond nicely to the user input.

    User: {user_input}
    Bot:"""
)

# Build the RunnableSequence: prompt → llm
chain = prompt | llm

# Simple chat loop
print("Bot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    response = chain.invoke({"user_input":user_input})
    print(f"Bot: {response.content}")
