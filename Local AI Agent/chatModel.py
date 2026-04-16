from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOllama(
    model="qwen3:8b",
    temperature=0.1
)

# The following code is a simple test of the ChatOllama model. It creates a conversation with a system prompt, a human message, and an AI message. Then it invokes the model with the conversation and prints the response. Finally, it streams the response and prints each chunk as it arrives.
# conversation = [
#     SystemMessage("You are a helpful assistant for questions regarding programming"),
#     HumanMessage("What is Python?"),
#     AIMessage("Python is an interpreted programming language."),
#     HumanMessage("When was it released?")
# ]

# response = model.invoke(conversation)
# print(response)
# print(response.content)

for chunk in model.stream('What is Python? When was it released?'):
    print(chunk.content, end='', flush=True)   