from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = ChatOpenRouter(
    model="openrouter/free",
    temperature=0,
)

response = model.invoke("What is RAG in one sentence?")

print(response.content)