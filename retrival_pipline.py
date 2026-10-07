# import os

# from langchain_chroma import Chroma
# # from langchain_openai import OpenAIEmbeddings


# from langchain_community.embeddings import JinaEmbeddings
# from dotenv import load_dotenv


# from langchain_openrouter import ChatOpenRouter
# from langchain_core.messages import HumanMessage, SystemMessage

# load_dotenv()

# persistent_directory = "db/chroma_db"


# # Load embeddings and vector store
# # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# embedding_model = JinaEmbeddings(
#     jina_api_key=os.getenv("JINA_API_KEY"),
#     model_name="jina-embeddings-v5-text-small",
# )

# db = Chroma(
#     persist_directory=persistent_directory,
#     embedding_function=embedding_model,
#     collection_metadata={"hnsw:space": "cosine"}
# )

# # Search for relevant documents
# # query = "What products and services does Google offer?"
# # query = "Who founded Google?"
# # query = "Who are the founders of Microsoft?"
# # query = "What is SpaceX known for?"
# # query = "Who founded SpaceX and when was it founded?"
# query = "Who is elon musk?"

# retriever = db.as_retriever(search_kwargs={"k": 5})

# # retriever = db.as_retriever(
# #     search_type="similarity_score_threshold",
# #     search_kwargs={
# #         "k": 3,
# #         "score_threshold": 0.3,  # Only return chunks with cosine similarity >= 0.3
# #     }
# # )

# relevant_docs = retriever.invoke(query)

# print(f"User Query: {query}")
# # Display results
# print("\n--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")




# # Combine the query and the relevant document contents
# combined_input = f"""Based on the following documents, please answer this question: {query}

# Documents:
# {chr(10).join(f"- {doc.page_content}" for doc in relevant_docs)}

# Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, please say "I dont have enough information to answer this question."
# """

# # Create a ChatOpenAI model
# # model = ChatOpenAI(model="gpt-4o")


# model = ChatOpenRouter( model="openrouter/free", temperature=0, )

# # Define the messages for the model
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content=combined_input),
# ]

# # Invoke the model with the combined input
# result = model.invoke(messages)

# # Display the full result and content only
# print("\n--- Generated Response ---")
# # print("Full result:")
# # print(result)

# print("Content only:")
# print(result.content)






import os

from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings


from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv

from answergeneration import generate_answer


from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"


# Load embeddings and vector store
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

embedding_model = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v5-text-small",
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
# query = "What products and services does Google offer?"
# query = "Who founded Google?"
# query = "Who are the founders of Microsoft?"
# query = "What is SpaceX known for?"
# query = "Who founded SpaceX and when was it founded?"
query = "Who is the founder of Nvidia?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3,  # Only return chunks with cosine similarity >= 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("\n--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


answer = generate_answer(query, relevant_docs)

print("\n--- Generated Response ---")
print("Content only:")
print(answer)
