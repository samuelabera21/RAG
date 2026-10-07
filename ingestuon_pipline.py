import os
import re
import sys
import unicodedata
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_voyageai import VoyageAIEmbeddings

from langchain_community.embeddings import JinaEmbeddings


from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")




def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""

    print(f"Loading documents from {docs_path}...")

    # Check if the docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist. "
            "Please create it and add your company files."
        )

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. "
            "Please add your company documents."
        )

    # Show the first two documents
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


def clean_documents(documents):
    """Remove invisible control characters and normalize whitespace."""

    print("Cleaning document text...")

    for document in documents:
        text = unicodedata.normalize("NFKC", document.page_content)
        text = "".join(
            character
            for character in text
            if character in "\n\t" or not unicodedata.category(character).startswith("C")
        )
        document.page_content = re.sub(r"[ \t]+", " ", text)
        document.page_content = re.sub(r"\n{3,}", "\n\n", document.page_content).strip()

    return documents



def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap."""

    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks







def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist a ChromaDB vector store from document chunks."""

    print("Creating embeddings and storing them in ChromaDB...")

    # embedding_model = OpenAIEmbeddings(
    #     model="text-embedding-3-small"
    # )


    # embedding_model = VoyageAIEmbeddings(
    #     model="voyage-4-lite",
    # )



    embedding_model = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v5-text-small",
   )

    print("--- Creating vector store ---")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vector_store


def main():
    print("Starting ingestion pipeline...")


    # Load documents
    documents = load_documents(docs_path="docs")

    # Remove unnecessary invisible characters and extra whitespace
    documents = clean_documents(documents)

    # Split documents into chunks
    chunks = split_documents(documents)

    # Create and persist the vector store
    vector_store = create_vector_store(chunks)


if __name__ == "__main__":
    main()