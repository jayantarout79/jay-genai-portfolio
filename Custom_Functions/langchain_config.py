from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

def build_langchain_rag(faiss_index_path):
    # Load FAISS index from disk
    embedding = OpenAIEmbeddings()  # or any other embedding model
    vectorstore = FAISS.load_local(faiss_index_path, embeddings=embedding)

    # Build retriever
    retriever = vectorstore.as_retriever()

    # QA chain
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain