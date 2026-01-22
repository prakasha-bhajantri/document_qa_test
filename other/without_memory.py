from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS #, Chroma, Milvus, Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------------------
# 1. LLM
# ----------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ----------------------------------
# 2. Load document
# ----------------------------------
loader = PyPDFLoader("Introduction_RAG_GenAI_Application.pdf")
documents = loader.load()

# ----------------------------------
# 3. Split text
# ----------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 1000 total, chunk 1 - 0 to 499, chunk 2 - 500 to 999 - 
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# ----------------------------------
# 4. Vector store
# ----------------------------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ----------------------------------
# 5. Prompt (NO MEMORY)
# ----------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.
Answer using ONLY the information in the context below.
You may summarize or infer the main topic if it is clearly implied.
If the answer cannot be derived from the context, say "I don't know".

Context:
{context} # searched results

Question:
{question} # question
"""
)

# ----------------------------------
# 6. RAG chain - connect each steps
# ----------------------------------
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
)

# ----------------------------------
# 7. Chat loop
# ----------------------------------
print("\n📄 Document QA system is ready!")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask a question: ")
    if question.lower() == "exit":
        break

    response = rag_chain.invoke({"question": question})
    docs = retriever.invoke(question)
    print("\n🔍 Retrieved chunks:")
    for i, d in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(d.page_content[:300])    

    print("\nAnswer:")
    print(response.content)
    print("-" * 50)