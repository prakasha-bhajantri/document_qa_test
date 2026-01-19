from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

load_dotenv()

# ----------------------------------
# 1. LLMs
# ----------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Used ONLY for question rewriting (memory)
question_rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ----------------------------------
# 2. Load document
# ----------------------------------
loader = PyPDFLoader("Introduction_RAG_GenAI_Application-1.pdf")
documents = loader.load()

# ----------------------------------
# 3. Split text
# ----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "\n", "•", ".", " "]
)

chunks = splitter.split_documents(documents)

chunks = [
    c for c in chunks
    if len(c.page_content.split()) > 30
]

# ----------------------------------
# 4. Vector store
# ----------------------------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ----------------------------------
# 5. Question rewriting prompt (MEMORY)
# ----------------------------------

rewrite_prompt = ChatPromptTemplate.from_template(
    """
You are rewriting user questions for a document-based QA system.

Goal:
Turn vague follow-up questions into explicit, standalone questions
that can be used for document retrieval.

Rules:
- If the question is vague (e.g., "give examples", "explain more", "why"),
  rewrite it to explicitly state *what* the examples/explanation are about,
  using the conversation history.
- Focus on concepts explained in the document.
- Do NOT add information not present in the history.

Conversation history:
{history}

Latest user question:
{question}
"""
)

rewrite_chain = rewrite_prompt | question_rewriter_llm

# ----------------------------------
# 6. RAG prompt (NO history here)
# ----------------------------------
rag_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.
Answer using ONLY the information in the context below.
You may summarize or infer the main topic if it is clearly implied.
if the question is form of greeting, make sure to greet user with saying 'Hi, How I can help you!'
If the answer cannot be derived from the context, say "I don't know" but make sure you check context for answer.

Context:
{context}

Question:
{question}
"""
)

# ----------------------------------
# 7. RAG chain
# ----------------------------------
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | rag_prompt
    | llm
)

# ----------------------------------
# 8. Memory store
# ----------------------------------
chat_history = ChatMessageHistory()

# ----------------------------------
# 9. Chain with memory (QUESTION REWRITING)
# ----------------------------------

# last_topic = None

def rag_with_memory(input_dict):
    # global last_topic

    question = input_dict["question"]
    history = chat_history.messages

    if len(history) == 0:
        rewritten = question
    else:
        rewritten = rewrite_chain.invoke(
            {"history": history, "question": question}
        ).content

    # store topic for future vague follow-ups
    # last_topic = rewritten

    return rewritten, rag_chain.invoke({"question": rewritten})

# ----------------------------------
# 10. Chat loop
# ----------------------------------
# print("\n📄 Document QA system with memory is ready!")
# print("Type 'exit' to quit.\n")

# while True:
#     question = input("Ask a question: ")

#     if question.lower() == "exit":
#         break

#     # 1️⃣ Rewrite & answer (history excludes current question)
#     rewritten_question, response = rag_with_memory({"question": question})

#     # 2️⃣ Store conversation
#     chat_history.add_user_message(question)
#     chat_history.add_ai_message(response.content)

#     # 3️⃣ Debug: rewritten query
#     print("\n🧠 Rewritten question:", rewritten_question)

#     # # 4️⃣ Debug: retrieval
#     # docs = retriever.invoke(rewritten_question)
#     # print("\n🔍 Retrieved chunks:")
#     # for i, d in enumerate(docs):
#     #     print(f"\n--- Chunk {i+1} ---")
#     #     print(d.page_content[:300])

#     # 5️⃣ Final answer
#     print("\nAnswer:")
#     print(response.content)
#     print("-" * 50)


@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("message") # read message "hi"
    print("question --> ", question)
    rewritten_question, response = rag_with_memory({"question": question})
    print("rewritten_question --> ", rewritten_question)
    
    # Store conversation
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response.content)

    return jsonify({"reply": response.content})

if __name__ == "__main__":
    app.run(debug=False)
