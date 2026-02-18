# ===============================
# 1. Imports & Setup
# ===============================

from typing import TypedDict, List

from flask import Flask, request, jsonify,render_template
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langgraph.graph import StateGraph, END

load_dotenv()
app = Flask(__name__)

# ===============================
# 2. LLMs
# ===============================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
question_rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===============================
# 3. Load & Index Document (RAG)
# ===============================

loader = PyPDFLoader("./data/Introduction_RAG_GenAI_Application-1.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "\n", "•", ".", " "]
)

chunks = [
    c for c in splitter.split_documents(documents)
    if len(c.page_content.split()) > 30
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ===============================
# 4. Prompts
# ===============================

rewrite_prompt = ChatPromptTemplate.from_template("""
You are rewriting user questions for a document-based QA system.

Conversation history:
{history}

Latest user question:
{question}
""")

rewrite_chain = rewrite_prompt | question_rewriter_llm

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer using the context below.
If the context partially answers the question, summarize what is relevant.
Only say "I don't know" if the context is completely unrelated.

Context:
{context}

Question:
{question}
""")

rag_chain = rag_prompt | llm

# ===============================
# 5. Graph State (Memory)
# ===============================

class GraphState(TypedDict):
    question: str
    rewritten_question: str
    context: str
    answer: str
    route: str
    messages: List[BaseMessage]

# ===============================
# 6. LangGraph Nodes
# ===============================

def router_node(state: GraphState):
    q = state["question"].lower()
    words = q.split()

    if any(w in {"hi", "hello", "hey"} for w in words):
        return {"route": "greeting"}

    return {"route": "rewrite"}


def greeting_node(state: GraphState):
    reply = "Hi, how can I help you!"

    return {
        "answer": reply,
        "messages": state["messages"] + [
            HumanMessage(content=state["question"]),
            AIMessage(content=reply)
        ]
    }


def rewrite_node(state: GraphState):
    messages = state["messages"]

    if not messages:
        rewritten = state["question"]
    else:
        rewritten = rewrite_chain.invoke({
            "history": messages,
            "question": state["question"]
        }).content

    print("rewritten---> ", rewritten)
    return {"rewritten_question": rewritten}


def retrieve_node(state: GraphState):
    docs = retriever.invoke(state["rewritten_question"])
    print("docs---> ", docs)
    if not docs:
        return {"route": "fallback"}

    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context, "route": "answer"}


def answer_node(state: GraphState):
    response = rag_chain.invoke({
        "question": state["rewritten_question"],
        "context": state["context"]
    })

    return {
        "answer": response.content,
        "messages": state["messages"] + [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content)
        ]
    }


def fallback_node(state: GraphState):
    reply = "I don't know based on the provided document."

    return {
        "answer": reply,
        "messages": state["messages"] + [
            HumanMessage(content=state["question"]),
            AIMessage(content=reply)
        ]
    }

# ===============================
# 7. Build LangGraph
# ===============================

graph = StateGraph(GraphState)

graph.add_node("router", router_node)
graph.add_node("greeting", greeting_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("fallback", fallback_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "greeting": "greeting",
        "rewrite": "rewrite"
    }
)

graph.add_edge("rewrite", "retrieve")

graph.add_conditional_edges(
    "retrieve",
    lambda state: state["route"],
    {
        "answer": "answer",
        "fallback": "fallback"
    }
)

graph.add_edge("answer", END)
graph.add_edge("greeting", END)
graph.add_edge("fallback", END)

rag_graph = graph.compile()

# ===============================
# 8. Flask API
# ===============================
@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.form.get("message")
    print("user_message ----> ", user_message)

    result = rag_graph.invoke({
        "question": user_message,
        "rewritten_question": "",
        "context": "",
        "answer": "",
        "route": "",
        "messages": []
    })

    return jsonify({"reply": result["answer"]})

# ===============================
# 9. Run App
# ===============================

if __name__ == "__main__":
    app.run(debug=False)