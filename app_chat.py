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
# 3. Graph State (Memory)
# ===============================

class GraphState(TypedDict):
    question: str
    route: str
    answer: str
    messages: List[BaseMessage]

def question_classfier(question):
    pass

    # return health if question related health related else fallback --> classifier

def router_node(state: GraphState):
    question = state["question"].lower()
    
    classifier = question_classfier(question)
    
    if classifier in ("fallback"):
        return {"route": "fallback"}
    else:
        return {"route": "healthQA"}

def fallback(state: GraphState):
    message =  "Sorry, I am not trained on this to Answer!"
    return {"answer": message}

health_qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer using the context below.
If the context partially answers the question, summarize what is relevant.
Only say "I don't know" if the context is completely unrelated.

Context:
{context}

Question:
{question}
""")


def healthQA(state: GraphState):
    # step 1 - modify prompt to include queston and history if
    # step 2 - call llm with prompt
    # get response
    return {"answer": message}

