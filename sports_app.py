# ===============================
# 1. Imports & Setup
# ===============================

from typing import TypedDict, List
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langgraph.graph import StateGraph, END

from flask import Flask, request, jsonify,render_template
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)


# 2. Config the LLM with api keys
llm = ChatOpenAI(model="gpt-4o-mini")

# ===============================
# 3. Define the Graph State
# ===============================

class GraphState(TypedDict):
    question: str
    classifier: str
    answer: str
    messages: List[BaseMessage]

# Define Prompts
query_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a query classifier for a sports analytics chatbot.

    Classify the user question into ONE of the following categories ONLY:
    - general_sports_qa
    - future_predict
    - sports_coach
    - fall_back

    Notes:
    - Return ONLY the CATEGORY name
    - No explanation needed

"""),
("human", "{question}")
])

# ===============================
# 5. Define the Agents
# ===============================
def call_llm(user_question, prompt):
    chain = prompt | llm
    response = chain.invoke(user_question)
    return response.content.strip()

# Agent 0
def query_classifier_node(state:GraphState):
    user_question = state["question"].lower()

    classifier = call_llm(user_question, query_classifier_prompt)
    print("classifier ---> ", classifier)
    return {"classifier":classifier}

# Agent 1
general_sports_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a sports expert.

    Answers general questions related to:
    - Sports knowledge
    - Sports health and fitness
    - Sports rules and regulations

    Notes:
    - Be clear, factual and concise

"""),
("human", "{question}")
])

def general_sports_qa_node(state:GraphState):
    user_question = state["question"].lower()

    answer = call_llm(user_question, general_sports_qa_prompt)

    return {"answer":answer}

# Agent 2

future_predict_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a sports analytics expert.

    Rules:
    - Only make predictions if data is provided
    - Clearly state the assumptions, reasons
    - Avoid guarantees
    - Use probabilistic language

    If data is insufficiant, say so cleary
"""),
("human", "{question}")
])

def future_predict_node(state:GraphState):
    user_question = state["question"].lower()

    answer = call_llm(user_question, future_predict_prompt)

    return {"answer":answer}

# Agent 3

sports_coach_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a professional sports coach.

    Provide:
    - Explain about the sports
    - Step-by-step guidance
    - Begineer friendly instructions
    - safety tips if required

    Keep the tone encouraging, supportive and practical
"""),
("human", "{question}")
])

def sports_coach_node(state:GraphState):
    user_question = state["question"].lower()

    answer = call_llm(user_question, sports_coach_prompt)

    return {"answer":answer}

# Agent 4
fall_back_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a guardrail agent.

    If the question is NOT related to the sports:
    - If the questions are type of Greeting - Respond to it with right greeting message and then steer back to Sports analytics
    - Politely refuse to answer
    - Say you only asnwer to the sports-related queries
    - Do NOT provide any extra information

"""),
("human", "{question}")
])

def fall_back_node(state:GraphState):
    user_question = state["question"].lower()

    answer = call_llm(user_question, fall_back_prompt)

    return {"answer":answer}

# route
def route(state:GraphState):
    classifier = state["classifier"]

    if classifier == "general_sports_qa":
        return "general_sports_qa"
    elif classifier == "future_predict":
        return "future_predict"
    elif classifier == "sports_coach":
        return "sports_coach"
    else:
        return "fall_back"                

# ===============================
# 5. Compile all agents Build LangGraph
# ===============================
# Create end points
graph = StateGraph(GraphState)

graph.add_node("query_classifier", query_classifier_node)
graph.add_node("general_sports_qa", general_sports_qa_node)
graph.add_node("future_predict", future_predict_node)
graph.add_node("sports_coach", sports_coach_node)
graph.add_node("fall_back", fall_back_node)

# set the entry point
graph.set_entry_point("query_classifier")

# create the conditional edges
graph.add_conditional_edges(
    "query_classifier", # this is 1st step where we get question category
    route, # this is where we use agent route to pass the question to right agent
    {
        "general_sports_qa": "general_sports_qa",
        "future_predict": "future_predict",
        "sports_coach": "sports_coach",
        "fall_back": "fall_back"
    }
)

# connect the edges
graph.add_edge("general_sports_qa", END)
graph.add_edge("future_predict", END)
graph.add_edge("sports_coach", END)
graph.add_edge("fall_back", END)


# Compile
sports_graph = graph.compile()


# API Endpoints
@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.form.get("message")
    result = sports_graph.invoke({
        "question": user_message,
        "messages": []
    })
    return jsonify({"reply": result["answer"]})


# ===============================
# 9. Run App
# ===============================

if __name__ == "__main__":
    app.run(debug=False)