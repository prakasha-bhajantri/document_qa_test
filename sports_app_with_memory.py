# ===============================
# 1. Imports & Setup
# ===============================

from typing import TypedDict, List
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, END

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json

from dotenv import load_dotenv

from logger_QA import logger

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    summary: str

# PROMPTS
query_classifier_prompt = ChatPromptTemplate.from_messages([
("system",
"""
You are a query classifier for a sports analytics chatbot.

Use the conversation history to understand follow-up questions.

If the current question depends on previous sports context,
classify it according to that context.

If the user question is ambiguous but previous conversation
was about sports, assume it is still about sports.

Classify into ONE category only:

- general_sports_qa
- future_predict
- sports_coach
- fall_back

Category definitions:

general_sports_qa:
Questions about sports rules, players, teams, tournaments, history.

sports_coach:
Questions about learning a sport, training, improving skills,
practice drills, coaching tips, fitness, or techniques.

future_predict:
Questions asking predictions about match outcomes or results.

fall_back:
Questions not related to sports.

IMPORTANT RULE:
If the user asks how to learn, practice, train, or improve in a sport,
ALWAYS classify it as sports_coach.

Examples:
"Teach me football" → sports_coach
"How can I improve my batting?" → sports_coach
"What is LBW rule?" → general_sports_qa
"Who will win the match?" → future_predict

Return ONLY the category name.
"""),
MessagesPlaceholder(variable_name="messages"),
("human", "{question}")
])

summary_prompt = ChatPromptTemplate.from_messages([
("system",
"""
Summarize the conversation briefly so future
questions can be answered with context.

Focus on:
- sport being discussed
- player/team/topic
- user's goal (training, analytics etc)

Keep it under 3 sentences.
"""),
("human","{messages}")
])

def call_llm(state, prompt):

    chain = prompt | llm

    response = chain.invoke({
        "question": state["question"],
        "messages": state.get("messages", []),
        "summary": state.get("summary", "")
    })

    return response.content.strip()


# Agent 0
def query_classifier_node(state: GraphState):

    logger.info("---- CLASSIFIER NODE ----")

    logger.info("MEMORY BEFORE CLASSIFICATION ---> %s", state.get("messages", []))

    classifier = call_llm(state, query_classifier_prompt)

    logger.info("CLASSIFIER RESULT ---> %s", classifier)

    return {"classifier": classifier}

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
 MessagesPlaceholder(variable_name="messages"),
("human", "{question}")
])

def general_sports_qa_node(state: GraphState):

    logger.info("---- GENERAL SPORTS AGENT ----")

    logger.info("MEMORY RECEIVED ---> %s", state.get("messages", []))

    answer = call_llm(state, general_sports_qa_prompt)

    updated_messages = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=answer)
    ]

    logger.info("UPDATED MEMORY ---> %s", updated_messages)

    logger.info("AGENT RESPONSE ---> %s", answer)

    return {
        "answer": answer,
        "messages": updated_messages
    }

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
 MessagesPlaceholder(variable_name="messages"),
("human", "{question}")
])

def future_predict_node(state:GraphState):

    logger.info("---- FUTURE PREDICT AGENT ----")

    logger.info("MEMORY RECEIVED ---> %s", state.get("messages", []))

    answer = call_llm(state, future_predict_prompt)

    updated_messages = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=answer)
    ]


    logger.info("UPDATED MEMORY ---> %s", updated_messages)

    logger.info("AGENT RESPONSE ---> %s", answer)

    return {
        "answer": answer,
        "messages": updated_messages
    }

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
 MessagesPlaceholder(variable_name="messages"),
("human", "{question}")
])

def sports_coach_node(state:GraphState):

    logger.info("---- SPORTS COACH AGENT ----")

    logger.info("MEMORY RECEIVED ---> %s", state.get("messages", []))

    answer = call_llm(state, sports_coach_prompt)

    updated_messages = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=answer)
    ]

    logger.info("UPDATED MEMORY ---> %s", updated_messages)

    logger.info("AGENT RESPONSE ---> %s", answer)

    return {
        "answer": answer,
        "messages": updated_messages
    }

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
 MessagesPlaceholder(variable_name="messages"),
("human", "{question}")
])

def fall_back_node(state:GraphState):

    logger.info("---- FALL BACK AGENT ----")

    logger.info("MEMORY RECEIVED ---> %s", state.get("messages", []))

    answer = call_llm(state, fall_back_prompt)

    updated_messages = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=answer)
    ]

    logger.info("UPDATED MEMORY ---> %s", updated_messages)

    logger.info("AGENT RESPONSE ---> %s", answer)

    return {
        "answer": answer,
        "messages": updated_messages
    }

def memory_summarizer_node(state: GraphState):

    logger.info("---- MEMORY SUMMARIZER AGENT ----")

    messages = state.get("messages", [])

    if len(messages) < 8:
        return {}

    chain = summary_prompt | llm

    conversation_text = "\n".join([m.content for m in messages])

    summary = chain.invoke({
        "messages": conversation_text
    }).content

    logger.info("SUMMARY UPDATED ---> %s", summary)

    return {
        "summary": summary,
        "messages": messages[-4:]
    }

# Router
def route(state: GraphState):

    classifier = state["classifier"]

    logger.info("ROUTING DECISION ---> %s", classifier)

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
graph.add_node("memory_summarizer", memory_summarizer_node)

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
graph.add_edge("general_sports_qa", "memory_summarizer")
graph.add_edge("future_predict", "memory_summarizer")
graph.add_edge("sports_coach", "memory_summarizer")
graph.add_edge("fall_back", "memory_summarizer")

graph.add_edge("memory_summarizer", END)

memory = MemorySaver()

# Compile
sports_graph = graph.compile(checkpointer=memory)


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "chatbot.html",
        {"request": request}
    )

@app.post("/ask")
async def ask(message: str = Form(...)):

    logger.info("USER QUESTION ---> %s", message)

    config = {
        "configurable": {
            "thread_id": "user_session"
        }
    }

    result = sports_graph.invoke(
        {"question": message},
        config=config
    )

    return {
        "reply": result.get("answer", "")
    }