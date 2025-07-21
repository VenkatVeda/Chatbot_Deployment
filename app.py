import os
from typing import List, Union, Optional

from dotenv import load_dotenv
import gradio as gr

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from chromadb import PersistentClient

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool
from langgraph.graph import StateGraph

load_dotenv(".env")

# ------------------------------------------------------------------------------
# Azure and Chroma Setup
# ------------------------------------------------------------------------------
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint="https://azai-uaip-sandbox-eastus-001.openai.azure.com/"
)

llm = AzureChatOpenAI(
    model='xponent-openai-gpt-4o-mini',
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version='2024-12-01-preview',
    azure_endpoint='https://azai-uaip-sandbox-eastus-001.openai.azure.com/',
    temperature=0.5
)

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "products_10_6_25"
dbclient = PersistentClient(path=CHROMA_DIR)
vectorstore = Chroma(
    client=dbclient,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

# ------------------------------------------------------------------------------
# Prompt Templates and Chains
# ------------------------------------------------------------------------------
prompt_template = PromptTemplate.from_template("""
You are a helpful and friendly shopping assistant.

Here is the conversation so far:
{chat_history}

User's question: {question}

Based on the following product context, recommend 2-3 good options.
{context}

Write a warm and engaging reply mentioning product highlights, pricing, and encouraging the user to explore.
""")

llm_chain = LLMChain(llm=llm, prompt=prompt_template)
combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='context')

question_generator_prompt = PromptTemplate.from_template("""
Given the following conversation and a follow-up question, 
rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}
Question:
{question}

Standalone Question:
""")

question_generator = LLMChain(llm=llm, prompt=question_generator_prompt)

retrieval_decision_prompt = PromptTemplate.from_template("""
You are an intelligent routing assistant for a shopping bot.

Based on the conversation history and the latest user question,
decide whether the bot should retrieve product data or just answer casually.

Chat History:
{chat_history}

User Question:
{question}

Should the bot retrieve product data? Answer "yes" or "no":
""")

retrieval_decider_chain = LLMChain(llm=llm, prompt=retrieval_decision_prompt)

# 🔐 NEW: Out-of-scope checker
out_of_scope_prompt = PromptTemplate.from_template("""
You are an assistant checking if a user message is relevant to a shopping assistant focused on bags, clothing, and products.

Here is the conversation so far:
{chat_history}

User's latest question:
{question}

Does the question relate to shopping, product info, pricing, delivery, bags, or greetings like "hi", "hello", "thank you"?

Answer ONLY "yes" or "no".
""")

out_of_scope_guard = LLMChain(llm=llm, prompt=out_of_scope_prompt)

# ------------------------------------------------------------------------------
# Memory and Retrieval Chain
# ------------------------------------------------------------------------------
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)

qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    memory=memory,
    combine_docs_chain=combine_docs_chain,
    question_generator=question_generator,
    return_source_documents=False,
    output_key='answer'
)

# ------------------------------------------------------------------------------
# LangGraph State & Nodes
# ------------------------------------------------------------------------------
class BotState(dict):
    chat_history: List[Union[HumanMessage, AIMessage]]
    question: str
    tool_results: Optional[str]
    answer: str
    should_retrieve: bool
    _skip_tool: bool 

# 🔁 route_question → decide if retrieval is needed
def route_question(state: BotState) -> BotState:
    result = retrieval_decider_chain.invoke({
        "chat_history": memory.chat_memory.messages,
        "question": state["question"]
    })
    decision = result["text"].strip().lower()
    state["should_retrieve"] = "yes" in decision
    return state

# 🛡️ check_relevance → block out-of-scope queries
def check_relevance(state: BotState) -> BotState:
    result = out_of_scope_guard.invoke({
        "chat_history": memory.chat_memory.messages,
        "question": state["question"]
    })
    decision = result["text"].strip().lower()
    if "no" in decision:
        state["answer"] = (
            "I'm here to help with shopping-related questions like products, prices, or bags. "
            "Let me know how I can assist you with that! 🛍️"
        )
        state["should_retrieve"] = False
        state["tool_results"] = None
        state["_skip_tool"] = True
    else:
        state["_skip_tool"] = False
    return state

# 🧠 call_tool_if_needed → perform retrieval or general reply
def call_tool_if_needed(state: BotState) -> BotState:
    if state.get('_skip_tool', False):
        # Don't call any LLM chain
        # Just add "user" and "AI" message to memory manually
        memory.chat_memory.add_user_message(state['question'])
        memory.chat_memory.add_ai_message(state['answer'])
        return state

    if state['should_retrieve']:
        result = qa_chain.invoke({"question": state['question']})
        state['tool_results'] = result['answer']
        state['answer'] = result['answer']
    else:
        general_prompt = PromptTemplate.from_template("""
        You are a friendly shopping assistant.
        Continue the conversation helpfully using the chat history.

        Chat history:
        {chat_history}

        User's message: {question}

        Your response:
        """)
        general_chain = LLMChain(llm=llm, prompt=general_prompt)

        result = general_chain.invoke({
            "chat_history": memory.chat_memory.messages,
            "question": state['question']
        })

        state['tool_results'] = None
        state['answer'] = result['text']

        memory.chat_memory.add_user_message(state['question'])
        memory.chat_memory.add_ai_message(result['text'])

    return state


# ✅ final step: return the state
def generate_response(state: BotState) -> BotState:
    return state

# ------------------------------------------------------------------------------
# LangGraph Definition
# ------------------------------------------------------------------------------
graph = StateGraph(BotState)
graph.add_node("route_question", route_question)
graph.add_node("check_relevance", check_relevance)
graph.add_node("call_tool_if_needed", call_tool_if_needed)
graph.add_node("generate_response", generate_response)

graph.set_entry_point("route_question")
graph.add_edge("route_question", "check_relevance")
graph.add_edge("check_relevance", "call_tool_if_needed")
graph.add_edge("call_tool_if_needed", "generate_response")
graph.set_finish_point("generate_response")

agent = graph.compile()

# ------------------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------------------
chat_history_ui = []

def gradio_chat_interface(user_input: str):
    global chat_history_ui

    if user_input.strip().lower() in ["exit", "quit"]:
        chat_history_ui.append(("user", user_input))
        chat_history_ui.append(("assistant", "Have a great day! 👋"))
        return [(chat_history_ui[i][1], chat_history_ui[i+1][1]) for i in range(0, len(chat_history_ui), 2)]

    state = {
        "chat_history": memory.chat_memory.messages,
        "question": user_input,
        "tool_results": None,
        "answer": "",
        "should_retrieve": False,
        "_skip_tool": False,
    }

    result = agent.invoke(state)
    chat_history_ui.append(("user", user_input))
    chat_history_ui.append(("assistant", result["answer"]))

    return [(chat_history_ui[i][1], chat_history_ui[i+1][1]) for i in range(0, len(chat_history_ui), 2)]

def clear_chat():
    global chat_history_ui
    chat_history_ui = []
    memory.clear()
    return []

with gr.Blocks() as demo:
    gr.Markdown("## 👜 Smart Shopping Assistant (LangGraph + Guardrails + Gradio)")
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask about products, prices, or chat casually...", show_label=False)
    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Chat")

    send_btn.click(fn=gradio_chat_interface, inputs=msg, outputs=chatbot)
    msg.submit(fn=gradio_chat_interface, inputs=msg, outputs=chatbot)
    clear_btn.click(fn=clear_chat, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()