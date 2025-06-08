from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langmem.short_term import SummarizationNode
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableSequence
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState # AgentState is kind of parent class, having messages, and remaining_steps key
from dotenv import load_dotenv
from typing import List, Any, Literal, Dict
from PIL import Image
import base64
import streamlit as st

load_dotenv()

# LLM initialization (will be used as the brain of the agent)
LLM = ChatOpenAI(model="gpt-4.1", temperature = 0.5)

# memory initialized
memory = MemorySaver()

class Custom_State(AgentState): # custom state schema built on top of the parent AgentState
    mode: str

@tool
def context_retriever(query: str) -> str:
    """Retrieve the top-3 most relevant context passages for the given query."""

    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    faiss_index = FAISS.load_local("FAISS_VECTORSTORE", embedding_model, allow_dangerous_deserialization = True)
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3}) # top 3
    docs = retriever.invoke(query)

    # Join the page_content of each Document with separators
    return "\n---\n".join(doc.page_content for doc in docs)

@tool
def generate_response(query: str, context: str, history: List[Dict[str, str]], mode: Literal["search", "reason"]) -> str:
    """
    This tool generates final response to the user's query using chat_history and the context.

    Args:
        query:   The user's current question.
        context: The retrieved context passages.
        history: List of prior turns, each dict with "role" and "content".
        mode: "search" for retrieval style or "reason" for step-by-step analysis.

    Returns:
        A string containing the LLM's answer.
    """

    # Format history as plain text
    history_str = "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" 
                             for turn in history)
    
    llm_search = ChatOpenAI(
    model = 'gpt-3.5-turbo',
    temperature = 0.4, 
    max_completion_tokens = 2048)

    llm_reason = ChatGroq(
    model = "deepseek-r1-distill-llama-70b",
    temperature = 0.7,
    max_tokens = 2048,
    )

    search_prompt = PromptTemplate(

        input_variables=["history", "context", "query"],
        template="""
        <SYSTEM>
        You are a helpful search assistant who is factual and concise.
        Use the information provided in the <CONTEXT> block to frame your answer to <USER QUESTION>.
        Use chat history <CHAT HISTORY> if needed, in case of a follow up question.
        Do not introduce outside knowledge. 
        Don't HALLUCINATE.
        </SYSTEM>

        <CHAT HISTORY>
        {history}
        </CHAT HISTORY>

        <CONTEXT>
        {context}
        </CONTEXT>

        <USER QUESTION>
        {query}
        </USER QUESTION>
        """,
        )

    reason_prompt = PromptTemplate(

        input_variables=["history", "context", "query"],
        template="""
        <SYSTEM>
        You are a helpful reasoning assistant who has a power to reason.
        Use this power and the given <CONTEXT> to answer user's query <USER QUESTION>.
        Give step-by-step explanation, with clear explanation and the ratinal behind it.
        Use chat history <CHAT HISTORY> if needed, in case of a follow up question.
        Do not introduce outside knowledge. 
        Don't HALLUCINATE.
        </SYSTEM>

        <CHAT HISTORY>
        {history}
        </CHAT HISTORY>

        <CONTEXT>
        {context}
        </CONTEXT>

        <USER QUESTION>
        {query}
        </USER QUESTION>
        """,
        )

    search_pipeline = search_prompt | llm_search
    reason_pipeline = reason_prompt | llm_reason

    # choose pipeline
    if mode == "search":
        pipeline: RunnableSequence = search_pipeline
    else:
        pipeline: RunnableSequence = reason_pipeline

    # Invoke the pipeline
    response = pipeline.invoke({
        "history": history_str,
        "context": context,
        "query": query
    })
    return response.content


# SummarizationNode setup
summarization_node = SummarizationNode(
    model = LLM,
    token_counter = count_tokens_approximately,
    max_tokens_before_summary = 4096,
    max_tokens = 4096,
    max_summary_tokens = 1024,
    output_messages_key="input_messages",  # this is what goes into the prompt
)

# ReAct Agent architecture
agent = create_react_agent(
    model = LLM,
    state_schema = Custom_State,
    tools = [context_retriever, generate_response],
    checkpointer = memory,
    pre_model_hook = summarization_node,
    prompt="""
    <SYSTEM>
    You are EduBOT, an interactive and helpful assistant. Your job is to understand and respond to the user’s query to the best of your ability.  

    You will receive a single string that may contain:  
    • Only a user question (plain text)  
    • One or more image descriptions (paragraphs beginning with “[IMAGE: filename] …”)  
    • Both a question and image descriptions in the same string  

    Depending on the content, follow these rules:

    1. **Text‐Only Queries**  
    - Determine if this is a **general interactive** question (greeting, small talk), a **follow-up** question (building on prior conversation), or a **specific** domain question.  
        • **General**: Answer directly—do not call any tools.  
        • **Follow-up** or **Specific**:  
        1. Call `context_retriever(query)` to fetch the top 3 relevant passages.  
        2. Inspect the returned context.  
            - If it supports answering the question, call  
                `generate_response(query, context, history, mode)`  
                (mode = “search” for fact look-up, “reason” for analytical reasoning).  
            - Validate that the generated answer correctly addresses the query.  
            - If the context is insufficient or the answer is off-topic, respond:  
                “I’m sorry, but the information you need isn’t in the provided documents. How else can I help?”  

    2. **Image-Only Queries**  
    - Do not call any tools initially.  
    - You may either **summarize** the image descriptions yourself or ask the user what they would like to know.  
    - If the user follows up by referencing “[IMAGE: filename]” tags, use those tags to identify which image they mean.
    - If the user does not mention the name of the image and asked a follow-up questions regarding them, ask the user to specify the image which they are talking about.

    3. **Mixed Text + Images**  
    - Read the question first. If it implies “summarize the images,” do so directly.  
    - Otherwise, treat it like a text-only specific query: combine the image descriptions and text into one composite query, call `context_retriever(combined_query)`, then `generate_response(combined_query, context, history, mode)`.  
    - Always validate the final answer against the question and context.

    **Tools Available**  
    • `context_retriever(query: str) → str`  
    Retrieves the top 3 relevant document passages (joined with “\n---\n”).  
    • `generate_response(query: str, context: str, history: List[{role,content}], mode: "search"|"reason") → str`  
    Generates a final answer using the given context, chat history, and mode (will be passed to you).

    **Validation & Tone**  
    - Always review tool outputs for relevance before replying.  
    - Maintain a clear, professional, and helpful tone.  
    - Never hallucinate or introduce outside knowledge.  
    - If unable to answer, apologize and offer to help in another way.  
    </SYSTEM>

    """
)




