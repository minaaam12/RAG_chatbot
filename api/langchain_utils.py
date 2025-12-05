from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap, RunnableLambda
from .chroma_utils import vector_store

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
output_parser = StrOutputParser()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question. If the answer is not present in the context, respond with 'This information does not exist in the documentation.'"),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def create_history_aware_retriever(llm, retriever):
    def contextualize(inputs):
        question = llm.invoke(
            contextualize_q_prompt.format_messages(**inputs)
        ).content
        return retriever.invoke(question)

    return RunnableLambda(contextualize)


def get_rag_chain(model="gpt-4o-mini", retriever=retriever, qa_prompt=qa_prompt):
    llm = ChatOpenAI(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever)
    rag_chain = (
    RunnableMap({
        "context": history_aware_retriever,
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"]
    })
    | qa_prompt
    | llm
    )
    return rag_chain