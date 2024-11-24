from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain import hub
from constants import API_KEY,LANGCHAIN_API_KEY

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY = LANGCHAIN_API_KEY
APIKEY = API_KEY

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500, api_key=API_KEY)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

embeddings = OpenAIEmbeddings(api_key=APIKEY)

vector_db = Chroma(embedding_function=embeddings, collection_name="tutorial", persist_directory="./chroma.db")

retriever = ContextualCompressionRetriever(
    base_compressor= LLMChainExtractor.from_llm(llm),
    base_retriever= vector_db.as_retriever()
)

'''
prompt_template = ChatPromptTemplate.from_template("""
    Context: {context}
    Chat History: {chat_history}
    Human: {question}
    AI: Please provide a relevant answer based on context and chat history.                                       
""")

chain = ConversationalRetrievalChain.from_llm(
    llm= llm,
    retriever= retriever,
    memory= memory,
    combine_doc_chain_kwargs={"prompt": prompt_template}
)
'''

prompt_template = hub.pull("langchain-ai/retrieval-qa-chat")


chat_retriever_chain = create_history_aware_retriever(
    llm, retriever, prompt_template
)

combine_docs_chain = create_stuff_documents_chain(
    llm, prompt_template
)

chain = create_retrieval_chain(retriever, combine_docs_chain)

def chat_response(user_input):
    chat_history = [] # Collect chat history here (a sequence of messages) 
    response = chain.invoke({"input": user_input, "chat_history": chat_history})
    return response