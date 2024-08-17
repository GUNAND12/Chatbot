import streamlit as st
import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, Tool, create_openai_tools_agent, create_tool_calling_agent
from langchain.chains import LLMMathChain, LLMChain
import uuid
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.base import StructuredTool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import re
import requests
from PIL import Image
from io import BytesIO

my_secret = os.environ['OPENAI_API_KEY']
groq_api_ket = os.environ['GROQ_API_KEY']

#os.environ["openai_api_key"] = ""
client = OpenAI(api_key=my_secret)
st.set_page_config(layout="wide")
# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize memory
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []




# Function to load the document using Langchain document loader
def load_document(file_path, file_type):
    if file_type == "application/pdf":
        loader = PyPDFLoader(file_path)  # pdf file
    elif file_type == "text/html":
        loader = BSHTMLLoader(file_path)  # HTML file
    elif file_type == "text/plain":
        loader = TextLoader(file_path)  # text file
    else:
        st.error("Unsopported file type!")
        return None
    return loader.load()


# Splitting the document into chunk
def split_documemt(document, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=200)
    docs = text_splitter.split_documents(document)
    return docs

def genImage(prompt):
    # Ensure the prompt is a string and not empty
    if not isinstance(prompt, str):
        prompt = prompt.get("input", "")
    if not prompt: # Check if prompt is empty
        return "Please provide a prompt to generate an image." # Return an error message

    response = client.images.generate(model="dall-e-3",
                                      prompt=prompt,
                                      size="1024x1024",
                                      quality="standard",
                                      n=1)
    url = response.data[0].url
    return url


def extract_image_url(response_text):
    # Use regex to find the image URL in the response text
    match = re.search(r'\!\[.*?\]\((.*?)\)', response_text)
    if match:
        return match.group(1)
    else:
        return None

# Option to choose LLM Model
llm_model_choice = st.sidebar.selectbox("Select LLM Model", ["GPT-4", "LLAMA3-70B", "Gemma2-9B-It"])

# Initialize model base on user choice
if llm_model_choice == "GPT-4":
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4")
elif llm_model_choice == "LLAMA3-70B":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-70b-8192")
elif llm_model_choice == "Gemma2-9B-It":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model="Gemma2-9B-It")

# add space
st.write("")
st.write("")


# File uploader for the user to upload
uploaded_files = st.sidebar.file_uploader("Upload a document",
                                          type=["txt", "pdf", "html"],
                                          accept_multiple_files=True)

# Number input for the user to specify the chunk size
chunk_size = st.sidebar.number_input("Enter chunk size",
                                     min_value=1,
                                     value=1000)

# Check if a file has been uploader
if uploaded_files:
    doc_list = []
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type

        # Create a tempory file to save the uploaded file
        with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_type.split("/")[-1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load the document using appropriate loader
        documents = load_document(tmp_file_path, file_type)

        # Split the document
        doc = split_documemt(documents, chunk_size)

        doc_list.extend(doc)

        os.remove(tmp_file_path)

    db = FAISS.from_documents(doc_list, embeddings)
    st.sidebar.write(f"Generated Embeddings")
    st.session_state.db = db

    retriever = db.as_retriever(search_type="similarity",
                                search_kwargs={"k": 2})
    retriever_tool = create_retriever_tool(
        retriever, "vectorstore_search",
        "retriver tool to retrieve relevant info from a vectorstore")

    search_tool = TavilySearchResults(max_results=2)

    image_gen_tool = StructuredTool.from_function(name="generateImage",
                                        func=genImage,
                                        description="use to generate image from generateImage tool")


    tools = [retriever_tool, image_gen_tool, search_tool]

    user_init_prompt = """
                Chat history is: {}.
                The Question is: {}.
                Go!
                """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert who answer user's question with the most relevant datascouce. You are equipped with Image generator tool , search tool and  retriever tool. Initially, use the retriever tool to search for the query answer in the vector store. If the retriever tool does not provide an answer, use alternative tools to obtain the information."
         ),
        ("user", user_init_prompt.format("{chat_history}", "{input}")),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   handle_parsing_errors=True,
                                   memory=st.session_state.conversation_memory,
                                   return_intermediate_steps=True)

    #Display the chat_history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your Query"):
        result = agent_executor.invoke({"input": prompt})
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        response_text = result["output"]
        print(response_text)
        #Extract image URL from response text
        # image_url = extract_image_url(response_text)
        # if image_url:
        #     try: 
        #         image = Image.open(BytesIO(requests.get(image_url).content))
        #         st.image(image, caption="Generated Image")
        #     except Exception as e:
        #         st.warning(f"Failed to display image from URL: {image_url}")

        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_text
        })
