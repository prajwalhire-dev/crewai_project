import os
import glob
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from crewai.tools import BaseTool
from typing import Any, Type # Added Type for ClassVar if needed, Any for general types

from langchain_anthropic import ChatAnthropic

from langchain_voyageai import VoyageAIEmbeddings
# --- Vector Store / RAG Specific Imports ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Ensure you have InstructorEmbedding installed: pip install InstructorEmbedding
# It's also good to ensure sentence-transformers is up-to-date: pip install -U sentence-transformers

# --------------------------------------------------------------------------
# 0. SET UP API KEYS & CONFIGURATION
# --------------------------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

# Configure API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
# Ensure SERPER_API_KEY is correctly set in your .env or directly assigned below for testing
# os.environ["SERPER_API_KEY"] = "your_actual_serper_api_key_if_not_in_env"


# --- Configuration for Document Processing and Vector Store ---
DOCUMENTS_DIR = "cricket_documents"  # Create this directory and put .txt, .pdf, .csv files in it
FAISS_INDEX_PATH = "faiss_cricket_index"
VOYAGE_EMBEDDING_MODEL_NAME = "voyage-3-large" # Example, replace with your desired VoyageAI model

# --------------------------------------------------------------------------
# 1. CONFIGURE THE LLM 
# --------------------------------------------------------------------------
try:
    claude_llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
        max_tokens=3000, 
    )
    print("Anthropic LLM (Claude 3.5 Sonnet) initialized.")
except Exception as e:
    print(f"Error initializing ChatAnthropic LLM: {e}")
    print("Please ensure your ANTHROPIC_API_KEY is set correctly and the model name is valid.")
    claude_llm = None

# --------------------------------------------------------------------------
# 2. SETUP VECTOR STORE AND EMBEDDINGS (for local RAG)
# --------------------------------------------------------------------------
vector_store = None
embedding_model = None
try:
    # MODIFIED: Using VoyageAIEmbeddings
    if VOYAGE_API_KEY: # Check if VoyageAI key is available
        print(f"Initializing VoyageAI embedding model: {VOYAGE_EMBEDDING_MODEL_NAME}...")
        embedding_model = VoyageAIEmbeddings(
            model=VOYAGE_EMBEDDING_MODEL_NAME, 
            voyage_api_key=VOYAGE_API_KEY
            # You can specify batch_size or other parameters if needed, e.g., batch_size=1
        )
        print("VoyageAI Embedding model initialized.")

        def setup_vector_store(docs_dir: str, index_path: str, embeddings):
            print(f"Attempting to set up vector store from documents in: {docs_dir}")
            if not os.path.exists(docs_dir):
                os.makedirs(docs_dir)
                print(f"Created documents directory: {docs_dir}. Please add your .txt, .pdf, or .csv files there.")
                return None

            if os.path.exists(index_path):
                print(f"Attempting to load existing FAISS index from: {index_path}")
                try:
                    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                    print("FAISS index loaded successfully.")
                    return vs
                except Exception as e:
                    print(f"Error loading existing FAISS index: {e}. Will attempt to re-create index.")

            print("No existing FAISS index found or error in loading. Creating new index...")
            all_doc_chunks = []
            
            txt_files = glob.glob(os.path.join(docs_dir, "*.txt"))
            print(f"Found {len(txt_files)} .txt files.")
            for doc_path in txt_files:
                try:
                    loader = TextLoader(doc_path, encoding='utf-8')
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    all_doc_chunks.extend(text_splitter.split_documents(docs))
                except Exception as e:
                    print(f"Error loading/splitting .txt file {doc_path}: {e}")
            
            pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
            print(f"Found {len(pdf_files)} .pdf files.")
            for doc_path in pdf_files:
                try:
                    loader = PyPDFLoader(doc_path)
                    docs = loader.load() 
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) 
                    all_doc_chunks.extend(text_splitter.split_documents(docs))
                except Exception as e:
                    print(f"Error loading/splitting .pdf file {doc_path}: {e}")

            csv_files = glob.glob(os.path.join(docs_dir, "*.csv"))
            print(f"Found {len(csv_files)} .csv files.")
            for doc_path in csv_files:
                try:
                    loader = CSVLoader(file_path=doc_path, encoding='utf-8', autodetect_encoding=True) 
                    docs = loader.load() 
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
                    all_doc_chunks.extend(text_splitter.split_documents(docs))
                except Exception as e:
                    print(f"Error loading/splitting .csv file {doc_path}: {e}")


            if not all_doc_chunks:
                print(f"No document chunks created from files in '{docs_dir}'. Cannot create vector store.")
                return None

            print(f"Loaded and split documents into {len(all_doc_chunks)} chunks.")
            print("Creating FAISS index using VoyageAI embeddings. This might take a while and incur API costs...")
            try:
                vs = FAISS.from_documents(all_doc_chunks, embeddings)
                vs.save_local(index_path)
                print(f"FAISS index created and saved to: {index_path}")
                return vs
            except Exception as e:
                print(f"Error creating FAISS index: {e}")
                return None

        if embedding_model: 
            vector_store = setup_vector_store(DOCUMENTS_DIR, FAISS_INDEX_PATH, embedding_model)
            if vector_store is None:
                print("CRITICAL: Vector store setup failed or returned None. Local document retrieval will not work.")
        else:
            print("VoyageAI Embedding model not initialized (likely missing API key or class not found). Skipping vector store setup.")
            
    elif not claude_llm: # This condition might be redundant if VOYAGE_API_KEY is the primary check for embeddings
         print("Claude LLM not initialized. Skipping vector store setup.")
    else: # If VOYAGE_API_KEY is missing
        print("VOYAGE_API_KEY not found. Skipping vector store setup with VoyageAI Embeddings.")


except ImportError as ie:
    if "VoyageAIEmbeddings" in str(ie):
        print("ERROR: Failed to import VoyageAIEmbeddings. Please ensure 'langchain-voyageai' is installed.")
        print("You might need to install/upgrade: pip install langchain-voyageai")
    else:
        print(f"ERROR during import for embeddings: {ie}")
    embedding_model = None 
    vector_store = None
except Exception as e: 
    print(f"ERROR setting up embeddings or vector store: {e}") 
    import traceback
    traceback.print_exc() 
    vector_store = None


# --------------------------------------------------------------------------
# 3. DEFINE TOOLS
# --------------------------------------------------------------------------

# --- 3a. Custom Document Retrieval Tool (for local FAISS vector store) ---
class DocumentRetrievalTool(BaseTool):
    name: str = "Local Cricket Document Search"
    description: str = "Searches the local vector database of cricket documents for relevant information."
    vs: FAISS = None
    # MODIFIED: Added type annotation for 'embeddings'
    embeddings: Any = None # Using Any for now, can be HuggingFaceInstructEmbeddings if strictly typed

    def _run(self, query: str) -> str:
        if self.vs is None:
            return "Vector store is not available. Cannot search local documents."
        try:
            docs = self.vs.similarity_search(query, k=3) # Get top 3 results
            if not docs:
                return "No relevant information found in the local document database for this query."
           
            results_str = "Found in local documents:\n"
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'N/A') # For PDFs
                row = doc.metadata.get('row', 'N/A') # For CSVs
                source_info = f"Source: {source}"
                if page != 'N/A':
                    source_info += f", Page: {page}"
                if row != 'N/A':
                     source_info += f", Row: {row}"

                results_str += f"\n--- Snippet {i+1} ({source_info}) ---\n"
                results_str += doc.page_content
                results_str += "\n-----------------------------------\n"
            return results_str
        except Exception as e:
            return f"Error searching local document database: {e}"

local_search_tool = DocumentRetrievalTool()
if vector_store and embedding_model: # Ensure both are available
    local_search_tool.vs = vector_store
    local_search_tool.embeddings = embedding_model
    print("DocumentRetrievalTool initialized with FAISS vector store.")
else:
    print("DocumentRetrievalTool not fully initialized as vector store or embeddings are missing.")


# --- 3b. SerperDevTool for general web searching ---
try:
    # Ensure SERPER_API_KEY is available in the environment for this tool
    if os.getenv("SERPER_API_KEY"):
        serper_search_tool = SerperDevTool()
        print("SerperDevTool initialized.")
    else:
        print("SERPER_API_KEY not found in environment. SerperDevTool will not be initialized.")
        serper_search_tool = None
except Exception as e:
    print(f"Error initializing SerperDevTool: {e}")
    serper_search_tool = None

# --- 3c. WebsiteSearchTool for scraping specific websites ---
website_search_tool = None
try:
    if claude_llm:
        scrape_config = {
            "llm": claude_llm,
            "embedder": None
        }
        # Ensure WebsiteSearchTool is correctly imported and no other conflicting 'website_search_tool' variable exists
        current_website_search_tool = WebsiteSearchTool(config=scrape_config)
        print("WebsiteSearchTool initialized and configured with Claude LLM and no embedder.")
        website_search_tool = current_website_search_tool # Assign to the global variable
    else:
        # Fallback if claude_llm is not available
        current_website_search_tool = WebsiteSearchTool()
        print("WebsiteSearchTool initialized (default config - may error if OpenAI API key is missing and LLM is needed).")
        website_search_tool = current_website_search_tool
except Exception as e:
    print(f"Error initializing WebsiteSearchTool: {e}")
    website_search_tool = None

# --------------------------------------------------------------------------
# 4. DEFINE AGENTS
# --------------------------------------------------------------------------
query_router_agent = None
information_retrieval_agent = None
synthesis_agent = None
evaluation_agent = None

if claude_llm:
    # --- Agent 1: Query Router ---
    query_router_agent = Agent(
        role="Query Routing Specialist",
        goal=(
            "Analyze the user's query and determine the most appropriate information source. "
            "Choose 'vector_database' if the query seems answerable from existing cricket documents "
            "(e.g., historical facts, established rules, player stats already loaded). "
            "Choose 'web_search' if the query requires very recent information, broad general knowledge "
            "not specific to loaded documents, or if local search is likely to be insufficient."
        ),
        backstory=(
            "You are an intelligent dispatcher. Your job is to efficiently route queries to the "
            "correct information retrieval channel to save time and resources."
        ),
        llm=claude_llm,
        tools=[],
        verbose=True,
        allow_delegation=False
    )
    print("QueryRouterAgent created.")

    # --- Agent 2: Information Retrieval ---
    retrieval_tools = []
    if local_search_tool and local_search_tool.vs:
        retrieval_tools.append(local_search_tool)
    if serper_search_tool:
        retrieval_tools.append(serper_search_tool)
    if website_search_tool: # Check if it was successfully initialized
        retrieval_tools.append(website_search_tool)
   
    if not retrieval_tools:
        print("WARNING: No retrieval tools available for InformationRetrievalAgent.")

    information_retrieval_agent = Agent(
        role="Multi-Source Information Retriever",
        goal=(
            "Based on the routing decision (vector_database or web_search) and the user's query, "
            "retrieve the most relevant information. If 'vector_database' was chosen, use the "
            "Local Cricket Document Search tool. If 'web_search' was chosen, use SerperDevTool for general "
            "searches and WebsiteSearchTool if a specific promising URL is identified for deeper content extraction."
        ),
        backstory=(
            "You are a master researcher, adept at using both internal knowledge bases and the vast expanse of the web. "
            "You follow instructions precisely to gather the necessary data."
        ),
        tools=retrieval_tools,
        llm=claude_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )
    print("InformationRetrievalAgent created.")

    # --- Agent 3: Synthesis Agent (formerly writer_agent) ---
    synthesis_agent = Agent(
        role='Insightful Cricket Analyst and Content Synthesizer',
        goal=(
            "Analyze the retrieved information (from local DB or web) in context of the original user query. "
            "Produce a comprehensive, well-structured, and engaging answer. "
            "If information is from multiple sources, synthesize it coherently."
        ),
        backstory=(
            "You are a respected cricket commentator and writer, known for your ability to synthesize "
            "complex information from various sources into clear and insightful narratives."
        ),
        llm=claude_llm,
        verbose=True,
        allow_delegation=False,
    )
    print("SynthesisAgent created.")

    # --- Agent 4: Response Evaluation Agent ---
    evaluation_agent = Agent(
        role="Quality Assurance Cricket Expert",
        goal=(
            "Evaluate the synthesized answer based on the original query and the retrieved context (if available). "
            "Assess for relevance to the query, factual accuracy (cross-referencing context if possible), "
            "coherence, and completeness. Provide a score (e.g., 1-10) and a brief justification for each criterion."
        ),
        backstory=(
            "You are a meticulous editor and fact-checker with deep cricket knowledge. "
            "Your role is to ensure the highest quality of information is delivered."
        ),
        llm=claude_llm,
        verbose=True,
        allow_delegation=False
    )
    print("ResponseEvaluationAgent created.")

else:
    print("LLM not initialized. Agents cannot be created.")

# --------------------------------------------------------------------------
# 5. DEFINE TASKS
# --------------------------------------------------------------------------
task_route_query = None
task_retrieve_info = None
task_synthesize_answer = None
task_evaluate_answer = None

if all([query_router_agent, information_retrieval_agent, synthesis_agent, evaluation_agent]):
    task_route_query = Task(
        description="Analyze the user's query: '{user_topic}'. Determine if the query is best answered by searching the 'vector_database' of existing cricket documents or by performing a 'web_search'. Output *only* 'vector_database' or 'web_search'.",
        expected_output="A single string: either 'vector_database' or 'web_search'.",
        agent=query_router_agent,
        async_execution=False 
    )

    # MODIFIED: Task descriptions updated to rely on agent's understanding of context
    task_retrieve_info = Task(
        description=(
            "The user's query is: '{user_topic}'. "
            "A routing decision has been made (available in the context from the previous task). "
            "Your task is to retrieve detailed information to answer the user's query based on this routing decision. "
            "If the routing decision indicates 'vector_database', use the 'Local Cricket Document Search' tool. "
            "If the routing decision indicates 'web_search', use 'SerperDevTool' for broad searches. "
            "If SerperDevTool provides a highly relevant URL, you may use 'WebsiteSearchTool' with that URL "
            "to get more specific content (e.g., by calling WebsiteSearchTool(website_url='some_url_from_serper')). "
            "Compile all relevant findings."
        ),
        expected_output="A comprehensive compilation of information relevant to the user's query, based on the chosen source (local documents or web search), guided by the routing decision from the previous step.",
        agent=information_retrieval_agent,
        context=[task_route_query], 
        async_execution=False
    )

    task_synthesize_answer = Task(
        description=(
            "The original user query was: '{user_topic}'. "
            "You have been provided with retrieved information (available in the context from the previous task). "
            "Based on the user's query and this retrieved information, formulate a clear, comprehensive, and engaging answer. "
            "If the retrieved information is lacking or insufficient to answer the query well, clearly state that in your response."
        ),
        expected_output="A well-written answer to the user's query, synthesized from the provided information. If no sufficient information was found, this should be stated.",
        agent=synthesis_agent,
        context=[task_retrieve_info], # Depends on the retrieved information
        async_execution=False
    )

    task_evaluate_answer = Task(
        description=(
            "The original user query was: '{user_topic}'. "
            "You have been provided with the information retrieved by the researcher (context from a previous task) and the synthesized answer (context from the immediately preceding task). "
            "Evaluate the synthesized answer. Provide a score (1-10, 10 being best) and brief justification for each of the following criteria: "
            "1. Relevance: How well does the answer address the original query? "
            "2. Accuracy: Based on the retrieved context, is the answer factually correct? (If context is unavailable or doesn't cover the answer, note this). "
            "3. Coherence: Is the answer well-structured and easy to understand? "
            "4. Completeness: Does the answer cover the main aspects of the query, given the information found?"
        ),
        expected_output="An evaluation report with scores and justifications for relevance, accuracy, coherence, and completeness of the synthesized answer. Format as a JSON-like structure or clear bullet points.",
        agent=evaluation_agent,
        context=[task_retrieve_info, task_synthesize_answer], 
        async_execution=False
    )
    print("Tasks created.")
else:
    print("Not all agents were initialized. Tasks cannot be created.")

# --------------------------------------------------------------------------
# 6. CREATE AND RUN THE CREW
# --------------------------------------------------------------------------
if all([task_route_query, task_retrieve_info, task_synthesize_answer, task_evaluate_answer]):
    cricket_rag_crew = Crew(
        agents=[query_router_agent, information_retrieval_agent, synthesis_agent, evaluation_agent],
        tasks=[task_route_query, task_retrieve_info, task_synthesize_answer, task_evaluate_answer],
        process=Process.sequential,
        verbose=True
    )
    print("Advanced Cricket RAG Crew created.")

    print("\n" + "="*50)
    user_cricket_topic = input("Please enter your cricket query: ")
    if not user_cricket_topic:
        user_cricket_topic = "What is the history of the Ashes series?" # Default for testing
        print(f"No query entered, defaulting to: '{user_cricket_topic}'")
    print("="*50 + "\n")
   
    print(f"🚀 Kicking off the Advanced Cricket RAG Crew for query: '{user_cricket_topic}'...")
    print("="*50 + "\n")

    try:
        inputs = {'user_topic': user_cricket_topic}
        result = cricket_rag_crew.kickoff(inputs=inputs)
       
        print("\n\n" + "="*50 + " FINAL EVALUATED RESULT " + "="*50)
        print(result)
        print("="*126)

        if task_synthesize_answer.output:
             print("\n" + "="*50 + " SYNTHESIZED ANSWER (before evaluation) " + "="*50)
             print(task_synthesize_answer.output.exported_output if hasattr(task_synthesize_answer.output, 'exported_output') else task_synthesize_answer.output)
             print("="*138)


    except Exception as e:
        print(f"\nAn error occurred during crew execution: {e}")
        print("This might be due to API key issues, tool errors, or issues with the LLM.")
        import traceback
        traceback.print_exc()
else:
    print("Crew cannot be formed due to missing tasks or agents. Please check initialization steps and API keys.")

