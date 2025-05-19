import os
import glob
import pickle
import re
import sqlite3 # Added for SQLite
import pandas as pd # Added for reading CSVs into SQLite
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from crewai.tools import BaseTool

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

# Imports for HybridSearchEngine
import PyPDF2
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl

# --------------------------------------------------------------------------
# 0. NLTK and SSL Setup
# --------------------------------------------------------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# try:
# nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt', quiet=True)
# try:
# nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords', quiet=True)

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# --------------------------------------------------------------------------
# 1. SET UP API KEYS & CONFIGURATION
# --------------------------------------------------------------------------
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not ANTHROPIC_API_KEY:
    print("CRITICAL WARNING: ANTHROPIC_API_KEY not found. Claude LLM will fail.")
if not SERPER_API_KEY:
    print("WARNING: SERPER_API_KEY not found. SerperDevTool might not work.")

# --- Configuration for Document Processing and Hybrid Search ---
DOCUMENTS_DIR = "cricket_documents"  # For .txt, .pdf, general .csv files for Hybrid Search
HYBRID_INDICES_DIR = "hybrid_indices"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Configuration for IPL SQLite Database ---
IPL_CSV_DIR = "ipl_csv_data"  # Create this directory and place matches.csv and deliveries.csv here
SQLITE_DB_PATH = "ipl_data.db"
IPL_MATCHES_CSV = os.path.join(IPL_CSV_DIR, "matches.csv") # From Kaggle
IPL_DELIVERIES_CSV = os.path.join(IPL_CSV_DIR, "deliveries.csv") # From Kaggle (assuming this is the second file)


# --------------------------------------------------------------------------
# 2. HybridSearchEngine Class Definition (User Provided)
# --------------------------------------------------------------------------
class HybridSearchEngine:
    """
    A hybrid search engine that combines keyword search, BM25, and vector search using FAISS.
    """
   
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.faiss_index: Optional[FAISS] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.document_chunks: List[LangchainDocument] = []
        self.stop_words = set(stopwords.words('english'))
        print(f"HybridSearchEngine initialized with embedding model: {self.embedding_model_name}")

    def _extract_text_from_pdf_pypdf2(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                if reader.is_encrypted: # Check if PDF is encrypted
                    try:
                        reader.decrypt('') # Try decrypting with empty password
                    except Exception as decrypt_error:
                        print(f"Could not decrypt PDF {pdf_path}: {decrypt_error}. Skipping.")
                        return ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if not text:
                print(f"Warning: PyPDF2 extracted no text from {pdf_path}. The PDF might be image-based or protected.")
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path} using PyPDF2: {e}")
        return text

    def ingest_documents(self, documents: List[LangchainDocument]):
        if not documents:
            print("No documents provided for ingestion to HybridSearchEngine.")
            return

        self.document_chunks = documents
        print(f"HybridSearchEngine: Ingesting {len(self.document_chunks)} pre-processed document chunks.")
       
        self._build_faiss_index()
        self._build_bm25_index()

    def _build_faiss_index(self) -> None:
        if not self.document_chunks:
            print("HybridSearchEngine: No document chunks to build FAISS index from.")
            return
        texts = [doc.page_content for doc in self.document_chunks]
        metadatas = [doc.metadata for doc in self.document_chunks]
        try:
            self.faiss_index = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            print("HybridSearchEngine: FAISS index built successfully.")
        except Exception as e:
            print(f"HybridSearchEngine: Error building FAISS index: {e}")
            self.faiss_index = None
   
    def _build_bm25_index(self) -> None:
        if not self.document_chunks:
            print("HybridSearchEngine: No document chunks to build BM25 index from.")
            return
        tokenized_docs = []
        for doc in self.document_chunks:
            tokens = word_tokenize(doc.page_content.lower())
            filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
            tokenized_docs.append(filtered_tokens)
       
        if not any(tokenized_docs):
            print("HybridSearchEngine: Warning: All documents resulted in empty tokens for BM25. BM25 index might be ineffective.")
            self.bm25_index = BM25Okapi([])
            return

        try:
            self.bm25_index = BM25Okapi(tokenized_docs)
            print("HybridSearchEngine: BM25 index built successfully.")
        except Exception as e:
            print(f"HybridSearchEngine: Error building BM25 index: {e}")
            self.bm25_index = None
   
    def _preprocess_query(self, query: str) -> List[str]:
        tokens = word_tokenize(query.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return filtered_tokens
   
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        results = []
        keywords = self._preprocess_query(query)
        if not keywords: return []

        for i, doc_obj in enumerate(self.document_chunks):
            score = 0
            text = doc_obj.page_content.lower()
            for keyword in keywords:
                if keyword in text:
                    score += text.count(keyword)
            if score > 0:
                results.append({
                    "document": doc_obj,
                    "score": score,
                    "method": "keyword"
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
   
    def _bm25_search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.bm25_index is None:
            print("HybridSearchEngine: BM25 index not built yet")
            return []
       
        tokenized_query = self._preprocess_query(query)
        if not tokenized_query: return []

        try:
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
        except ValueError as e:
            print(f"HybridSearchEngine: BM25 Error: {e}. Likely query tokens not in BM25 vocabulary.")
            return []

        results = []
        # Ensure we don't go out of bounds if bm25_scores is shorter than document_chunks
        for i, score in enumerate(bm25_scores):
            if i < len(self.document_chunks) and score > 0:
                results.append({
                    "document": self.document_chunks[i],
                    "score": float(score),
                    "method": "bm25"
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
   
    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.faiss_index is None:
            print("HybridSearchEngine: FAISS index not built yet")
            return []
       
        docs_and_scores = self.faiss_index.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in docs_and_scores:
            similarity_score = 1.0 / (1.0 + float(score)) if score is not None else 0.0
            results.append({
                "document": doc,
                "score": similarity_score,
                "method": "vector"
            })
        return results
   
    def hybrid_search(self,
                      query: str,
                      top_k: int = 5,
                      keyword_weight: float = 0.1,
                      bm25_weight: float = 0.3,
                      vector_weight: float = 0.6) -> List[Dict]:
        if self.faiss_index is None or self.bm25_index is None:
            print("HybridSearchEngine: Indices not built yet. Cannot perform hybrid search.")
            return []
       
        total_weight = keyword_weight + bm25_weight + vector_weight
        if total_weight == 0:
            print("HybridSearchEngine: Warning: All search weights are zero.")
            return []
           
        keyword_weight /= total_weight
        bm25_weight /= total_weight
        vector_weight /= total_weight
       
        fetch_k = max(top_k * 2, 10)
        keyword_results = self._keyword_search(query, top_k=fetch_k)
        bm25_results = self._bm25_search(query, top_k=fetch_k)
        vector_results = self._vector_search(query, top_k=fetch_k)
       
        all_results: Dict[str, Dict[str, Any]] = {} # Added type hint
       
        max_k_score = max(r["score"] for r in keyword_results) if keyword_results else 1.0
        max_b_score = max(r["score"] for r in bm25_results) if bm25_results else 1.0

        def get_doc_id(doc: LangchainDocument) -> str: # Added type hint
            # Use a more robust ID if chunk_id isn't always unique across different source files
            return f"{doc.metadata.get('source', 'unknown_source')}_{doc.metadata.get('chunk_id', 'unknown_chunk')}"

        for res_list, score_type_key in [
            (keyword_results, "keyword_score"),
            (bm25_results, "bm25_score"),
            (vector_results, "vector_score")
        ]:
            for result in res_list:
                doc_obj = result["document"]
                doc_id = get_doc_id(doc_obj)

                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "document": doc_obj,
                        "keyword_score_norm": 0.0, "bm25_score_norm": 0.0, "vector_score_norm": 0.0,
                        "combined_score": 0.0,
                        "methods_contributed": [] # Track which methods found this doc
                    }
               
                current_score = result["score"]
                normalized_score = 0.0
                if score_type_key == "keyword_score":
                    normalized_score = current_score / max_k_score if max_k_score > 0 else 0.0
                elif score_type_key == "bm25_score":
                    normalized_score = current_score / max_b_score if max_b_score > 0 else 0.0
                    if normalized_score < 0: normalized_score = 0
                elif score_type_key == "vector_score":
                    normalized_score = current_score
               
                all_results[doc_id][score_type_key + "_norm"] = normalized_score
                if result["method"] not in all_results[doc_id]["methods_contributed"]:
                    all_results[doc_id]["methods_contributed"].append(result["method"])
       
        for doc_id in all_results:
            all_results[doc_id]["combined_score"] = (
                all_results[doc_id]["keyword_score_norm"] * keyword_weight +
                all_results[doc_id]["bm25_score_norm"] * bm25_weight +
                all_results[doc_id]["vector_score_norm"] * vector_weight
            )
           
        final_results_list = sorted(list(all_results.values()), key=lambda x: x["combined_score"], reverse=True)
        return final_results_list[:top_k]
   
    def save_indices(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        if self.faiss_index:
            # Langchain FAISS save_local expects folder_path and index_name
            self.faiss_index.save_local(folder_path=directory, index_name="hybrid_faiss_index")
        if self.bm25_index:
            with open(os.path.join(directory, "hybrid_bm25_index.pkl"), "wb") as f:
                pickle.dump(self.bm25_index, f)
        if self.document_chunks:
            with open(os.path.join(directory, "hybrid_document_chunks.pkl"), "wb") as f:
                pickle.dump(self.document_chunks, f)
        print(f"HybridSearchEngine: Indices saved to {directory}")
   
    def load_indices(self, directory: str) -> bool:
        faiss_index_file = os.path.join(directory, "hybrid_faiss_index.faiss")
        faiss_pkl_file = os.path.join(directory, "hybrid_faiss_index.pkl")
        bm25_file_path = os.path.join(directory, "hybrid_bm25_index.pkl")
        chunks_file_path = os.path.join(directory, "hybrid_document_chunks.pkl")

        if not (os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file) and \
                os.path.exists(bm25_file_path) and os.path.exists(chunks_file_path)):
            print("HybridSearchEngine: One or more index files not found. Cannot load indices.")
            return False
        try:
            self.faiss_index = FAISS.load_local(
                folder_path=directory,
                embeddings=self.embeddings,
                index_name="hybrid_faiss_index",
                allow_dangerous_deserialization=True
            )
            with open(bm25_file_path, "rb") as f:
                self.bm25_index = pickle.load(f)
            with open(chunks_file_path, "rb") as f:
                self.document_chunks = pickle.load(f)
            print(f"HybridSearchEngine: Indices loaded successfully from {directory}")
            return True
        except Exception as e:
            print(f"HybridSearchEngine: Error loading indices: {e}")
            self.faiss_index = None
            self.bm25_index = None
            self.document_chunks = []
            return False

# --------------------------------------------------------------------------
# 3. Global Hybrid Search Engine Instance & Setup Function
# --------------------------------------------------------------------------
hybrid_search_engine_instance = HybridSearchEngine(embedding_model_name=EMBEDDING_MODEL_NAME)

def setup_hybrid_engine_and_general_docs(engine: HybridSearchEngine, docs_dir: str, indices_dir: str):
    if engine.load_indices(indices_dir):
        print("Hybrid search engine loaded from existing indices for general documents.")
        return True

    print(f"Building new hybrid indices for general documents from {docs_dir} into {indices_dir}...")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Created documents directory: {docs_dir}. Please add general .txt, .pdf, or .csv files there.")
        return False

    processed_docs_for_hybrid_engine: List[LangchainDocument] = []
    file_types = ["*.txt", "*.pdf", "*.csv"]
    doc_type_map = {"*.txt": TextLoader, "*.pdf": PyPDFLoader, "*.csv": CSVLoader}

    for file_type in file_types:
        for doc_path in glob.glob(os.path.join(docs_dir, file_type)):
            try:
                print(f"Processing general document: {doc_path}")
                if file_type == "*.pdf": # Use HybridSearchEngine's PDF extraction for consistency
                    raw_text = engine._extract_text_from_pdf_pypdf2(doc_path)
                    if raw_text:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        chunks = text_splitter.split_text(raw_text)
                        for i, chunk_text in enumerate(chunks):
                            processed_docs_for_hybrid_engine.append(LangchainDocument(
                                page_content=chunk_text,
                                metadata={"source": doc_path, "chunk_id": f"{os.path.basename(doc_path)}_pdf_{i}"}
                            ))
                else: # For TXT and CSV
                    loader_class = doc_type_map[file_type]
                    loader_args = {'file_path': doc_path, 'encoding': 'utf-8'} if file_type == "*.csv" else {'file_path': doc_path, 'encoding': 'utf-8'}
                    if file_type == "*.csv": loader_args['autodetect_encoding'] = True
                   
                    loader = loader_class(**loader_args)
                    docs = loader.load()
                   
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    split_docs = text_splitter.split_documents(docs)
                   
                    for i, chunk_doc in enumerate(split_docs):
                        chunk_doc.metadata["source"] = doc_path
                        chunk_doc.metadata["chunk_id"] = f"{os.path.basename(doc_path)}_{file_type.split('.')[1]}_{i}" # e.g. _txt_0
                    processed_docs_for_hybrid_engine.extend(split_docs)
            except Exception as e:
                print(f"Error processing general document file {doc_path}: {e}")

    if not processed_docs_for_hybrid_engine:
        print(f"No general documents found or processed in '{docs_dir}'. Hybrid engine cannot be built for these.")
        return False # Or True if it's okay for this to be empty and rely on other sources

    print(f"Total general document chunks for HybridSearchEngine: {len(processed_docs_for_hybrid_engine)}")
    engine.ingest_documents(processed_docs_for_hybrid_engine)
    engine.save_indices(indices_dir)
    return True

# --- IPL SQLite Database Setup ---
def setup_ipl_sqlite_database(db_path: str, matches_csv_path: str, deliveries_csv_path: str):
    if os.path.exists(db_path):
        print(f"IPL SQLite database already exists at {db_path}.")
        return True # Assume it's correctly populated if it exists

    print(f"Creating IPL SQLite database at {db_path}...")
    if not (os.path.exists(matches_csv_path) and os.path.exists(deliveries_csv_path)):
        os.makedirs(IPL_CSV_DIR, exist_ok=True)
        print(f"ERROR: IPL CSV files not found. Please place '{os.path.basename(matches_csv_path)}' and '{os.path.basename(deliveries_csv_path)}' in '{IPL_CSV_DIR}'.")
        return False
   
    try:
        conn = sqlite3.connect(db_path)
        matches_df = pd.read_csv(matches_csv_path)
        deliveries_df = pd.read_csv(deliveries_csv_path)

        # Clean column names (remove spaces, special chars for SQL compatibility)
        matches_df.columns = matches_df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
        deliveries_df.columns = deliveries_df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
       
        matches_df.to_sql("matches", conn, if_exists="replace", index=False)
        deliveries_df.to_sql("deliveries", conn, if_exists="replace", index=False)
        conn.close()
        print("IPL SQLite database created and populated successfully.")
        return True
    except Exception as e:
        print(f"Error creating IPL SQLite database: {e}")
        if os.path.exists(db_path): # Clean up partial DB if creation failed
            os.remove(db_path)
        return False

# Initialize and setup the data sources
hybrid_engine_ready = setup_hybrid_engine_and_general_docs(hybrid_search_engine_instance, DOCUMENTS_DIR, HYBRID_INDICES_DIR)
ipl_db_ready = setup_ipl_sqlite_database(SQLITE_DB_PATH, IPL_MATCHES_CSV, IPL_DELIVERIES_CSV)

if not hybrid_engine_ready:
    print("CRITICAL: Hybrid Search Engine for general documents is not ready.")
if not ipl_db_ready:
    print("CRITICAL: IPL SQLite Database is not ready.")


# --------------------------------------------------------------------------
# 4. DEFINE TOOLS (LLM, Web Search, Hybrid Search, and new SQLite Tool)
# --------------------------------------------------------------------------
try:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
        max_tokens=3000,
    )
    print("Anthropic LLM (Claude 3.5 Sonnet) initialized.")
except Exception as e:
    print(f"Error initializing ChatAnthropic LLM: {e}")
    llm = None

# --- Hybrid Search Tool for CrewAI ---
class HybridSearchCrewAITool(BaseTool): # Renamed to avoid conflict
    name: str = "Local General Document Hybrid Search"
    description: str = (
        "Performs a sophisticated hybrid search on the local knowledge base of general cricket documents "
        "(TXT, PDF, general CSVs). Use this for queries about Indian players or general cricket topics "
        "when the routing decision is 'hybrid_local_search'."
    )
    search_engine_instance: HybridSearchEngine

    def _run(self, query: str) -> str:
        print(f"HybridSearchCrewAITool: Received query: '{query}'")
        if not self.search_engine_instance or \
           not self.search_engine_instance.faiss_index or \
           not self.search_engine_instance.bm25_index:
            return "Hybrid search engine for general documents is not properly initialized. Cannot search."
        try:
            results = self.search_engine_instance.hybrid_search(query, top_k=3)
            if not results:
                return "No relevant information found in the local general documents using hybrid search for this query."
           
            formatted_results = "Found in local general documents (Hybrid Search Results):\n"
            for i, res_item in enumerate(results):
                doc = res_item["document"]
                source = doc.metadata.get('source', 'Unknown source')
                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                combined_score = res_item.get('combined_score', 0.0)
               
                source_info_parts = [f"Source: {os.path.basename(source)}"] if source != 'Unknown source' else []
                if chunk_id != 'N/A':
                    source_info_parts.append(f"Chunk: {chunk_id}")
                source_info_parts.append(f"Score: {combined_score:.4f}")
                source_info_str = ", ".join(source_info_parts)
               
                formatted_results += f"\n--- Snippet {i+1} ({source_info_str}) ---\n"
                content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                formatted_results += content_preview
                formatted_results += "\n-----------------------------------\n"
            return formatted_results
        except Exception as e:
            print(f"Error during HybridSearchCrewAITool execution: {e}")
            return f"Error during hybrid search of general documents: {e}"

# --- SQLite Query Tool for CrewAI ---
class SQLiteQueryTool(BaseTool):
    name: str = "IPL Statistics Database Query"
    description: str = (
        "Executes a SQL query against the IPL statistics database (tables: 'matches', 'deliveries'). "
        "Use this for queries about IPL match details, scores, player performance in IPL, locations, etc., "
        "when the routing decision is 'sqlite_ipl_database'. Input should be a valid SQL query."
    )
    db_path: str

    def _run(self, sql_query: str) -> str:
        print(f"SQLiteQueryTool: Received SQL query: '{sql_query}'")
        if not os.path.exists(self.db_path):
            return "IPL SQLite database not found. Cannot execute query."
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            conn.close()

            if not results:
                return "SQL query executed successfully, but returned no results."
           
            # Format results for the LLM (e.g., as a string representation of a list of tuples)
            # For large results, consider summarizing or returning a limited number
            formatted_results = f"Query Results for '{sql_query}':\n"
            # Get column names
            column_names = [description[0] for description in cursor.description]
            formatted_results += ", ".join(column_names) + "\n"
            for row in results[:10]: # Limit to 10 rows for brevity in LLM context
                formatted_results += str(row) + "\n"
            if len(results) > 10:
                formatted_results += f"... and {len(results) - 10} more rows.\n"
            return formatted_results
        except sqlite3.Error as e:
            print(f"SQLite Error executing query '{sql_query}': {e}")
            return f"Error executing SQL query: {e}. Please ensure the query is valid for the tables 'matches' and 'deliveries'."
        except Exception as e:
            print(f"Unexpected error in SQLiteQueryTool: {e}")
            return f"An unexpected error occurred while querying the IPL database."

# Initialize CrewAI Tools
hybrid_search_crew_tool_instance = None
if hybrid_engine_ready:
    hybrid_search_crew_tool_instance = HybridSearchCrewAITool(search_engine_instance=hybrid_search_engine_instance)
    print("HybridSearchCrewAITool for CrewAI initialized.")
else:
    print("HybridSearchCrewAITool for CrewAI could not be initialized as the engine is not ready.")

sqlite_tool_instance = None
if ipl_db_ready:
    sqlite_tool_instance = SQLiteQueryTool(db_path=SQLITE_DB_PATH)
    print("SQLiteQueryTool for CrewAI initialized.")
else:
    print("SQLiteQueryTool for CrewAI could not be initialized as the IPL database is not ready.")

# --- SerperDevTool & WebsiteSearchTool ---
serper_search_tool_instance = None
if SERPER_API_KEY:
    try:
        serper_search_tool_instance = SerperDevTool()
        print("SerperDevTool initialized.")
    except Exception as e:
        print(f"Error initializing SerperDevTool: {e}")
else:
    print("WARNING: SERPER_API_KEY not found. SerperDevTool will not be initialized.")

website_search_tool_instance = None
try:
    if llm:
        scrape_config = {"llm": llm, "embedder": None}
        website_search_tool_instance = WebsiteSearchTool(config=scrape_config)
        print("WebsiteSearchTool initialized and configured with Claude LLM and no embedder.")
    else:
        website_search_tool_instance = WebsiteSearchTool()
        print("WebsiteSearchTool initialized (default config - LLM not available).")
except Exception as e:
    print(f"Error initializing WebsiteSearchTool: {e}")

#------Deepeval evaluation tool------

# --------------------------------------------------------------------------
# 5. DEFINE AGENTS
# --------------------------------------------------------------------------
query_router_agent = None
information_retrieval_agent = None
synthesis_agent = None
evaluation_agent = None

if llm:
    query_router_agent = Agent(
        role="Intelligent Query Router",
        goal=(
            "Analyze the user's query. Based on the query content, decide the primary data source: "
            "1. If about specific Indian players (such as Virat Kolhi, M S Dhoni, Rohith) (non-IPL stats, career, biography) or general cricket history/rules from provided documents, output 'hybrid_local_search'. "
            "2. If about IPL statistics (runs, balls, scores, specific match details, player performance *in IPL*), output 'sqlite_ipl_database'. "
            "3. For any other cricket query (recent news, information not covered by local/IPL DB), output 'web_search'."
        ),
        backstory="You are an expert dispatcher. Your primary function is to categorize the user's cricket query and determine the most suitable information retrieval strategy. Be precise in your output.",
        llm=llm, tools=[], verbose=True, allow_delegation=False
    )

    retrieval_tools_list = []
    if hybrid_search_crew_tool_instance:
        retrieval_tools_list.append(hybrid_search_crew_tool_instance)
    if sqlite_tool_instance:
        retrieval_tools_list.append(sqlite_tool_instance)
    if serper_search_tool_instance:
        retrieval_tools_list.append(serper_search_tool_instance)
    if website_search_tool_instance:
        retrieval_tools_list.append(website_search_tool_instance)

    if not retrieval_tools_list:
        print("CRITICAL WARNING: No retrieval tools available for InformationRetrievalAgent.")
   
    information_retrieval_agent = Agent(
        role="Comprehensive Information Retriever",
        goal=(
            "Based on the routing decision ('hybrid_local_search', 'sqlite_ipl_database', or 'web_search') and the user's original query, "
            "retrieve the most relevant information using the appropriate tool. "
            "If 'hybrid_local_search', use the 'Local General Document Hybrid Search' tool. "
            "If 'sqlite_ipl_database', formulate a precise SQL query for the 'IPL Statistics Database Query' tool (tables are 'matches' and 'deliveries'). "
            "If 'web_search', use 'SerperDevTool' and potentially 'WebsiteSearchTool'."
        ),
        backstory="You are a master researcher, skilled with hybrid local search, SQL database querying, and web search tools. You select and use the correct tool based on the routing directive.",
        tools=retrieval_tools_list, llm=llm, verbose=True, allow_delegation=False, max_iter=3 # Reduced max_iter for focused retrieval
    )

    synthesis_agent = Agent(
        role='Insightful Cricket Analyst and Content Synthesizer',
        goal="Analyze retrieved information to produce a comprehensive, well-structured answer to the user's query.",
        backstory="You synthesize complex information into clear, insightful narratives.",
        llm=llm, verbose=True, allow_delegation=False
    )

    evaluation_agent = Agent(
        role="Quality Assurance Cricket Expert",
        goal="Evaluate the synthesized answer for relevance, accuracy, coherence, and completeness, providing scores and justifications.",
        backstory="You ensure the highest quality of information, meticulously checking facts and clarity.",
        llm=llm, verbose=True, allow_delegation=False
    )
    print("All agents created.")
else:
    print("Main LLM not initialized. Agents cannot be created.")

# --------------------------------------------------------------------------
# 6. DEFINE TASKS
# --------------------------------------------------------------------------
task_route_query = None
task_retrieve_info = None
task_synthesize_answer = None
task_evaluate_answer = None

if all([query_router_agent, information_retrieval_agent, synthesis_agent, evaluation_agent]):
    task_route_query = Task(
        description=(
            "Analyze the user's query: '{user_topic}'. "
            "Based on the content of the query, decide the most appropriate data source. "
            "If the query is about specific Indian players (non-IPL stats, career, biography from provided documents) or general cricket history/rules from these documents, output 'hybrid_local_search'. "
            "If the query is about IPL statistics (runs, balls, scores, specific match details, player performance *in IPL*), output 'sqlite_ipl_database'. "
            "For any other cricket query (e.g., recent news, information not covered by local general documents or the IPL database), output 'web_search'. "
            "Your output MUST be one of these three exact strings: 'hybrid_local_search', 'sqlite_ipl_database', or 'web_search'."
        ),
        expected_output="A single string: 'hybrid_local_search', 'sqlite_ipl_database', or 'web_search'.",
        agent=query_router_agent,
        async_execution=False
    )

    task_retrieve_info = Task(
        description=(
            "The user's original query is: '{user_topic}'. "
            "The routing decision from the previous step, which is shared in context. "
            "Your task is to retrieve detailed information to answer the user's query. "
            "If the routing decision is 'hybrid_local_search', use the 'Local General Document Hybrid Search' tool with the original query: '{user_topic}'. "
            "If the routing decision is 'sqlite_ipl_database', you MUST formulate an appropriate SQL query based on '{user_topic}' to retrieve the relevant IPL data from the 'IPL Statistics Database Query' tool. For example, if the query is 'most runs by Kohli in IPL 2016', your SQL might be 'SELECT SUM(batsman_runs) FROM deliveries WHERE batsman = \"V Kohli\" AND id IN (SELECT id FROM matches WHERE season = 2016)'. The tables are 'matches' and 'deliveries'. "
            "If the routing decision is 'web_search', use the 'SerperDevTool' with the query '{user_topic}'. If SerperDevTool provides a highly relevant URL, you may then use 'WebsiteSearchTool' with that specific URL. "
            "Compile all relevant findings."
        ),
        expected_output="A comprehensive compilation of information relevant to the user's query, retrieved using the method specified by the routing decision. If using SQL, include the SQL query and its results.",
        agent=information_retrieval_agent,
        context=[task_route_query],
        async_execution=False
    )

    task_synthesize_answer = Task(
        description=(
            "The original user query was: '{user_topic}'. "
            "Retrieved information (which might include routing decision, SQL query used, and search results) from context. "
            "Based on the user's query and ALL the retrieved information, formulate a clear, comprehensive, and engaging answer. "
            "If the retrieved information is lacking or insufficient, clearly state that."
        ),
        expected_output="A well-written answer to the user's query, synthesized from the provided information. If no sufficient information was found, this should be stated.",
        agent=synthesis_agent,
        context=[task_retrieve_info], # Depends on the retrieved information
        async_execution=False
    )

    task_evaluate_answer = Task(
        description=(
            "The original user query was: '{user_topic}'. "
            "Retrieved context (from task_retrieve_info), which you can get it from the context. "
            "Synthesized answer (from task_synthesize_answer):, which you can get it from the context. "
            "Evaluate the synthesized answer. Provide a score (1-10, 10 being best) and brief justification for each of the following criteria: "
            "1. Relevance: How well does the answer address the original query? "
            "2. Accuracy: Based on the retrieved context, is the answer factually correct? (If context is unavailable or doesn't cover the answer, note this). "
            "3. Coherence: Is the answer well-structured and easy to understand? "
            "4. Completeness: Does the answer cover the main aspects of the query, given the information found?"
        ),
        expected_output="An evaluation report with scores and justifications for relevance, accuracy, coherence, and completeness of the synthesized answer. Format clearly.",
        agent=evaluation_agent,
        context=[task_retrieve_info, task_synthesize_answer],
        async_execution=False
    )
    print("Tasks created.")
else:
    print("Not all agents initialized. Tasks cannot be created.")

# --------------------------------------------------------------------------
# 7. CREATE AND RUN THE CREW
# --------------------------------------------------------------------------
if all([task_route_query, task_retrieve_info, task_synthesize_answer, task_evaluate_answer]):
    cricket_rag_crew = Crew(
        agents=[query_router_agent, information_retrieval_agent, synthesis_agent, evaluation_agent],
        tasks=[task_route_query, task_retrieve_info, task_synthesize_answer, task_evaluate_answer],
        process=Process.sequential,
        verbose=True
    )
    print("Advanced Cricket RAG Crew with Hybrid & SQL Search created.")

    print("\n" + "="*50)
    user_cricket_topic = input("Please enter your cricket query: ")
    if not user_cricket_topic:
        user_cricket_topic = "Who was the player of the match in the IPL 2019 final?" # Example for SQL
        # user_cricket_topic = "Tell me about Rahul Dravid's test career" # Example for Hybrid
        # user_cricket_topic = "Latest cricket news about Australia" # Example for Web
        print(f"No query entered, defaulting to: '{user_cricket_topic}'")
    print("="*50 + "\n")
   
    print(f"🚀 Kicking off the Crew for query: '{user_cricket_topic}'...")
    print("="*50 + "\n")

    try:
        inputs = {'user_topic': user_cricket_topic}
        result = cricket_rag_crew.kickoff(inputs=inputs)
       
        print("\n\n" + "="*50 + " FINAL EVALUATED RESULT " + "="*50)
        print(result)
        print("="*126)

        if task_synthesize_answer.output:
            print("\n" + "="*50 + " SYNTHESIZED ANSWER (before evaluation) " + "="*50)
            synthesized_output = task_synthesize_answer.output.exported_output \
                if hasattr(task_synthesize_answer.output, 'exported_output') \
                else task_synthesize_answer.output
            print(synthesized_output)
            print("="*138)

    except Exception as e:
        print(f"\nAn error occurred during crew execution: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Crew cannot be formed due to missing tasks or agents.")