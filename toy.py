import pyterrier as pt
import pandas as pd
import os
import re
from pathlib import Path
import json
from typing import List, Dict, Tuple
import PyPDF2
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CybersecurityIRSystem:
    """
    Information Retrieval System for Cybersecurity Books using PyTerrier
    """
    
    def __init__(self, index_path: str = "./cybersec_index"):
        """
        Initialize the IR system
        
        Args:
            index_path: Path where the Terrier index will be stored
        """
        # Initialize PyTerrier
        if not pt.java.started():
            pt.java.init()
        
        # Create absolute path for index
        self.index_path = os.path.abspath(index_path)
        
        # Create index directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        
        self.index = None
        self.retrieval_model = None
        self.documents = []
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better indexing
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX file
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 30, overlap: int = 400) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk (in characters)
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        # Split by paragraphs first (double newlines)
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap from the end of current chunk
                    if overlap > 0:
                        words = current_chunk.split()
                        overlap_words = words[-overlap//20:] if len(words) > overlap//20 else words[-20:]
                        current_chunk = ' '.join(overlap_words) + '\n\n' + para
                    else:
                        current_chunk = para
                else:
                    # Single paragraph is too large, split by sentences
                    sentences = sent_tokenize(para)
                    temp_chunk = ""
                    
                    for sent in sentences:
                        if len(temp_chunk) + len(sent) + 1 > chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = sent
                            else:
                                # Single sentence is very long, split by words
                                words = sent.split()
                                word_chunk = ""
                                for word in words:
                                    if len(word_chunk) + len(word) + 1 > chunk_size:
                                        if word_chunk:
                                            chunks.append(word_chunk.strip())
                                        word_chunk = word
                                    else:
                                        word_chunk += " " + word if word_chunk else word
                                if word_chunk:
                                    chunks.append(word_chunk.strip())
                        else:
                            temp_chunk += " " + sent if temp_chunk else sent
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100]
        
        return chunks if chunks else [text]  # Return original text if no good chunks
    
    def load_documents_from_directory(self, directory_path: str) -> List[Dict]:
        """
        Load and process documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processed documents
        """
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory {directory_path} does not exist")
            return documents
        
        doc_id = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                try:
                    if file_ext == '.pdf':
                        text = self.extract_text_from_pdf(str(file_path))
                    elif file_ext == '.docx':
                        text = self.extract_text_from_docx(str(file_path))
                    elif file_ext == '.txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    else:
                        continue
                    
                    if text.strip():
                        # Preprocess text
                        processed_text = self.preprocess_text(text)
                        
                        # Create chunks for better retrieval
                        chunks = self.chunk_text(processed_text)
                        
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'docno': f"doc{doc_id}_c{i}",
                                'text': chunk,
                                'title': file_path.stem,
                                'source_file': str(file_path),
                                'chunk_id': i,
                                'doc_id': doc_id
                            })
                        
                        doc_id += 1
                        logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Total documents processed: {len(documents)}")
        return documents
    
    def load_single_document(self, file_path: str, title: str = None) -> List[Dict]:
        """
        Load and process a single document
        
        Args:
            file_path: Path to the document
            title: Optional title for the document
            
        Returns:
            List of processed document chunks
        """
        documents = []
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File {file_path} does not exist")
            return documents
        
        file_ext = path.suffix.lower()
        
        try:
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_ext == '.docx':
                text = self.extract_text_from_docx(file_path)
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return documents
            
            if text.strip():
                processed_text = self.preprocess_text(text)
                chunks = self.chunk_text(processed_text)
                
                doc_title = title or path.stem
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'docno': f"doc0_c{i}",
                        'text': chunk,
                        'title': doc_title,
                        'source_file': file_path,
                        'chunk_id': i,
                        'doc_id': 0
                    })
                
                logger.info(f"Processed {path.name}: {len(chunks)} chunks")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        return documents
    
    def build_index(self, documents: List[Dict] = None):
        """
        Build Terrier index from documents
        
        Args:
            documents: List of documents to index (uses self.documents if None)
        """
        if documents is not None:
            self.documents = documents
        
        if not self.documents:
            logger.error("No documents to index")
            return
        
        # Create DataFrame for PyTerrier
        df = pd.DataFrame(self.documents)
        
        # Build index with proper meta configuration
        logger.info("Building index...")
        
        # Get the maximum text length from documents
        max_text_length = max(len(doc['text']) for doc in self.documents) if self.documents else 10000
        # Add some buffer to the max length
        text_field_length = max_text_length + 1000
        
        logger.info(f"Configuring index with text field length: {text_field_length}")
        
        # Configure meta field lengths to handle content
        meta_config = {
            'docno': 50,  # Length for document IDs
            'text': text_field_length  # Dynamic length based on actual content
        }
        
        try:
            indexer = pt.IterDictIndexer(self.index_path, overwrite=True, meta=meta_config)
            self.index = indexer.index(self.documents)
        except Exception as e:
            logger.error(f"Error building index: {e}")
            # Try with a very large text field length as fallback
            logger.info("Retrying with maximum text field length...")
            meta_config['text'] = 500000  # 500K characters
            indexer = pt.IterDictIndexer(self.index_path, overwrite=True, meta=meta_config)
            self.index = indexer.index(self.documents)
        
        logger.info(f"Index built successfully with {len(self.documents)} documents")
        
        # Initialize retrieval model with different scoring options
        self.retrieval_model = pt.BatchRetrieve(self.index, wmodel="BM25")
        
        # Also create alternative models for comparison
        self.tfidf_model = pt.BatchRetrieve(self.index, wmodel="TF_IDF")
        self.dph_model = pt.BatchRetrieve(self.index, wmodel="DPH")
    
    def load_existing_index(self):
        """
        Load an existing Terrier index
        """
        if os.path.exists(self.index_path):
            try:
                self.index = pt.IndexFactory.of(self.index_path)
                self.retrieval_model = pt.BatchRetrieve(self.index, wmodel="BM25")
                logger.info("Existing index loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")
                return False
        else:
            logger.warning(f"Index path {self.index_path} does not exist")
            return False
    
    def search(self, query: str, num_results: int = 10, model: str = "BM25") -> pd.DataFrame:
        """
        Search the index for relevant documents
        
        Args:
            query: Search query
            num_results: Number of results to return
            model: Retrieval model to use ("BM25", "TF_IDF", "DPH")
            
        Returns:
            DataFrame with search results
        """
        if self.retrieval_model is None:
            logger.error("Index not built or loaded")
            return pd.DataFrame()
        
        # Select the appropriate model
        if model == "TF_IDF" and hasattr(self, 'tfidf_model'):
            retrieval_model = self.tfidf_model
        elif model == "DPH" and hasattr(self, 'dph_model'):
            retrieval_model = self.dph_model
        else:
            retrieval_model = self.retrieval_model
        
        # Create query DataFrame
        query_df = pd.DataFrame([{'qid': '1', 'query': query}])
        
        # Perform search
        results = retrieval_model.transform(query_df)
        
        # Limit results
        results = results.head(num_results)
        
        # Add document metadata
        if not results.empty and hasattr(self, 'documents') and self.documents:
            doc_lookup = {doc['docno']: doc for doc in self.documents}
            results['title'] = results['docno'].map(lambda x: doc_lookup.get(x, {}).get('title', ''))
            results['source_file'] = results['docno'].map(lambda x: doc_lookup.get(x, {}).get('source_file', ''))
            results['chunk_id'] = results['docno'].map(lambda x: doc_lookup.get(x, {}).get('chunk_id', ''))
        
        return results
    
    def get_document_text(self, docno: str) -> str:
        """
        Get the full text of a document by its ID
        
        Args:
            docno: Document ID
            
        Returns:
            Document text
        """
        if self.index is None:
            return ""
        
        try:
            # Get document from index
            meta_index = self.index.getMetaIndex()
            docid = self.index.getCollectionStatistics().getNumberOfDocuments()
            
            for i in range(docid):
                if meta_index.getItem("docno", i) == docno:
                    return meta_index.getItem("text", i)
            
            return ""
        except Exception as e:
            logger.error(f"Error retrieving document {docno}: {e}")
            return ""
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Generate query suggestions based on document terms
        
        Args:
            partial_query: Partial query string
            
        Returns:
            List of suggested queries
        """
        # This is a simple implementation - you could enhance with more sophisticated methods
        cybersec_terms = [
            "malware", "virus", "trojan", "ransomware", "phishing", "firewall",
            "encryption", "authentication", "authorization", "vulnerability",
            "exploit", "penetration testing", "incident response", "forensics",
            "network security", "web security", "cryptography", "ssl", "tls",
            "intrusion detection", "security policy", "risk assessment",
            "threat modeling", "secure coding", "access control"
        ]
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        for term in cybersec_terms:
            if partial_lower in term or term.startswith(partial_lower):
                suggestions.append(term)
        
        return suggestions[:5]  # Return top 5 suggestions

# Example usage and utility functions
def main():
    """
    Example usage of the CybersecurityIRSystem
    """
    # Initialize the IR system
    ir_system = CybersecurityIRSystem()
    
    # Check if there are any PDF files in the current directory
    current_dir = Path(".")
    pdf_files = list(current_dir.glob("*.pdf"))
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF file(s):")
        for i, pdf_file in enumerate(pdf_files):
            print(f"{i+1}. {pdf_file.name}")
        
        # Use the first PDF file found
        selected_file = pdf_files[0]
        print(f"\nProcessing: {selected_file.name}")
        
        # Load the document
        documents = ir_system.load_single_document(str(selected_file), selected_file.stem)
        
        if documents:
            print(f"Loaded {len(documents)} document chunks")
            
            # Build the index
            print("Building index...")
            ir_system.build_index(documents)
            
            # Example searches
            queries = [
                "privacy compromise",
                "white box"
            ]
            
            print("\n" + "="*60)
            print("SEARCH RESULTS")
            print("="*60)
            
            for query in queries:
                print(f"\n--- Search Results for: '{query}' ---")
                
                # Try multiple retrieval models
                for model_name in ["BM25", "TF_IDF"]:
                    print(f"\n  Using {model_name} model:")
                    results = ir_system.search(query, num_results=3, model=model_name)
                    
                    if not results.empty:
                        for idx, row in results.iterrows():
                            print(f"    Rank {row['rank']}: {row['docno']} (Score: {row['score']:.4f})")
                            print(f"    Title: {row.get('title', 'N/A')}")
                            # Get text preview
                            doc_text = ""
                            if hasattr(ir_system, 'documents') and ir_system.documents:
                                for doc in ir_system.documents:
                                    if doc['docno'] == row['docno']:
                                        doc_text = doc['text']
                                        break

                            if doc_text:
                                query_terms = query.lower().split()
                                lower_text = doc_text.lower()

                                # Search for each term in text and show context
                                context_found = False
                                for term in query_terms:
                                    pos = lower_text.find(term)
                                    if pos != -1:
                                        start = max(0, pos - 100)
                                        end = min(len(doc_text), pos + len(term) + 100)
                                        context = doc_text[start:end].replace('\n', ' ').strip()
                                        print(f"    Context: ...{context}...")
                                        context_found = True
                                        break
                                
                                if not context_found:
                                    p = doc_text[:300].replace('\n', ' ').strip()
                                    print(f"    Preview: {p}...")
                            else:
                                print("    No results found.")
                            
            print(f"\nSystem ready! You can now search the {selected_file.name} document.")
            print("To search interactively, modify the main function or create a search loop.")
        
        else:
            print(f"Failed to load documents from {selected_file.name}")
            print("Please check if the file is a valid PDF and not corrupted.")
    
    else:
        print("No PDF files found in the current directory.")
        print("\nTo use this system:")
        print("1. Place your cybersecurity book (PDF format) in the current directory")
        print("2. Or modify the main() function to specify the file path")
        print("3. Or use load_documents_from_directory() for multiple files")
        
        # Show example of how to use with a specific file
        print("\nExample usage:")
        print("documents = ir_system.load_single_document('path/to/your/book.pdf', 'Book Title')")
        print("ir_system.build_index(documents)")
        print("results = ir_system.search('your search query')")

def interactive_search():
    """
    Interactive search function for testing
    """
    ir_system = CybersecurityIRSystem()
    
    # Try to load existing index first
    if ir_system.load_existing_index():
        print("Loaded existing index.")
    else:
        print("No existing index found. Please run main() first to build an index.")
        return
    
    print("\n" + "="*50)
    print("INTERACTIVE SEARCH MODE")
    print("="*50)
    print("Enter your search queries (type 'quit' to exit):")
    
    while True:
        query = input("\nSearch: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        results = ir_system.search(query, num_results=5)
        
        if not results.empty:
            print(f"\nFound {len(results)} results:")
            for idx, row in results.iterrows():
                print(f"\n{row['rank']}. {row['docno']} (Score: {row['score']:.4f})")
                
                # Get document text for preview
                doc_text = ""
                if hasattr(ir_system, 'documents') and ir_system.documents:
                    for doc in ir_system.documents:
                        if doc['docno'] == row['docno']:
                            doc_text = doc['text']
                            break
                
                if doc_text:
                    preview = doc_text[:200].replace('\n', ' ').strip()
                    print(f"   {preview}...")
        else:
            print("No results found.")
    
    print("Search session ended.")

if __name__ == "__main__":
    main()