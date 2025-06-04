import pyterrier as pt
import pandas as pd
import os
import re
from pathlib import Path
import json
from typing import List, Dict
import PyPDF2
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CybersecurityIRSystem:
    def __init__(self, index_path: str = "./cybersec_index"):
        # Initialize PyTerrier
        if not pt.started():
            pt.init()

        self.index_path = os.path.abspath(index_path)
        os.makedirs(self.index_path, exist_ok=True)
        self.index = None
        self.retrieval_model = None
        self.documents = []

        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """Less aggressive preprocessing to preserve important terms"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Only remove some punctuation, keep important ones like hyphens
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase but preserve original structure
        return text.lower()

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                return text
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            return "\n".join(full_text)
        except Exception as e:
            logger.error(f"DOCX extraction failed for {file_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Improved chunking with sentence boundaries"""
        if not text or len(text) < 50:
            return []
        
        # Try sentence-based chunking first
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                # Fallback to word-based chunking
                words = word_tokenize(text)
                if len(words) < 50:
                    return [text] if len(words) > 10 else []
                
                chunks = []
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) > 10:  # Only keep substantial chunks
                        chunks.append(" ".join(chunk_words))
                return chunks
            
            # Sentence-based chunking
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_words = len(word_tokenize(sentence))
                
                if current_length + sentence_words > chunk_size and current_chunk:
                    # Finalize current chunk
                    chunk_text = " ".join(current_chunk)
                    if len(word_tokenize(chunk_text)) > 20:  # Only keep substantial chunks
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > 1:
                        overlap_sentences = current_chunk[-2:]  # Keep last 2 sentences
                        current_chunk = overlap_sentences + [sentence]
                        current_length = sum(len(word_tokenize(s)) for s in current_chunk)
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_words
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_words
            
            # Add final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(word_tokenize(chunk_text)) > 20:
                    chunks.append(chunk_text)
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}, falling back to word-based chunking")
            
            # Fallback to word-based chunking
            words = word_tokenize(text)
            if len(words) < 50:
                return [text] if len(words) > 10 else []
            
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) > 10:
                    chunks.append(" ".join(chunk_words))
            return chunks

    def load_documents_from_directory(self, directory_path: str) -> List[Dict]:
        documents = []
        doc_id = 0
        directory = Path(directory_path)

        supported_extensions = ['.pdf', '.docx', '.txt']
        files_found = list(directory.glob("*.*"))
        
        logger.info(f"Found {len(files_found)} files in directory")
        
        for file_path in files_found:
            ext = file_path.suffix.lower()
            if ext not in supported_extensions:
                logger.info(f"Skipping unsupported file: {file_path.name}")
                continue

            try:
                logger.info(f"Processing file: {file_path.name}")
                text = ""
                
                if ext == '.pdf':
                    text = self.extract_text_from_pdf(str(file_path))
                elif ext == '.docx':
                    text = self.extract_text_from_docx(str(file_path))
                elif ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                if not text or len(text.strip()) < 50:
                    logger.warning(f"No meaningful text extracted from {file_path.name}")
                    continue

                # Store original text for preview
                original_text = text
                text = self.preprocess_text(text)
                
                if not text:
                    logger.warning(f"Text became empty after preprocessing for {file_path.name}")
                    continue

                chunks = self.chunk_text(text)
                logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
                
                if not chunks:
                    logger.warning(f"No chunks created for {file_path.name}")
                    continue

                for i, chunk in enumerate(chunks):
                    # Store both processed and original text portions for preview
                    chunk_start = i * 400  # Approximate start position
                    chunk_end = min(chunk_start + 800, len(original_text))
                    original_chunk = original_text[chunk_start:chunk_end]
                    
                    documents.append({
                        'docno': f'doc{doc_id}_chunk{i}',
                        'text': chunk,
                        'title': file_path.stem,
                        'source_file': str(file_path),
                        'chunk_id': str(i),
                        'doc_id': str(doc_id)
                    })
                    
                    # Store preview information separately for display
                    self.preview_info = getattr(self, 'preview_info', {})
                    self.preview_info[f'doc{doc_id}_chunk{i}'] = {
                        'original_text': original_chunk[:500] + "..." if len(original_chunk) > 500 else original_chunk,
                        'file_extension': ext,
                        'chunk_size': len(chunk)
                    }
                
                logger.info(f"Successfully processed {file_path.name}: {len(chunks)} chunks")
                doc_id += 1

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue

        self.documents = documents
        logger.info(f"Total documents loaded: {len(documents)} chunks from {doc_id} files")
        return documents

    def build_index(self, documents: List[Dict] = None):
        if documents is not None:
            self.documents = documents

        if not self.documents:
            logger.error("No documents to index")
            return

        logger.info(f"Building index with {len(self.documents)} document chunks")
        
        # Clean documents and ensure all required fields are strings
        cleaned_documents = []
        for doc in self.documents:
            cleaned_doc = {
                'docno': str(doc.get('docno', '')),
                'text': str(doc.get('text', '')),
                'title': str(doc.get('title', '')),
                'source_file': str(doc.get('source_file', '')),
                'chunk_id': str(doc.get('chunk_id', '0')),
                'doc_id': str(doc.get('doc_id', '0'))
            }
            # Only add non-empty documents
            if cleaned_doc['text'].strip():
                cleaned_documents.append(cleaned_doc)
        
        if not cleaned_documents:
            logger.error("No valid documents after cleaning")
            return
            
        self.documents = cleaned_documents
        logger.info(f"Cleaned documents: {len(cleaned_documents)} valid documents")
        
        # Calculate max text length safely
        text_lengths = []
        for doc in cleaned_documents:
            text_len = len(doc.get('text', ''))
            if text_len > 0:
                text_lengths.append(text_len)
        
        if not text_lengths:
            logger.error("No documents with text content found")
            return
            
        max_text_len = max(text_lengths) + 1000
        logger.info(f"Maximum text length: {max_text_len}")

        # Simplified metadata configuration - only essential fields
        meta_config = {
            'docno': 100,
            'text': max(10000, int(max_text_len)),  # Ensure minimum size
            'title': 300,
            'source_file': 500,
            'chunk_id': 50,
            'doc_id': 50
        }

        try:
            logger.info("Creating indexer...")
            indexer = pt.IterDictIndexer(
                self.index_path, 
                overwrite=True, 
                meta=meta_config,
                blocks=True
            )
            
            logger.info("Starting indexing process...")
            self.index = indexer.index(cleaned_documents)
            
            # Initialize retrieval models
            logger.info("Initializing retrieval models...")
            self.retrieval_model = pt.BatchRetrieve(self.index, wmodel="BM25", num_results=20)
            self.tfidf_model = pt.BatchRetrieve(self.index, wmodel="TF_IDF", num_results=20)
            self.dph_model = pt.BatchRetrieve(self.index, wmodel="DPH", num_results=20)
            
            logger.info(f"Index built successfully with {len(cleaned_documents)} documents")
            
            # Try to get index statistics
            try:
                stats = self.index.getCollectionStatistics()
                logger.info(f"Index statistics: {stats}")
            except Exception as stats_e:
                logger.warning(f"Could not retrieve index statistics: {stats_e}")
            
        except Exception as e:
            logger.error(f"Index build failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback with larger text field and simpler config
            logger.info("Attempting fallback indexing...")
            try:
                fallback_config = {
                    'docno': 100,
                    'text': max(500000, int(max_text_len * 3)),
                    'title': 300,
                    'source_file': 500,
                    'chunk_id': 50,
                    'doc_id': 50
                }
                
                indexer = pt.IterDictIndexer(
                    self.index_path, 
                    overwrite=True, 
                    meta=fallback_config,
                    blocks=True
                )
                self.index = indexer.index(cleaned_documents)
                self.retrieval_model = pt.BatchRetrieve(self.index, wmodel="BM25", num_results=20)
                self.tfidf_model = pt.BatchRetrieve(self.index, wmodel="TF_IDF", num_results=20)  
                self.dph_model = pt.BatchRetrieve(self.index, wmodel="DPH", num_results=20)
                logger.info(f"Fallback index built successfully with {len(cleaned_documents)} documents")
            except Exception as fallback_e:
                logger.error(f"Fallback index build also failed: {fallback_e}")
                import traceback
                logger.error(f"Fallback traceback: {traceback.format_exc()}")
                self.index = None

    def search(self, query: str, num_results: int = 10, model: str = "BM25") -> pd.DataFrame:
        if self.index is None:
            logger.error("Index not built")
            return pd.DataFrame()

        original_query = query
        query = self.preprocess_text(query)
        
        if not query:
            logger.error("Invalid or empty query")
            return pd.DataFrame()

        logger.info(f"Searching for: '{original_query}' -> '{query}'")

        models = {
            "BM25": self.retrieval_model,
            "TF_IDF": self.tfidf_model,
            "DPH": self.dph_model
        }
        retriever = models.get(model, self.retrieval_model)
        
        try:
            qdf = pd.DataFrame([{'qid': '1', 'query': query}])
            results = retriever.transform(qdf).head(num_results)

            if not results.empty:
                # Create lookup dictionary for document metadata
                doc_lookup = {doc['docno']: doc for doc in self.documents}
                preview_lookup = getattr(self, 'preview_info', {})
                
                # Add metadata to results
                results['title'] = results['docno'].map(lambda x: doc_lookup.get(x, {}).get('title', 'Unknown'))
                results['source_file'] = results['docno'].map(lambda x: doc_lookup.get(x, {}).get('source_file', 'Unknown'))
                results['chunk_id'] = results['docno'].map(lambda x: doc_lookup.get(x, {}).get('chunk_id', '0'))
                results['doc_id'] = results['docno'].map(lambda x: doc_lookup.get(x, {}).get('doc_id', '0'))
                
                # Add preview information
                results['original_text'] = results['docno'].map(lambda x: preview_lookup.get(x, {}).get('original_text', ''))
                results['file_extension'] = results['docno'].map(lambda x: preview_lookup.get(x, {}).get('file_extension', ''))
                
                # Sort by score (descending)
                results = results.sort_values('score', ascending=False)
                
                logger.info(f"Found {len(results)} results for query: {original_query}")
            else:
                logger.warning(f"No results found for query: {original_query}")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{original_query}': {e}")
            return pd.DataFrame()

    def display_search_results(self, results: pd.DataFrame, show_preview: bool = True):
        """Display search results with preview"""
        if results.empty:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:")
        print("=" * 80)
        
        for idx, row in results.iterrows():
            print(f"\nRank {idx + 1}:")
            print(f"Document: {row.get('title', 'Unknown')}")
            print(f"Source: {Path(row.get('source_file', 'Unknown')).name}")
            print(f"Chunk ID: {row.get('chunk_id', 'Unknown')}")
            print(f"Score: {row.get('score', 0):.4f}")
            
            if show_preview and 'original_text' in row and row['original_text']:
                print(f"Preview: {row['original_text'][:300]}...")
            
            print("-" * 40)

    def get_document_stats(self):
        """Get statistics about loaded documents"""
        if not self.documents:
            return "No documents loaded"
        
        stats = {}
        for doc in self.documents:
            source = Path(doc['source_file']).name
            if source not in stats:
                stats[source] = {'chunks': 0, 'total_chars': 0}
            stats[source]['chunks'] += 1
            stats[source]['total_chars'] += len(doc.get('text', ''))
        
        print("\nDocument Statistics:")
        print("=" * 50)
        for source, info in stats.items():
            print(f"{source}: {info['chunks']} chunks, {info['total_chars']:,} characters")
        
        return stats

if __name__ == "__main__":
    try:
        ir = CybersecurityIRSystem()
        
        # Load documents
        print("Loading documents...")
        docs = ir.load_documents_from_directory(".")
        
        if docs:
            print(f"Loaded {len(docs)} document chunks")
            ir.get_document_stats()
            
            # Build index
            print("\nBuilding index...")
            ir.build_index(docs)
            
            if ir.index is not None:
                # Test queries
                queries = ["privacy compromise", "white box", "Bayes-based spam classifiers", "cybersecurity", "malware"]
                
                for q in queries:
                    print(f"\n{'='*60}")
                    print(f"Query: '{q}'")
                    print('='*60)
                    
                    for model in ["BM25", "TF_IDF"]:
                        print(f"\n--- Results using {model} ---")
                        results = ir.search(q, num_results=5, model=model)
                        
                        if not results.empty:
                            ir.display_search_results(results, show_preview=True)
                        else:
                            print("No results found")
            else:
                print("Failed to build index")
        else:
            print("No documents found in directory")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()