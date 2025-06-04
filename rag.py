import os
import pickle
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

# PDF processing
import PyPDF2
import fitz  # PyMuPDF - alternative PDF reader

# Text processing
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

# Vector database (using FAISS for efficient similarity search)
import faiss

# For the generation part (you can replace with your preferred LLM)
from transformers import pipeline

class PDFProcessor:
    """Handles PDF reading and text extraction"""
    
    def __init__(self):
        # Download NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading {pdf_path} with PyPDF2: {e}")
            return ""
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (fallback method)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            print(f"Error reading {pdf_path} with PyMuPDF: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text with fallback methods"""
        # Try PyPDF2 first
        text = self.extract_text_pypdf2(pdf_path)
        
        # If PyPDF2 fails or returns empty, try PyMuPDF
        if not text.strip():
            text = self.extract_text_pymupdf(pdf_path)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()

class TextChunker:
    """Handles text chunking for better retrieval"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences with overlap"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_with_overlap(self, text: str) -> List[str]:
        """Create overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

class VectorStore:
    """Handles vector embeddings and similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents and create embeddings"""
        self.documents.extend(documents)
        
        # Extract text for embedding
        texts = [doc['content'] for doc in documents]
        
        # Create embeddings
        new_embeddings = self.model.encode(texts, show_progress_bar=True)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Build FAISS index
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index for efficient similarity search"""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self._build_index()

class RAGSystem:
    """Main RAG system that combines retrieval and generation"""
    
    def __init__(self, pdf_directory: str, model_name: str = "all-MiniLM-L6-v2"):
        self.pdf_directory = Path(pdf_directory)
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker()
        self.vector_store = VectorStore(model_name)
        
        # Initialize generator (you can replace with any LLM)
        try:
            self.generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=-1  # Use CPU
            )
        except:
            print("Warning: Could not load generator model. Using simple concatenation.")
            self.generator = None
    
    def load_pdfs(self):
        """Load and process all PDFs in the directory"""
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            # Extract text
            text = self.pdf_processor.extract_text(str(pdf_file))
            
            if not text.strip():
                print(f"Warning: No text extracted from {pdf_file.name}")
                continue
            
            # Clean text
            clean_text = self.pdf_processor.clean_text(text)
            
            # Chunk text
            chunks = self.chunker.chunk_by_sentences(clean_text)
            
            # Create document objects
            for i, chunk in enumerate(chunks):
                doc = {
                    'content': chunk,
                    'source': pdf_file.name,
                    'chunk_id': i,
                    'metadata': {
                        'file_path': str(pdf_file),
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                }
                all_documents.append(doc)
        
        # Add to vector store
        if all_documents:
            print(f"Adding {len(all_documents)} document chunks to vector store...")
            self.vector_store.add_documents(all_documents)
            print("PDF processing complete!")
        else:
            print("No documents were successfully processed.")
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(question, k=k)
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': []
            }
        
        # Prepare context
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # Generate answer
        if self.generator:
            try:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                response = self.generator(prompt, max_length=200, num_return_sequences=1)
                answer = response[0]['generated_text'].split("Answer:")[-1].strip()
            except:
                answer = self._simple_answer_generation(question, context)
        else:
            answer = self._simple_answer_generation(question, context)
        
        # Prepare sources
        sources = []
        for doc in retrieved_docs:
            sources.append({
                'source': doc['source'],
                'similarity_score': doc['similarity_score'],
                'chunk_id': doc['chunk_id'],
                'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            })
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context': context[:500] + "..." if len(context) > 500 else context
        }
    
    def _simple_answer_generation(self, question: str, context: str) -> str:
        """Simple answer generation when no LLM is available"""
        # This is a very basic approach - you should replace with a proper LLM
        sentences = sent_tokenize(context)
        
        # Simple keyword matching
        question_words = set(question.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            if overlap > 0:
                scored_sentences.append((overlap, sentence))
        
        if scored_sentences:
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            return scored_sentences[0][1]
        else:
            return "Based on the available documents: " + sentences[0] if sentences else "No relevant information found."
    
    def save_index(self, filepath: str):
        """Save the vector index"""
        self.vector_store.save(filepath)
    
    def load_index(self, filepath: str):
        """Load a saved vector index"""
        self.vector_store.load(filepath)

# Example usage and testing
def main():
    # Initialize RAG system
    pdf_directory = "./pdfs"  # Change this to your PDF directory
    rag = RAGSystem(pdf_directory)
    
    # Check if directory exists
    if not os.path.exists(pdf_directory):
        print(f"Creating directory: {pdf_directory}")
        os.makedirs(pdf_directory)
        print(f"Please add PDF files to {pdf_directory} and run again.")
        return
    
    # Load PDFs
    print("Loading and processing PDFs...")
    rag.load_pdfs()
    
    # Save index for future use
    index_path = "rag_index.pkl"
    rag.save_index(index_path)
    print(f"Index saved to {index_path}")
    
    # Interactive querying
    print("\n" + "="*50)
    print("RAG System Ready! Ask questions about your PDFs.")
    print("Type 'quit' to exit.")
    print("="*50)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            result = rag.query(question)
            
            print(f"\nQuestion: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"\nSources ({len(result['sources'])}):")
            
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['source']} (similarity: {source['similarity_score']:.3f})")
                print(f"   Preview: {source['content_preview']}")
        
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()