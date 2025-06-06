import os
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import numpy as np
from pathlib import Path
import json
import fitz  # PyMuPDF
import re
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline
import torch

class OptimizedPDFProcessor:
    """Enhanced PDF processing with better text extraction"""
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
    
    def extract_text_with_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text while preserving structure information"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            pages_text = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                pages_text.append({
                    'page_num': page_num + 1,
                    'text': page_text,
                    'word_count': len(page_text.split())
                })
                full_text += f"\n[Page {page_num + 1}]\n{page_text}"
            doc.close()
            return {
                'full_text': full_text,
                'pages': pages_text,
                'total_pages': len(pages_text),
                'total_words': len(full_text.split())
            }
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {'full_text': '', 'pages': [], 'total_pages': 0, 'total_words': 0}
    
    def advanced_text_cleaning(self, text: str) -> str:
        """Advanced text cleaning with better preprocessing"""
        text = re.sub(r'\[Page \d+\]', '', text)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        return text.strip()

class AdaptiveChunker:
    """Smart chunking that adapts to content structure"""
    def __init__(self, base_chunk_size: int = 400, overlap_ratio: float = 0.15):
        self.base_chunk_size = base_chunk_size
        self.overlap_ratio = overlap_ratio
        self.min_chunk_size = 100
        self.max_chunk_size = 800
    
    def semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Create semantically coherent chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())
            target_chunk_size = self._calculate_dynamic_chunk_size(sentence)
            if (len(current_chunk.split()) + sentence_words <= target_chunk_size or 
                len(current_chunk.split()) < self.min_chunk_size):
                current_chunk += sentence + " "
                current_sentences.append(i)
            else:
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'sentence_indices': current_sentences.copy(),
                        'word_count': len(current_chunk.split()),
                        'chunk_id': len(chunks)
                    })
                overlap_sentences = self._get_overlap_sentences(sentences, current_sentences)
                current_chunk = " ".join(overlap_sentences) + " " + sentence + " "
                current_sentences = list(range(max(0, i - len(overlap_sentences)), i + 1))
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'sentence_indices': current_sentences,
                'word_count': len(current_chunk.split()),
                'chunk_id': len(chunks)
            })
        return chunks
    
    def _calculate_dynamic_chunk_size(self, sentence: str) -> int:
        """Calculate optimal chunk size based on content"""
        words = sentence.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        if avg_word_length > 6:
            return int(self.base_chunk_size * 0.8)
        elif avg_word_length < 4:
            return int(self.base_chunk_size * 1.2)
        else:
            return self.base_chunk_size
    
    def _get_overlap_sentences(self, sentences: List[str], current_indices: List[int]) -> List[str]:
        """Get sentences for overlap"""
        if not current_indices:
            return []
        overlap_count = max(1, int(len(current_indices) * self.overlap_ratio))
        start_idx = max(0, current_indices[-1] - overlap_count + 1)
        end_idx = current_indices[-1] + 1
        return sentences[start_idx:end_idx]

class EnhancedVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=450, overlap_ratio=0.15):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device for embeddings: {self.device}")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.vector_db = None
        self.documents = []
        self.primary_model = SentenceTransformer(model_name)
        self.models = [self.primary_model]
        self.indexes = []
        self.embeddings_list = []

    def embed_documents(self, texts: list[str]):
        """Generate embeddings for a list of texts"""
        print(f"Generating embeddings on {self.device}...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        return embeddings.cpu().numpy()
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents with multi-model embeddings"""
        self.documents.extend(documents)
        texts = [doc['content'] for doc in documents]
        for i, model in enumerate(self.models):
            print(f"Creating embeddings with model {i+1}/{len(self.models)}")
            new_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
            if len(self.embeddings_list) <= i:
                self.embeddings_list.append(new_embeddings)
            else:
                self.embeddings_list[i] = np.vstack([self.embeddings_list[i], new_embeddings])
        self._build_indexes()
    
    def _build_indexes(self):
        """Build FAISS indexes for each embedding model"""
        self.indexes = []
        for embeddings in self.embeddings_list:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            index.add(normalized_embeddings.astype('float32'))
            self.indexes.append(index)
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Advanced search combining multiple models and reranking"""
        if not self.indexes:
            return []
        all_results = []
        for i, (model, index) in enumerate(zip(self.models, self.indexes)):
            query_embedding = model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            scores, indices = index.search(query_embedding.astype('float32'), min(k*2, len(self.documents)))
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    result = self.documents[idx].copy()
                    result['similarity_score'] = float(score)
                    result['model_id'] = i
                    all_results.append(result)
        unique_results = self._deduplicate_and_rerank(all_results, query)
        return unique_results[:k]
    
    def _deduplicate_and_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """Remove duplicates and rerank results"""
        content_groups = {}
        for result in results:
            content_key = result['content'][:100]
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(result)
        unique_results = []
        for group in content_groups.values():
            best_result = max(group, key=lambda x: x['similarity_score'])
            keyword_bonus = self._calculate_keyword_bonus(query, best_result['content'])
            best_result['final_score'] = best_result['similarity_score'] + keyword_bonus
            unique_results.append(best_result)
        unique_results.sort(key=lambda x: x['final_score'], reverse=True)
        return unique_results
    
    def _calculate_keyword_bonus(self, query: str, content: str) -> float:
        """Calculate bonus score based on keyword overlap"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words))
        return overlap * 0.05

class Llama32Generator:
    def __init__(self, config: Dict):
        self.model_path = config.get('llama_model_name', 'Llama-3.2-3B-Instruct-Q4_K_M')
        self.max_context_length = config.get('max_context_length', 2048)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        self.model = None
        self.base_url = config.get('base_url', 'http://127.0.0.1:1234/v1')
        self.model_name = self.model_path
        self.use_chat_endpoint = True
        self.load_model()

    def load_model(self):
        """Initialize the OpenAI client for LM Studio's local server"""
        try:
            print(f"Checking if LM Studio server is running at {self.base_url}")
            response = requests.get(f"{self.base_url}/models")
            if response.status_code != 200:
                raise Exception(f"Server not running or inaccessible: {response.status_code} {response.text}")
            print(f"Connecting to LM Studio server at {self.base_url}")
            self.model = OpenAI(base_url=self.base_url, api_key='not-needed')
            print(f"Testing connection to {self.model_name}...")
            try:
                test_response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    max_tokens=20,
                    temperature=self.temperature
                )
                output = test_response.choices[0].message.content
            except Exception as chat_error:
                print(f"Chat endpoint failed: {chat_error}")
                print("Trying completions endpoint...")
                self.use_chat_endpoint = False
                test_response = self.model.completions.create(
                    model=self.model_name,
                    prompt="Hello, how are you?",
                    max_tokens=20,
                    temperature=self.temperature
                )
                output = test_response.choices[0].text.strip()
            if not output:
                raise Exception("No valid response from server")
            print(f"Test output: {output}")
            print(f"✅ Connected to LM Studio server using {'/chat/completions' if self.use_chat_endpoint else '/completions'} endpoint")
        except Exception as e:
            error_msg = f"❌ Failed to connect to LM Studio server: {str(e)}"
            print(error_msg)
            print("Please ensure: (1) LM Studio is running, (2) the model is loaded, (3) the port matches base_url.")
            raise Exception(error_msg)

    def create_rag_prompt(self, question: str, context: str, system_prompt: str = None, history: List[Dict[str, str]] = None) -> str:
        """Create optimized prompt for RAG with enhanced conversation history"""
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context and conversation history. 
Use the conversation history to maintain context, especially for follow-up questions, and resolve ambiguous references (e.g., 'it' or 'this') by referring to previous questions and answers. 
Provide accurate, detailed responses using the given context and relevant history. If the context or history lacks sufficient information, clearly state so."""
        
        # Filter relevant history entries (e.g., containing 'PII' or related terms)
        relevant_history = []
        if history:
            query_words = set(question.lower().split())
            for entry in history[-5:]:  # Last 5 exchanges
                entry_text = (entry['question'] + " " + entry['answer']).lower()
                if any(word in entry_text for word in query_words | {'pii', 'personally', 'identifiable', 'information'}):
                    relevant_history.append(entry)
        
        # Summarize history to save tokens
        history_text = ""
        max_history_length = self.max_context_length // 3  # Reserve 1/3 for history
        current_length = 0
        for entry in reversed(relevant_history):
            summary = f"Q: {entry['question']}\nA: {entry['answer'][:100] + '...' if len(entry['answer']) > 100 else entry['answer']}\n"
            if current_length + len(summary) <= max_history_length:
                history_text = summary + history_text
                current_length += len(summary)
            else:
                break
        
        # Resolve pronouns in the question
        resolved_question = question
        if 'it' in question.lower().split() and relevant_history:
            # Assume 'it' refers to the main topic of the last relevant question (e.g., PII)
            last_topic = relevant_history[-1]['question'].lower()
            if 'pii' in last_topic or 'personally identifiable information' in last_topic:
                resolved_question = re.sub(r'\bit\b', 'PII', question, flags=re.IGNORECASE)
        
        if self.use_chat_endpoint:
            return f"{system_prompt}\n\nConversation History:\n{history_text}\nContext:\n{context}\n\nQuestion: {resolved_question}\n\nAnswer:"
        else:
            return f"{system_prompt}\nConversation History:\n{history_text}\nContext: {context}\nQuestion: {resolved_question}\nAnswer:"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: Optional[float] = None) -> str:
        """Generate text using LM Studio's local server"""
        try:
            if self.use_chat_endpoint:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=self.top_p
                )
                return response.choices[0].message.content.strip()
            else:
                response = self.model.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=self.top_p
                )
                return response.choices[0].text.strip()
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return ""

    def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: Optional[float] = None) -> str:
        """Stream text generation"""
        try:
            response_text = ""
            if self.use_chat_endpoint:
                stream = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=self.top_p,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end='', flush=True)
                        response_text += content
            else:
                stream = self.model.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=self.top_p,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].text:
                        content = chunk.choices[0].text
                        print(content, end='', flush=True)
                        response_text += content
                print()
            return response_text.strip()
        except Exception as e:
            print(f"❌ Streaming error: {str(e)}")
            return ""

class ImprovedRAGWithLlama32:
    """Enhanced RAG system with Llama 3.2-3B integration and improved conversation context"""
    
    def __init__(self, pdf_directory: str, config: Dict[str, Any] = None):
        self.pdf_directory = Path(pdf_directory)
        self.config = config or self._get_default_config()
        self.pdf_processor = OptimizedPDFProcessor()
        self.chunker = AdaptiveChunker(
            base_chunk_size=self.config['chunk_size'],
            overlap_ratio=self.config['overlap_ratio']
        )
        self.vector_store = EnhancedVectorStore(
            model_name=self.config['embedding_model'],
        )
        self.conversation_history = []  # Store detailed conversation history
        self._initialize_llama32()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Llama 3.2-3B RAG"""
        model_path = r'C:\Users\Yojith\.lmstudio\models\lmstudio-community\Llama-3.2-3B-Instruct-GGUF\Llama-3.2-3B-Instruct-Q4_K_M.gguf'
        return {
            'chunk_size': 400,
            'overlap_ratio': 0.15,
            'embedding_model': 'all-MiniLM-L6-v2',
            'use_multiple_models': False,
            'llama_model_name': model_path,
            'base_url': 'http://127.0.0.1:1234/v1',
            'device_map': 'auto',
            'max_context_length': 2048,
            'retrieval_k': 5,
            'generation_params': {
                'max_new_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True
            }
        }
    
    def _initialize_llama32(self):
        """Initialize Llama 3.2-3B generator"""
        try:
            self.llama32_generator = Llama32Generator(self.config)
            print("✅ Llama 3.2-3B RAG system initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize Llama 3.2-3B: {e}")
            self.llama32_generator = None
    
    def load_pdfs_optimized(self):
        """Load and process PDFs"""
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            return
        print(f"Processing {len(pdf_files)} PDF files with optimizations...")
        all_documents = []
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            extracted_data = self.pdf_processor.extract_text_with_structure(str(pdf_file))
            if not extracted_data['full_text'].strip():
                print(f"Warning: No text extracted from {pdf_file.name}")
                continue
            clean_text = self.pdf_processor.advanced_text_cleaning(extracted_data['full_text'])
            chunks = self.chunker.semantic_chunking(clean_text)
            print(f"Created {len(chunks)} semantic chunks from {pdf_file.name}")
            for chunk_data in chunks:
                doc = {
                    'content': chunk_data['content'],
                    'source': pdf_file.name,
                    'chunk_id': chunk_data['chunk_id'],
                    'word_count': chunk_data['word_count'],
                    'sentence_indices': chunk_data['sentence_indices'],
                    'metadata': {
                        'file_path': str(pdf_file),
                        'total_chunks': len(chunks),
                        'total_pages': extracted_data['total_pages'],
                        'chunk_quality_score': self._calculate_chunk_quality(chunk_data['content'])
                    }
                }
                all_documents.append(doc)
        if all_documents:
            print(f"Adding {len(all_documents)} optimized chunks to vector store...")
            self.vector_store.add_documents(all_documents)
            print("Optimized PDF processing complete!")
            self._print_processing_stats(all_documents)
        else:
            print("No documents were successfully processed.")
    
    def _calculate_chunk_quality(self, content: str) -> float:
        """Calculate quality score for chunk"""
        words = content.split()
        if not words:
            return 0.0
        length_score = min(1.0, len(words) / 200)
        sentences = sent_tokenize(content)
        completeness_score = 1.0 if content.rstrip().endswith(('.', '!', '?')) else 0.5
        unique_words = len(set(word.lower() for word in words))
        density_score = unique_words / len(words)
        return (length_score + completeness_score + density_score) / 3
    
    def _print_processing_stats(self, documents: List[Dict]):
        """Print processing statistics"""
        total_chunks = len(documents)
        avg_chunk_size = sum(doc['word_count'] for doc in documents) / total_chunks
        avg_quality = sum(doc['metadata']['chunk_quality_score'] for doc in documents) / total_chunks
        print(f"\n=== Processing Statistics ===")
        print(f"Total chunks: {total_chunks}")
        print(f"Average chunk size: {avg_chunk_size:.1f} words")
        print(f"Average chunk quality: {avg_quality:.3f}")
        print(f"Chunk size range: {min(doc['word_count'] for doc in documents)} - {max(doc['word_count'] for doc in documents)} words")
    
    def enhanced_query_llama32(self, question: str, k: int = None, stream: bool = False, 
                            custom_system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced querying with Llama 3.2-3B via LM Studio with improved conversation context"""
        k = k or self.config['retrieval_k']
        
        if not self.llama32_generator:
            return {
                'question': question,
                'answer': "❌ Llama 3.2-3B model is not available. Please check the model loading.",
                'sources': [],
                'confidence': 0.0
            }
        
        retrieved_docs = self.vector_store.hybrid_search(question, k=k)
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': [],
                'confidence': 0.0
            }
        
        context = self._prepare_enhanced_context(retrieved_docs, question)
        max_context_length = self.config['max_context_length'] // 2
        if len(context) > max_context_length:
            print(f"⚠️ Context truncated to {max_context_length} characters")
            context = context[:max_context_length]
        
        rag_prompt = self.llama32_generator.create_rag_prompt(
            question, context, custom_system_prompt, self.conversation_history
        )
        
        print("🦙 Generating response with Llama 3.2-3B...")
        if stream:
            print("📝 Streaming response:")
            answer = self.llama32_generator.generate_stream(
                rag_prompt,
                max_tokens=self.config['generation_params']['max_new_tokens'],
                temperature=self.config['generation_params']['temperature']
            )
        else:
            answer = self.llama32_generator.generate(
                rag_prompt,
                max_tokens=self.config['generation_params']['max_new_tokens'],
                temperature=self.config['generation_params']['temperature']
            )
        
        # Update conversation history with more details
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'sources': [doc['source'] for doc in retrieved_docs],
            'confidence': self._calculate_answer_confidence(retrieved_docs, answer)
        })
        
        confidence = self._calculate_answer_confidence(retrieved_docs, answer)
        sources = self._prepare_detailed_sources(retrieved_docs)
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context': context,
            'confidence': confidence,
            'retrieval_scores': [doc['final_score'] for doc in retrieved_docs],
            'prompt_used': rag_prompt if self.config.get('include_prompt', False) else None
        }
    
    def _prepare_enhanced_context(self, docs: List[Dict], question: str) -> str:
        """Prepare context with better organization"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_part = f"[Source {i}: {doc['source']}]\n{doc['content']}\n"
            context_parts.append(context_part)
        full_context = "\n".join(context_parts)
        max_length = self.config['max_context_length'] // 2
        if len(full_context) > max_length:
            full_context = full_context[:max_length] + "..."
            print(f"⚠️ Context truncated to {max_length} characters")
        return full_context
    
    def _calculate_answer_confidence(self, docs: List[Dict], answer: str) -> float:
        """Calculate confidence in the answer"""
        if not docs or not answer:
            return 0.0
        avg_score = sum(doc.get('final_score', 0) for doc in docs) / len(docs)
        answer_words = len(answer.split())
        length_factor = min(1.0, answer_words / 50) if answer_words < 200 else max(0.5, 200 / answer_words)
        source_factor = min(1.0, len(set(doc['source'] for doc in docs)) / 3)
        confidence = (avg_score + length_factor + source_factor) / 3
        return min(1.0, confidence)
    
    def _prepare_detailed_sources(self, docs: List[Dict]) -> List[Dict]:
        """Prepare detailed source information"""
        sources = []
        for doc in docs:
            sources.append({
                'source': doc['source'],
                'similarity_score': doc.get('similarity_score', 0),
                'final_score': doc.get('final_score', 0),
                'chunk_id': doc['chunk_id'],
                'word_count': doc.get('word_count', 0),
                'quality_score': doc.get('metadata', {}).get('chunk_quality_score', 0),
                'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            })
        return sources

def main():
    config = {
        'chunk_size': 450,
        'overlap_ratio': 0.15,
        'embedding_model': 'all-MiniLM-L6-v2',
        'use_multiple_models': True,
        'llama_model_name': 'Llama-3.2-3B-Instruct-Q4_K_M',
        'base_url': 'http://localhost:1234/v1',
        'device_map': 'auto',
        'max_context_length': 2048,
        'retrieval_k': 5,
        'generation_params': {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        },
        'include_prompt': False
    }
    
    pdf_directory = "./pdfs"
    rag = ImprovedRAGWithLlama32(pdf_directory, config)
    
    print("Loading PDFs with optimizations...")
    rag.load_pdfs_optimized()
    
    print("\n" + "="*60)
    print("🦙 LLAMA 3.2-3B RAG System with Improved Context Ready!")
    print("="*60)
    
    while True:
        question = input("\nYour question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            stream_choice = input("Stream response? (y/n, default=n): ").strip().lower()
            stream = stream_choice in ['y', 'yes']
            
            result = rag.enhanced_query_llama32(question, stream=stream)
            
            if not stream:
                print(f"\n📋 Question: {result['question']}")
                print(f"🦙 Llama 3.2-3B Answer: {result['answer']}")
            
            print(f"\n🎯 Confidence: {result['confidence']:.2f}")
            print(f"📊 Retrieved {len(result['sources'])} sources")
            
            print("\n📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['source']} (Score: {source['final_score']:.3f})")
                print(f"   Preview: {source['content_preview']}")
            
            # Display conversation topic summary
            if rag.conversation_history:
                print("\n🗣️ Recent Conversation Context:")
                relevant_topics = set()
                for entry in rag.conversation_history[-3:]:
                    if any(term in entry['question'].lower() for term in ['pii', 'personally', 'identifiable', 'information']):
                        relevant_topics.add("PII and data privacy")
                    # Add other topic detection as needed
                if relevant_topics:
                    print(f"Current topic(s): {', '.join(relevant_topics)}")
                print("Recent exchanges:")
                for i, entry in enumerate(rag.conversation_history[-2:], 1):
                    print(f"{i}. Q: {entry['question']}")
                    print(f"   A: {entry['answer'][:100]}...")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def test_llama32_rag():
    config = {
        'chunk_size': 300,
        'overlap_ratio': 0.2,
        'embedding_model': 'all-MiniLM-L6-v2',
        'use_multiple_models': False,
        'llama_model_name': 'C:/Users/Yojith/.lmstudio/models/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
        'base_url': 'http://127.0.0.1:1234/v1',
        'device_map': 'auto',
        'max_context_length': 1024,
        'retrieval_k': 3,
        'generation_params': {
            'max_new_tokens': 256,
            'temperature': 0.6,
            'top_p': 0.85,
            'do_sample': True
        }
    }
    
    test_questions = [
        "What is PII?",
        "Where is PII used?",
        "Is PII protection effective?"
    ]
    
    try:
        pdf_directory = "./test_pdfs"
        rag = ImprovedRAGWithLlama32(pdf_directory, config)
        
        print("Loading test PDFs...")
        rag.load_pdfs_optimized()
        
        print("\n" + "="*50)
        print("🧪 TESTING LLAMA 3.2-3B RAG")
        print("="*50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test {i} ---")
            result = rag.enhanced_query_llama32(question)
            
            print(f"Q: {result['question']}")
            print(f"A: {result['answer'][:200]}...")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {len(result['sources'])}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def benchmark_rag_performance():
    configurations = [
        {
            'name': 'Standard Config',
            'config': {
                'chunk_size': 400,
                'overlap_ratio': 0.15,
                'use_multiple_models': False,
                'retrieval_k': 5
            }
        },
        {
            'name': 'High Quality Config',
            'config': {
                'chunk_size': 300,
                'overlap_ratio': 0.2,
                'use_multiple_models': True,
                'retrieval_k': 7
            }
        },
        {
            'name': 'Fast Config',
            'config': {
                'chunk_size': 500,
                'overlap_ratio': 0.1,
                'use_multiple_models': False,
                'retrieval_k': 3
            }
        }
    ]
    
    test_questions = [
        "What are the main findings?",
        "Explain the methodology used.",
        "What are the conclusions?"
    ]
    
    print("🏁 RAG Performance Benchmark")
    print("="*50)
    
    for config_info in configurations:
        print(f"\n🔧 Testing: {config_info['name']}")
        
        try:
            full_config = {
                'llama_model_name': 'Llama-3.2-3B-Instruct-Q4_K_M',
                'base_url': 'http://127.0.0.1:1234/v1',
                'max_context_length': 2048,
                'temperature': 0.7,
                'top_p': 0.9,
            }
            full_config.update(config_info['config'])
            
            import time
            start_time = time.time()
            
            rag = ImprovedRAGWithLlama32("./pdfs", full_config)
            rag.load_pdfs_optimized()
            
            setup_time = time.time() - start_time
            
            query_times = []
            confidences = []
            
            for question in test_questions:
                query_start = time.time()
                result = rag.enhanced_query_llama32(question)
                query_time = time.time() - query_start
                query_times.append(query_time)
                confidences.append(result['confidence'])
            
            print(f"   Setup time: {setup_time:.2f}s")
            print(f"   Avg query time: {sum(query_times)/len(query_times):.2f}s")
            print(f"   Avg confidence: {sum(confidences)/len(confidences):.3f}")
            
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")

def create_custom_system_prompts():
    prompts = {
        'academic': """You are an expert academic researcher. Provide detailed, scholarly responses 
        based on the given context. Include specific citations and acknowledge limitations in the source material. 
        Focus on accuracy, nuance, and comprehensive analysis.""",
        'business': """You are a business analyst providing insights from company documents. 
        Focus on actionable information, key metrics, trends, and strategic implications. 
        Present information in a clear, executive-friendly format.""",
        'technical': """You are a technical expert explaining complex concepts. 
        Break down technical information into clear, understandable explanations. 
        Provide step-by-step guidance where appropriate and highlight important technical details.""",
        'legal': """You are a legal assistant analyzing legal documents. 
        Focus on key legal points, relevant statutes, precedents, and potential implications. 
        Be precise with legal terminology and clearly distinguish between facts and interpretations.""",
        'medical': """You are a medical information specialist. 
        Provide accurate medical information while clearly stating that this is for informational purposes only 
        and should not replace professional medical advice. Focus on evidence-based information."""
    }
    return prompts

def advanced_rag_demo():
    config = {
        'chunk_size': 500,
        'overlap_ratio': 0.15,
        'embedding_model': 'all-MiniLM-L6-v2',
        'use_multiple_models': True,
        'llama_model_name': 'C:/Users/Yojith/.lmstudio/models/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
        'base_url': 'http://127.0.0.1:1234/v1',
        'device_map': 'auto',
        'max_context_length': 2048,
        'retrieval_k': 5,
        'generation_params': {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        },
        'include_prompt': True
    }
    
    rag = ImprovedRAGWithLlama32("./pdfs", config)
    rag.load_pdfs_optimized()
    
    custom_prompts = create_custom_system_prompts()
    
    print("\n🚀 Advanced RAG Features Demo")
    print("="*50)
    
    test_question = "What are the main research findings mentioned in the documents?"
    
    for prompt_type, system_prompt in custom_prompts.items():
        print(f"\n--- {prompt_type.upper()} STYLE ---")
        result = rag.enhanced_query_llama32(
            test_question, 
            custom_system_prompt=system_prompt
        )
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Confidence: {result['confidence']:.3f}")
    
    print(f"\n--- STREAMING DEMO ---")
    print("Question: What methodology was used in the research?")
    result = rag.enhanced_query_llama32(
        "What methodology was used in the research?",
        stream=True
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama 3.2-3B RAG System with Improved Context")
    parser.add_argument("--mode", choices=["interactive", "test", "benchmark", "demo"], 
                       default="interactive", help="Mode to run")
    parser.add_argument("--pdf-dir", default="./pdfs", help="PDF directory path")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        main()
    elif args.mode == "test":
        test_llama32_rag()
    elif args.mode == "benchmark":
        benchmark_rag_performance()
    elif args.mode == "demo":
        advanced_rag_demo()
    else:
        print("Invalid mode selected")