import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import requests
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import logging
import time
import json
import subprocess
import streamlit as st
from torch import autocast
import torch

# Fix asyncio on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration settings with defaults and validation."""
    
    DEFAULT_CONFIG = {
        'chunk_size': 300,
        'overlap_ratio': 0.15,
        'embedding_model': 'all-MiniLM-L6-v2',
        'use_multiple_models': False,
        'qwen_model_name': 'Qwen-3-14B',
        'base_url': 'http://127.0.0.1:1234/v1',
        'device_map': 'auto',
        'max_context_length': 8192,
        'retrieval_k': 5,
        'embedding_batch_size': 64,
        'history_file': 'conversation_history',
        'upload_dir': 'uploads',
        'generation_params': {
            'max_new_tokens': 2048,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        },
        'include_prompt': False
    }

    @staticmethod
    def get_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Returns validated configuration with optional overrides."""
        config = ConfigManager.DEFAULT_CONFIG.copy()
        if overrides:
            config.update(overrides)
        ConfigManager._validate_config(config)
        return config

    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        """Validates configuration parameters."""
        if config['chunk_size'] < 100 or config['chunk_size'] > 1000:
            raise ValueError("Chunk size must be between 100 and 1000")
        if not 0 <= config['overlap_ratio'] <= 0.5:
            raise ValueError("Overlap ratio must be between 0 and 0.5")
        if config['retrieval_k'] < 1:
            raise ValueError("Retrieval k must be positive")
        if config['embedding_batch_size'] < 1:
            raise ValueError("Embedding batch size must be positive")
        if config['history_file'].endswith('.json'):
            config['history_file'] = config['history_file'][:-5]
        if not Path(config['upload_dir']).is_dir():
            Path(config['upload_dir']).mkdir(exist_ok=True)

class OptimizedPDFProcessor:
    """Processes PDF files with optimized text extraction."""

    def __init__(self):
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english'))

    def _ensure_nltk_resources(self):
        """Ensures required NLTK resources are available."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extracts text from PDF while preserving structure."""
        try:
            with fitz.open(pdf_path) as doc:
                full_text = []
                pages_text = []
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text()
                    pages_text.append({
                        'page_num': page_num,
                        'text': text,
                        'word_count': len(text.split())
                    })
                    full_text.append(f"\n[Page {page_num}]\n{text}")
                return {
                    'full_text': ''.join(full_text),
                    'pages': pages_text,
                    'total_pages': len(pages_text),
                    'total_words': sum(p['word_count'] for p in pages_text)
                }
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {'full_text': '', 'pages': [], 'total_pages': 0, 'total_words': 0}

    def clean_text(self, text: str) -> str:
        """Cleans extracted text with optimized preprocessing."""
        text = re.sub(r'\[Page \d+\]', '', text)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        return text.strip()

class AdaptiveChunker:
    """Implements adaptive chunking based on content structure."""

    def __init__(self, chunk_size: int = 400, overlap_ratio: float = 0.15):
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.min_chunk_size = 100
        self.max_chunk_size = 800

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Creates semantically coherent chunks."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_sentences = []

        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())
            target_chunk_size = self._get_dynamic_chunk_size(sentence)

            if len(current_chunk.split()) + sentence_words <= target_chunk_size or \
               len(current_chunk.split()) < self.min_chunk_size:
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

    def _get_dynamic_chunk_size(self, sentence: str) -> int:
        """Calculates dynamic chunk size based on sentence complexity."""
        words = sentence.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        if avg_word_length > 6:
            return int(self.chunk_size * 0.8)
        elif avg_word_length < 4:
            return int(self.chunk_size * 1.2)
        return self.chunk_size

    def _get_overlap_sentences(self, sentences: List[str], current_indices: List[int]) -> List[str]:
        """Gets sentences for chunk overlap."""
        if not current_indices:
            return []
        overlap_count = max(1, int(len(current_indices) * self.overlap_ratio))
        start_idx = max(0, current_indices[-1] - overlap_count + 1)
        return sentences[start_idx:current_indices[-1] + 1]

class EnhancedVectorStore:
    """Optimized vector store for document embeddings and search."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 64):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            logger.warning("No GPU available, falling back to CPU for embeddings. Performance may be slower.")
        else:
            logger.info(f"Using GPU ({self.device}) for embeddings")
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.documents = []
        self.index = None
        self.embeddings = None

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for texts using GPU with mixed precision."""
        logger.info(f"Generating embeddings on {self.device} with batch size {self.batch_size}...")
        embeddings = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                with autocast(device_type="cuda"):
                    batch_embeddings = self.embedder.encode(
                        batch,
                        batch_size=self.batch_size,
                        show_progress_bar=True,
                        convert_to_tensor=True,
                        device=self.device
                    )
                embeddings.append(batch_embeddings.cpu().numpy())
            return np.vstack(embeddings)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory, falling back to CPU for this batch")
                self.embedder.to('cpu')
                embeddings = self.embedder.encode(
                    texts,
                    batch_size=self.batch_size // 2,
                    show_progress_bar=True,
                    convert_to_tensor=False
                )
                self.embedder.to(self.device)
                return embeddings
            raise e

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Adds documents to vector store."""
        self.documents.extend(documents)
        texts = [doc['content'] for doc in documents]
        new_embeddings = self.embed_documents(texts)
        self.embeddings = np.vstack([self.embeddings, new_embeddings]) if self.embeddings is not None else new_embeddings
        self._build_index()

    def _build_index(self):
        """Builds FAISS index for embeddings."""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Performs hybrid search with reranking."""
        if not self.index:
            return []
        query_embedding = self.embedder.encode([query], convert_to_tensor=True, device=self.device)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        scores, indices = self.index.search(query_embedding.cpu().numpy().astype('float32'), min(k * 2, len(self.documents)))
        results = [
            {**self.documents[idx], 'similarity_score': float(score)}
            for score, idx in zip(scores[0], indices[0]) if idx < len(self.documents)
        ]
        return self._rerank_results(results, query)[:k]

    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Reranks search results with keyword bonus."""
        content_groups = {}
        for result in results:
            content_key = result['content'][:100]
            content_groups.setdefault(content_key, []).append(result)

        unique_results = []
        for group in content_groups.values():
            best_result = max(group, key=lambda x: x['similarity_score'])
            best_result['final_score'] = best_result['similarity_score'] + self._calculate_keyword_bonus(query, best_result['content'])
            unique_results.append(best_result)

        return sorted(unique_results, key=lambda x: x['final_score'], reverse=True)

    def _calculate_keyword_bonus(self, query: str, content: str) -> float:
        """Calculates bonus score based on keyword overlap."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        return len(query_words.intersection(content_words)) * 0.05

class Qwen14BGenerator:
    """Handles text generation with Qwen-3 14B via LM Studio."""

    def __init__(self, config: Dict):
        self.model_path = config['qwen_model_name']
        self.base_url = config['base_url']
        self.max_context_length = config['max_context_length']
        self.temperature = config['generation_params']['temperature']
        self.top_p = config['generation_params']['top_p']
        self.use_chat_endpoint = True
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initializes connection to LM Studio server."""
        try:
            logger.info(f"Connecting to LM Studio at {self.base_url}")
            response = requests.get(f"{self.base_url}/models")
            if response.status_code != 200:
                raise Exception(f"LM Studio server not running: {response.status_code}")
            self.model = OpenAI(base_url=self.base_url, api_key='not-needed')
            logger.info(f"Testing connection to {self.model_path}...")
            try:
                test_response = self.model.chat.completions.create(
                    model=self.model_path,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=20,
                    temperature=self.temperature
                )
            except Exception:
                logger.info("Falling back to completions endpoint")
                self.use_chat_endpoint = False
                test_response = self.model.completions.create(
                    model=self.model_path,
                    prompt="Test",
                    max_tokens=20,
                    temperature=self.temperature
                )
            logger.info("Successfully connected to LM Studio")
        except Exception as e:
            logger.error(f"Failed to connect to LM Studio: {str(e)}")
            raise

    def create_rag_prompt(self, question: str, context: str, system_prompt: Optional[str] = None,
                         history: List[Dict[str, str]] = None) -> str:
        """Creates optimized RAG prompt with source citation instructions."""
        system_prompt = system_prompt or (
            "You are a cybersecurity expert and educator with deep knowledge of information security, ethical hacking, network security, and security best practices. "
            "- Always prioritize ethical and legal approaches to cybersecurity"
            "- Explain concepts clearly for different skill levels"
            "- Provide practical, actionable guidance"
            "- Include real-world context and examples"
            "- Emphasize defense over offense"
            "- Do not provide specific instructions for malicious attacks"
            "- Focus on defensive strategies and threat awareness"
            "- Explain vulnerabilities in educational context only"
            "- Redirect harmful requests toward legitimate security practices"
            "- Structure responses with clear sections"
            "- Include step-by-step explanations when appropriate"
            "- Provide relevant examples and case studies"
            "- Suggest additional resources for deeper learning"
            "- Balance technical depth with accessibility"
            "Encourage continuous learning and professional development"
            "Emphasize the importance of authorization and consent"
            "Cite sources explicitly in your response using the format [Source X: filename]. "
            "If you have cited a source already, no need to cite it again "
            "If the context doesn't contain enough information, say so clearly."
        )
        
        history_text = ""
        if history:
            max_history_length = self.max_context_length // 3
            current_length = 0
            for entry in reversed(history[-5:]):
                summary = f"Q: {entry['question']}\nA: {entry['answer'][:100] + '...' if len(entry['answer']) > 100 else entry['answer']}\n"
                if 'semgrep_findings' in entry:
                    summary += f"Semgrep Findings: {len(entry['semgrep_findings'])} issues detected\n"
                if current_length + len(summary) <= max_history_length:
                    history_text = summary + history_text
                    current_length += len(summary)
                else:
                    break

        resolved_question = question
        if history and 'it' in question.lower().split():
            last_topic = history[-1]['question'].lower()
            if 'pii' in last_topic or 'personally identifiable information' in last_topic:
                resolved_question = re.sub(r'\bit\b', 'PII', question, flags=re.IGNORECASE)
        prompt = (system_prompt + "\n\nContext:\n" + context + "\n\n" +
                 ("Conversation History:\n" + history_text if history_text else "") +
                 "Question: " + resolved_question + "\nAnswer:")
        return prompt

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: Optional[float] = None) -> str:
        """Generates text using LM Studio."""
        try:
            params = {
                'model': self.model_path,
                'max_tokens': max_tokens,
                'temperature': temperature or self.temperature,
                'top_p': self.top_p
            }
            if self.use_chat_endpoint:
                response = self.model.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    **params
                )
                answer = response.choices[0].message.content.strip()
            else:
                response = self.model.completions.create(
                    prompt=prompt,
                    **params
                )
                answer = response.choices[0].text.strip()
        
            if not answer:
                logger.warning("Empty response from model")
                return "No response generated."
            logger.info(f"Generated response of length {len(answer)} characters")
            return answer
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"Error generating response: {str(e)}"

class ImprovedRAGWithQwen14B:
    """Enhanced RAG system with Qwen-3 14B and Semgrep integration."""

    def __init__(self, pdf_directory: str, config: Dict[str, Any] = None):
        self.pdf_directory = Path(pdf_directory)
        self.config = ConfigManager.get_config(config)
        self.pdf_processor = OptimizedPDFProcessor()
        self.chunker = AdaptiveChunker(self.config['chunk_size'], self.config['overlap_ratio'])
        self.vector_store = EnhancedVectorStore(self.config['embedding_model'], self.config['embedding_batch_size'])
        self.qwen_generator = Qwen14BGenerator(self.config)
        self.conversation_history = []
        self.history_file = Path(f"{self.config['history_file']}.json")
        self.upload_dir = Path(self.config['upload_dir'])
        self.upload_dir.mkdir(exist_ok=True)
        self._load_conversation_history()

    def _load_conversation_history(self):
        """Loads conversation history from JSON file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info(f"Loaded {len(self.conversation_history)} entries from {self.history_file}")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in {self.history_file}, initializing empty history")
                self.conversation_history = []
            except Exception as e:
                logger.error(f"Error loading {self.history_file}: {str(e)}")
                self.conversation_history = []
        else:
            logger.info(f"No history file found at {self.history_file}, initializing empty history")
            self.conversation_history = []

    def _save_conversation_history(self):
        """Saves conversation history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.conversation_history)} entries to {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving {self.history_file}: {str(e)}")

    def load_pdfs(self):
        """Loads and processes PDF files."""
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return

        logger.info(f"Processing {len(pdf_files)} PDF files...")
        all_documents = []

        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}")
            extracted_data = self.pdf_processor.extract_text(str(pdf_file))
            if not extracted_data['full_text'].strip():
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue

            clean_text = self.pdf_processor.clean_text(extracted_data['full_text'])
            chunks = self.chunker.chunk_text(clean_text)
            logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")

            for chunk_data in chunks:
                all_documents.append({
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
                })

        if all_documents:
            logger.info(f"Adding {len(all_documents)} chunks to vector store...")
            self.vector_store.add_documents(all_documents)
            self._log_processing_stats(all_documents)
        else:
            logger.warning("No documents processed.")

    def _calculate_chunk_quality(self, content: str) -> float:
        """Calculates quality score for a chunk."""
        words = content.split()
        if not words:
            return 0.0
        length_score = min(1.0, len(words) / 200)
        completeness_score = 1.0 if content.rstrip().endswith(('.', '!', '?')) else 0.5
        unique_words = len(set(word.lower() for word in words))
        density_score = unique_words / len(words)
        return (length_score + completeness_score + density_score) / 3

    def _log_processing_stats(self, documents: List[Dict]):
        """Logs processing statistics."""
        total_chunks = len(documents)
        avg_chunk_size = sum(doc['word_count'] for doc in documents) / total_chunks
        avg_quality = sum(doc['metadata']['chunk_quality_score'] for doc in documents) / total_chunks
        logger.info(f"\n=== Processing Statistics ===\n"
                    f"Total chunks: {total_chunks}\n"
                    f"Average chunk size: {avg_chunk_size:.1f} words\n"
                    f"Average chunk quality: {avg_quality:.3f}\n"
                    f"Chunk size range: {min(doc['word_count'] for doc in documents)} - "
                    f"{max(doc['word_count'] for doc in documents)} words")

    def scan_code_with_semgrep(self, code_content: str, file_extension: str, rules: str = "auto") -> List[Dict]:
        """Scans uploaded code with Semgrep and returns findings."""
        try:
            timestamp = int(time.time())
            temp_file = self.upload_dir / f"uploaded_code_{timestamp}.{file_extension}"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code_content)

            logger.info(f"Running Semgrep scan on {temp_file} with rules: {rules}")
            result = subprocess.run(
                ["semgrep", "--config", rules, "--json", str(temp_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False
            )

            if result.returncode != 0:
                logger.error(f"Semgrep failed with exit code {result.returncode}: {result.stderr}")
                temp_file.unlink()
                return [{'error': f"Semgrep scan failed: {result.stderr}"}]

            findings = []
            try:
                semgrep_output = json.loads(result.stdout)
                for finding in semgrep_output.get('results', []):
                    findings.append({
                        'rule_id': finding.get('check_id', 'unknown'),
                        'message': finding.get('extra', {}).get('message', ''),
                        'severity': finding.get('extra', {}).get('severity', 'UNKNOWN'),
                        'file_path': finding.get('path', str(temp_file)),
                        'line': finding.get('start', {}).get('line', 0),
                        'code_snippet': finding.get('extra', {}).get('lines', ''),
                        'cwe': finding.get('extra', {}).get('metadata', {}).get('cwe', []),
                        'owasp': finding.get('extra', {}).get('metadata', {}).get('owasp', [])
                    })
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Semgrep JSON output: {str(e)}")
                findings.append({'error': f"Failed to parse Semgrep output: {str(e)}"})

            temp_file.unlink()
            logger.info(f"Semgrep scan completed: {len(findings)} issues found")
            return findings

        except subprocess.CalledProcessError as e:
            logger.error(f"Semgrep scan failed: {e.stderr}")
            temp_file.unlink() if temp_file.exists() else None
            return [{'error': f"Semgrep scan failed: {e.stderr}"}]
        except Exception as e:
            logger.error(f"Semgrep scan failed: {str(e)}")
            temp_file.unlink() if temp_file.exists() else None
            return [{'error': f"Semgrep scan failed: {str(e)}"}]

    def query(self, question: str, k: int = None, stream: bool = True,
              system_prompt: Optional[str] = None, code_content: Optional[str] = None,
              file_extension: Optional[str] = None) -> Dict[str, Any]:
        """Executes query with Qwen-3 14B, including Semgrep findings if code is provided."""
        k = k or self.config['retrieval_k']
        if not self.qwen_generator.model:
            return {
                'question': question,
                'answer': "Qwen-3 14B model unavailable.",
                'sources': [],
                'confidence': 0.0,
                'semgrep_findings': []
            }

        semgrep_findings = []
        if code_content and file_extension:
            semgrep_findings = self.scan_code_with_semgrep(code_content, file_extension)
            context = f"Semgrep Analysis Results:\n{json.dumps(semgrep_findings, indent=2)}\n\n"
        else:
            context = ""

        retrieved_docs = self.vector_store.search(question, k)
        if retrieved_docs:
            context += self._prepare_context(retrieved_docs, question)
        elif not semgrep_findings:
            context += "No relevant information found in PDFs."

        if len(context) > self.config['max_context_length']:
            context = context[:self.config['max_context_length']]
            logger.warning(f"Context truncated to {self.config['max_context_length']} characters")

        prompt = self.qwen_generator.create_rag_prompt(question, context, system_prompt, self.conversation_history)
        logger.info("Generating response...")
        answer = self.qwen_generator.generate(
            prompt,
            max_tokens=self.config['generation_params']['max_new_tokens'],
            temperature=self.config['generation_params']['temperature']
        )

        confidence = self._calculate_answer_confidence(retrieved_docs, answer)
        sources = self._prepare_sources(retrieved_docs)
        history_entry = {
            'question': question,
            'answer': answer,
            'sources': [doc['source'] for doc in retrieved_docs],
            'confidence': confidence,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        if semgrep_findings:
            history_entry['semgrep_findings'] = semgrep_findings

        self.conversation_history.append(history_entry)
        self._save_conversation_history()

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'retrieval_scores': [doc['final_score'] for doc in retrieved_docs],
            'prompt_used': prompt if self.config.get('include_prompt') else None,
            'semgrep_findings': semgrep_findings
        }

    def _prepare_context(self, docs: List[Dict], question: str) -> str:
        """Prepares organized context from retrieved documents."""
        max_length = self.config['max_context_length'] // 2
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            part = f"[Source {i}: {doc['source']}, Chunk {doc['chunk_id']}]\n{doc['content']}\n"
            if current_length + len(part) <= max_length:
                context_parts.append(part)
                current_length += len(part)
            else:
                remaining_length = max_length - current_length
                if remaining_length > 50:
                    context_parts.append(part[:remaining_length] + "...")
                logger.warning(f"Context truncated at {max_length} characters; {len(docs) - i + 1} documents excluded")
                break
        
        return "\n".join(context_parts)

    def _calculate_answer_confidence(self, docs: List[Dict], answer: str) -> float:
        """Calculates confidence score for the answer."""
        if not docs or not answer:
            return 0.0
        avg_score = sum(doc.get('final_score', 0) for doc in docs) / len(docs)
        answer_words = len(answer.split())
        length_factor = min(1.0, answer_words / 50) if answer_words < 200 else max(0.5, 200 / answer_words)
        source_factor = min(1.0, len(set(doc['source'] for doc in docs)) / 3)
        return min(1.0, (avg_score + length_factor + source_factor) / 3)

    def _prepare_sources(self, docs: List[Dict]) -> List[Dict]:
        """Prepares detailed source information."""
        return [{
            'source': doc['source'],
            'similarity_score': doc.get('similarity_score', 0),
            'final_score': doc.get('final_score', 0),
            'chunk_id': doc['chunk_id'],
            'word_count': doc.get('word_count', 0),
            'quality_score': doc.get('metadata', {}).get('chunk_quality_score', 0),
            'content_preview': doc['content'][:200] + ("..." if len(doc['content']) > 200 else "")
        } for doc in docs]


@st.cache_resource
def initialize_rag(pdf_dir: str, history_file: str) -> ImprovedRAGWithQwen14B:
    """Initialize and cache the RAG instance with PDF loading."""
    logger.info(f"Initializing RAG instance for {history_file}")
    config = ConfigManager.get_config({'history_file': history_file})
    rag = ImprovedRAGWithQwen14B(pdf_dir, config)
    rag.load_pdfs()
    return rag


def main():
    """Streamlit frontend for the RAG system with Semgrep integration."""
    st.set_page_config(page_title="Qwen-3 14B RAG with Semgrep", layout="wide")
    st.title("🤖 Qwen-3 14B RAG System with Semgrep")
    st.markdown("""
    Welcome to the RAG system! Ask cybersecurity questions or upload code for Semgrep analysis.
    - Select or create a conversation.
    - Enter a question or upload a code file (e.g., .py, .js, .java).
    - View responses, Semgrep findings, and sources below.
    """)

    # Initialize session state
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False

    # Conversation selection
    conversation_dir = Path("conversations")
    conversation_dir.mkdir(exist_ok=True)
    existing_chats = sorted([f.stem for f in conversation_dir.glob("*.json")])
    conversation_options = ["Create new conversation"] + existing_chats

    st.subheader("📬 Select or Create Conversation")
    selected_conversation = st.selectbox("Choose a conversation:", conversation_options, key="conversation_select")

    # Handle conversation selection or creation
    if selected_conversation != st.session_state.current_conversation:
        st.session_state.current_conversation = selected_conversation
        st.session_state.rag_initialized = False
        st.session_state.history = []

    if selected_conversation == "Create new conversation":
        new_conversation_name = st.text_input("Enter a name for the new conversation:", key="new_conversation_name")
        if new_conversation_name:
            new_conversation_name = re.sub(r'[^\w\s-]', '', new_conversation_name).replace(' ', '_')
            if not new_conversation_name:
                st.error("Invalid name. Use alphanumeric characters, spaces, or hyphens.")
            elif (conversation_dir / f"{new_conversation_name}.json").exists():
                st.error(f"A conversation named '{new_conversation_name}' already exists.")
            else:
                conversation_file = str(conversation_dir / f"{new_conversation_name}.json")
                st.session_state.rag = initialize_rag("./pdfs", conversation_file)
                st.session_state.rag_initialized = True
                st.session_state.history = st.session_state.rag.conversation_history
                st.success(f"Created new conversation: {new_conversation_name}")
    elif selected_conversation:
        conversation_file = str(conversation_dir / f"{selected_conversation}.json")
        if not st.session_state.rag_initialized:
            st.session_state.rag = initialize_rag("./pdfs", conversation_file)
            st.session_state.rag_initialized = True
            st.session_state.history = st.session_state.rag.conversation_history
            st.success(f"Loaded conversation: {selected_conversation}")

    # Input section
    if st.session_state.rag_initialized:
        st.subheader("❓ Ask a Question or Upload Code")
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_area("Enter your question:", height=100, key="question_input")
        with col2:
            uploaded_file = st.file_uploader("Upload code file:", type=['py', 'js', 'java', 'c', 'cpp', 'go'], key="file_uploader")

        if st.button("Submit", key="submit_button"):
            if not question and not uploaded_file:
                st.error("Please enter a question or upload a code file.")
            else:
                code_content = None
                file_extension = None
                if uploaded_file:
                    code_content = uploaded_file.read().decode('utf-8', errors='replace')
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    question = question or "Analyze the uploaded code for vulnerabilities and provide recommendations."

                try:
                    with st.spinner("Processing..."):
                        result = st.session_state.rag.query(question, code_content=code_content, file_extension=file_extension)
                        st.session_state.history = st.session_state.rag.conversation_history

                    # Display response
                    st.subheader("📝 Response")
                    st.markdown(f"**Question:** {result['question']}")
                    st.markdown(f"**Answer:** {result['answer']}")
                    st.markdown(f"**Confidence:** {result['confidence']:.2f}")

                    if result['semgrep_findings']:
                        st.subheader("🔍 Semgrep Findings")
                        for i, finding in enumerate(result['semgrep_findings'], 1):
                            if 'error' in finding:
                                st.error(f"{i}. Error: {finding['error']}")
                            else:
                                st.markdown(f"""
                                **{i}. Rule: {finding['rule_id']}**
                                - **Severity:** {finding['severity']}
                                - **File:** {finding['file_path']} (Line {finding['line']})
                                - **Message:** {finding['message']}
                                - **Code:** `{finding['code_snippet']}`
                                - **CWE:** {', '.join(finding['cwe']) if finding['cwe'] else 'N/A'}
                                - **OWASP:** {', '.join(finding['owasp']) if finding['owasp'] else 'N/A'}
                                """)

                    if result['sources']:
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"""
                            {i}. **{source['source']}** (Chunk {source['chunk_id']}, Score: {source['final_score']:.3f})
                            - {source['content_preview']}
                            """)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Query error: {str(e)}")

        # Display conversation history
        st.subheader("📜 Conversation History")
        if st.session_state.history:
            for entry in reversed(st.session_state.history[-5:]):
                with st.expander(f"Q: {entry['question']} ({entry['timestamp']})"):
                    st.markdown(f"**Answer:** {entry['answer']}")
                    st.markdown(f"**Confidence:** {entry['confidence']:.2f}")
                    if 'semgrep_findings' in entry and entry['semgrep_findings']:
                        st.markdown("**Semgrep Findings:**")
                        for i, finding in enumerate(entry['semgrep_findings'], 1):
                            if 'error' in finding:
                                st.markdown(f"{i}. Error: {finding['error']}")
                            else:
                                st.markdown(f"""
                                {i}. **Rule:** {finding['rule_id']}
                                - **Severity:** {finding['severity']}
                                - **File:** {finding['file_path']} (Line {finding['line']})
                                - **Message:** {finding['message']}
                                - **Code:** `{finding['code_snippet']}`
                                """)
                    if entry['sources']:
                        st.markdown("**Sources:**")
                        for source in entry['sources']:
                            st.markdown(f"- {source}")
        else:
            st.write("No conversation history yet.")

if __name__ == "__main__":
    main()