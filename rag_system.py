import os
import chromadb
from pypdf import PdfReader
import requests
import json
from typing import List, Dict
import re
from collections import Counter
import math

class RAGSystem:
    def __init__(self, model_name: str = "gemma3:270m"):
        """Initialize RAG system with simple TF-IDF for retrieval"""
        self.model_name = model_name
        self.chunks = []
        self.ollama_url = "http://localhost:11434/api"
        
        print(f"✅ RAG System initialized with model: {model_name}")
        
    def load_document(self, file_path: str) -> List[str]:
        """Load and chunk document (PDF or TXT)"""
        chunks = []
        
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Split into chunks of ~500 characters
        chunk_size = 500
        overlap = 50
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        print(f"✅ Document loaded: {len(chunks)} chunks created")
        return chunks
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def compute_tfidf(self, query_tokens: List[str], doc_tokens: List[str], all_docs_tokens: List[List[str]]) -> float:
        """Compute TF-IDF similarity score"""
        # Term frequency in query
        query_tf = Counter(query_tokens)
        
        # Term frequency in document
        doc_tf = Counter(doc_tokens)
        
        # Document frequency (how many docs contain each term)
        df = Counter()
        for doc in all_docs_tokens:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1
        
        # Compute TF-IDF score
        score = 0.0
        num_docs = len(all_docs_tokens)
        
        for term in query_tf:
            if term in doc_tf:
                tf = doc_tf[term] / len(doc_tokens) if len(doc_tokens) > 0 else 0
                idf = math.log(num_docs / (df[term] + 1)) if df[term] > 0 else 0
                score += tf * idf * query_tf[term]
        
        return score
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant chunks using TF-IDF"""
        if not self.chunks:
            return []
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        # Tokenize all chunks
        all_tokens = [self.tokenize(chunk) for chunk in self.chunks]
        
        # Compute scores for each chunk
        scores = []
        for i, doc_tokens in enumerate(all_tokens):
            score = self.compute_tfidf(query_tokens, doc_tokens, all_tokens)
            scores.append((score, i))
        
        # Sort by score and get top K
        scores.sort(reverse=True)
        top_indices = [idx for score, idx in scores[:top_k] if score > 0]
        
        # If no good matches, return first few chunks
        if not top_indices:
            top_indices = list(range(min(top_k, len(self.chunks))))
        
        return [self.chunks[i] for i in top_indices]
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using Ollama"""
        # Construct RAG prompt
        context_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
        
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context below. If the answer cannot be found in the context, say "I cannot find this information in the provided document."

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
        
        # Call Ollama API
        try:
            print(f"🤔 Generating answer...")
            response = requests.post(
                f"{self.ollama_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'No response generated')
                print(f"✅ Answer generated")
                return answer
            else:
                error_msg = f"Error: Ollama returned status {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text}"
                print(error_msg)
                return error_msg
        
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model took too long to respond."
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running with CPU mode."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def query(self, question: str) -> Dict:
        """Main RAG query pipeline"""
        # Step 1: Retrieve relevant context
        print(f"🔍 Searching for relevant context...")
        context = self.retrieve_context(question, top_k=3)
        
        if not context:
            return {
                "answer": "No documents have been indexed yet.",
                "context": []
            }
        
        print(f"✅ Found {len(context)} relevant chunks")
        
        # Step 2: Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            "answer": answer,
            "context": context
        }
    
    def ingest_document(self, file_path: str):
        """Complete ingestion pipeline"""
        print(f"📄 Starting document ingestion: {file_path}")
        
        # Load and chunk
        self.chunks = self.load_document(file_path)
        
        print("✅ Document ingestion complete!")

# Example usage
if __name__ == "__main__":
    rag = RAGSystem()
    
    # Ingest a document
    rag.ingest_document("your_document.pdf")
    
    # Test queries
    result = rag.query("What is this document about?")
    print(f"\nAnswer: {result['answer']}")
    
    # Test hallucination prevention
    result = rag.query("What is the weather today?")
    print(f"\nAnswer: {result['answer']}")