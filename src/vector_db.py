import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

class QAEngine:
    def __init__(self, kb_path='data/python_kb.json', api_key=None):
        self.kb_path = kb_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb_data = []
        self.index = None
        self._load_and_build()
        
        # Initialize Gemini Client
        # It looks for GEMINI_API_KEY environment variable if not passed directly
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def _load_and_build(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}")
            
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            self.kb_data = json.load(f)

        texts = [f"Question: {item['question']} Answer: {item['answer']}" for item in self.kb_data]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def get_answer(self, query: str, top_k: int = 2):
        # 1. Retrieve most relevant context from Vector DB
        query_vector = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)

        retrieved_context = ""
        for idx in indices[0]:
            if idx < len(self.kb_data):
                item = self.kb_data[idx]
                retrieved_context += f"Topic: {item['topic']}\nQA Context: {item['question']} -> {item['answer']}\n\n"

        # 2. Construct Prompt for Gemini
        prompt = f"""
        You are an expert Python assistant. Answer the user's question clearly and concisely.
        
        Use the following retrieved context from our Python knowledge base as primary reference if relevant:
        ---
        {retrieved_context}
        ---

        User Question: {query}
        
        If the question is closely answered by the context, summarize and format it cleanly.
        If the context does not fully answer the question, use your general Python knowledge to give a complete, helpful response.
        """

        # 3. Call Gemini Model
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        return response.text