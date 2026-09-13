# 🐍 Python Q&A Intelligent Chatbot

An interactive conversational assistant built to provide instant, structured answers to Python programming questions using document retrieval and cosine similarity search.

---

## 📌 Features
- **Semantic Retrieval:** Matches user queries against verified Python technical documentation and question banks.
- **Contextual Query Processing:** Cleans and normalizes developer input via custom NLP pipelines.
- **Interactive Chat Interface:** Session-state driven conversation UI powered by Streamlit.
- **Confidence Scoring:** Validates match relevance before returning solutions to minimize hallucinations.

---

## 🛠️ Tech Stack
- **Language:** Python
- **NLP / Information Retrieval:** scikit-learn, NLTK
- **UI Framework:** Streamlit
- **Data Handling:** pandas, numpy

---

## ⚙️ Installation & Setup

```bash
git clone [https://github.com/Sinchana-175/python-qa-chatbot.git](https://github.com/Sinchana-175/python-qa-chatbot.git)
cd python-qa-chatbot
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
