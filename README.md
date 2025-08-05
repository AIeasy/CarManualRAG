# CarManualRAG: PDF-Based QA Pipeline with BM25 and BGE-Reranker

This project builds a question-answering system that reads questions from `questions.json` and finds the best matching content from a car manual PDF using BM25 ranking followed by BGE-based reranking. It then sends the selected content and question to OpenAI's Chat API to generate an answer.

---

## 🚀 Features

- 📄 Extracts text from each page of a PDF using `pdfplumber`
- 🔍 Uses BM25 to find top-3 candidate pages
- 📊 Applies `BAAI/bge-reranker-base` model to rerank candidates
- 🧠 Sends the most relevant content and question to OpenAI GPT to generate a step-by-step answer
- 📝 Saves results in `answer.json` (incrementally) and `full.json` (final output)

---

## 📁 Requirements

- Python 3.8+
- CUDA-enabled GPU for reranker model

### Python Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should contain:

```txt
jieba
numpy
scikit-learn
rank_bm25
requests
tqdm
pdfplumber
torch
transformers
```

---

## 📂 Input Files

- `dataset.pdf`: Car manual PDF
- `questions.json`: A list of questions in JSON format, for example:

```json
[
  {
    "question": "如何更换机油？"
  },
  ...
]
```

---

## ⚙️ How to Run

```bash
python main_cn_git.py
```

### What it does:

1. Parses each page of `dataset.pdf`
2. Loads all questions from `questions.json`
3. For each question:
   - Retrieves top-3 pages using BM25
   - Reranks with BGE reranker
   - Sends final page and question to GPT (via OpenAI API)
   - Appends result to `answer.json`
4. Stops after 100 questions (customizable)
5. Outputs final results in `full.json`

---

## 🔑 OpenAI API Setup

Update the `Authorization` header in `ask_gpt()` function with your API key:

```python
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_OPENAI_API_KEY'
}
```

---

## 📝 Output

- `answer.json`: Partial results saved incrementally
- `full.json`: Final saved output for all processed questions

---

## 🛠 Tips

- Ensure the PDF text is extractable (not image-based)
- Tune reranking threshold or number of top-k candidates if needed
- Use GPU for faster reranking

---

## 📄 License

MIT License

---