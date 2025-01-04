import jieba, json, pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import requests
from tqdm import tqdm
questions = json.load(open("questions.json",encoding='UTF-8'))

pdf = pdfplumber.open("dataset.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })

# 加载重排序模型
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
rerank_model.cuda()


def ask_gpt(content):
  url = "https://api.openai.com/v1/chat/completions"
  headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer '
  }

  data = {
    "model": "gpt-3.5-turbo-16k", # "gpt-4-0613",
    "messages": [
      {"role": "user", "content": content},
    ]
  }

  response = requests.post(url, headers=headers, json=data)
  return response.json()

pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

for query_idx in tqdm(range(len(questions))):
    doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
    max_score_page_idxs = doc_scores.argsort()[-3:]

    pairs = []
    for idx in max_score_page_idxs:
        pairs.append([questions[query_idx]["question"], pdf_content[idx]['content']])

    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx + 1)

    prompt = '''你是一个汽车专家，帮我结合给定的资料和页码，回答下面的问题。如果问题无法从资料中获得，或无法从资料中进行回答，请无法回答问题。如果问题可以从资料中获得，则请逐步回答。
资料：{0}
问题：{1}

    '''.format(
        '第'+str(max_score_page_idx + 1)+'页:'+pdf_content[max_score_page_idx]['content'],
        questions[query_idx]["question"]
    )
    answer = ask_gpt(prompt)['choices'][0]['message']['content']
    questions[query_idx]['answer'] = answer
    with open('answer.json', 'w', encoding='utf8') as up:
        json.dump(questions, up, ensure_ascii=False, indent=4)
    if query_idx == 100:
        break
with open('full.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)