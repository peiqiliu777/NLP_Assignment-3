import json
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load data
with open('aml_data.json') as f:
    raw_data = json.load(f)

clean_docs = []
for doc in raw_data:
    # handle html
    if doc["type"] == "HTML":
        soup = BeautifulSoup(doc["content"], 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
    else:
        text = doc["content"]
    
    # chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " "]
    )
    chunks = splitter.split_text(text)
    
    # store results
    for i, chunk in enumerate(chunks):
        clean_docs.append({
            "doc_id": f"{doc['url']}_chunk{i}",
            "url": doc["url"],
            "text": chunk,
            "metadata": {"level": doc["level"]}
        })

# store data
with open('aml_clean.json', 'w') as f:
    json.dump(clean_docs, f, ensure_ascii=False, indent=2)