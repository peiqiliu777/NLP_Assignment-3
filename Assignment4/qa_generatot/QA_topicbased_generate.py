import os
import openai
import json
import re
import time
from typing import List, Dict, Any

# ========== CONFIGURATION ==========
os.environ["LITELLM_API_KEY"] = "sk-xta-CRprnx0q3fdY6t6IBA"
client = openai.OpenAI(
    api_key=os.environ.get("LITELLM_API_KEY"),
    base_url="https://cmu.litellm.ai"
)

NUM_TOPICS = 232
MAX_QA_PAIRS_PER_TOPIC = 3
MAX_DOCS = 15
DOC_CHUNK_LENGTH = 300

# ========== STEP 1: TEXT EXTRACTION ==========

def extract_text_chunks(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    chunks = []
    for i, item in enumerate(data):
        if "content" in item and item["content"]:
            content = re.sub(r'\\n', '\n', item["content"])
            chunks.append({
                "title": f"AML Topic {i+1}",
                "url": item.get("url", f"https://source.fake/{i+1}"),
                "content": content
            })
    return chunks

# ========== STEP 2: SPLIT LONG CHUNKS ==========

def split_topic_content(topic: Dict[str, str], max_len: int = 1200) -> List[Dict[str, str]]:
    content = topic["content"]
    parts = [content[i:i+max_len].strip() for i in range(0, len(content), max_len)]
    return [
        {"title": topic["title"], "url": topic["url"], "content": part}
        for part in parts if len(part.split()) > 50
    ]

# ========== STEP 3: GPT-BASED QA GENERATION ==========

def generate_qa_triplet(topic: Dict[str, str]) -> List[Dict[str, Any]]:
    system_prompt = """You are a financial compliance assistant helping to build a Self-RAG QA dataset for enterprise AML education.

Your task is to generate 3 related question-answer pairs from the topic below. Each QA pair must include:
- `question`: enterprise-relevant AML question
- `answer`: grounded, factual, full answer
- `short_answers`: list of 2-3 concise answers
- `context`: (1-2 sentence summary to support or frame the question)
- `source_text`: supporting content from the passage

Rules:
- Use ONLY the provided content as source
- Questions must be distinct, topically related
- Do NOT repeat or paraphrase the same question
- All fields must be completed per QA pair

Return a valid JSON list like:
[
  {
    "question": "...",
    "answer": "...",
    "short_answers": ["...", "..."],
    "context": "...",
    "source_text": "..."
  },
  ...
]
"""

    user_prompt = f"""Based on the following content, generate 3 enterprise-facing QA pairs following the format above.

---CONTENT START---
{topic['content']}
---CONTENT END---
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=2500
        )
        result = response.choices[0].message.content
        match = re.search(r'\[\s*{.*?}\s*\]', result, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"❌ Error generating QA for topic {topic['title']}: {e}")
    return []

# ========== STEP 4: FORMAT FOR SELF-RAG ==========

def create_asqa_sample(topic: Dict[str, str], qa_pairs: List[Dict[str, Any]], sample_id: str) -> Dict[str, Any]:
    docs = []
    for i in range(MAX_DOCS):
        start = i * DOC_CHUNK_LENGTH
        end = start + DOC_CHUNK_LENGTH
        chunk_text = topic["content"][start:end].strip()
        if not chunk_text:
            continue
        docs.append({
            "id": f"doc-{sample_id}-{i+1}",
            "title": f"{topic['title']} Part {i+1}",
            "text": chunk_text,
            "score": round(0.9 - i * 0.03, 3),
            "summary": chunk_text[:150].replace('\n', ' ') + "...",
            "extraction": chunk_text[:200]
        })

    main_question = qa_pairs[0]["question"]
    short_summary_answer = " ".join([qa["answer"].split(".")[0] + "." for qa in qa_pairs])
    full_answer = "\n\n".join([qa["answer"] for qa in qa_pairs])

    return {
        "qa_pairs": [
            {
                "context": qa.get("context", "No context provided"),
                "question": qa["question"],
                "short_answers": qa["short_answers"],
                "wikipage": None
            } for qa in qa_pairs
        ],
        "wikipages": [
            {"title": topic["title"], "url": topic["url"]}
        ],
        "annotations": [
            {
                "knowledge": [],
                "long_answer": short_summary_answer
            },
            {
                "knowledge": [
                    {
                        "content": qa["source_text"],
                        "wikipage": topic["title"]
                    } for qa in qa_pairs
                ],
                "long_answer": full_answer
            }
        ],
        "sample_id": sample_id,
        "question": main_question,
        "docs": docs[:MAX_DOCS],
        "answer": short_summary_answer
    }

# ========== STEP 5: MAIN DRIVER ==========

def main():
    with open("get_data/fdic_extracted_translated_strict.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    text_chunks = extract_text_chunks(data)
    dataset = []
    seen_questions = set()

    for i, topic in enumerate(text_chunks[:NUM_TOPICS]):
        print(f"\n🔍 Processing topic {i+1}/{len(text_chunks)}: {topic['title']}")

        subtopics = split_topic_content(topic, max_len=1200)
        subtopics = subtopics[:MAX_QA_PAIRS_PER_TOPIC]

        actual_generated = 0

        for j, subtopic in enumerate(subtopics):
            qa_triplet = generate_qa_triplet(subtopic)

            if qa_triplet:
                qa_triplet_filtered = []
                for qa in qa_triplet:
                    question = qa["question"].strip().lower()
                    if question not in seen_questions:
                        seen_questions.add(question)
                        qa_triplet_filtered.append(qa)

                if qa_triplet_filtered:
                    sample_id = f"{topic['title'].lower().replace(' ', '-')}-{j+1}"
                    sample = create_asqa_sample(subtopic, qa_triplet_filtered, sample_id=sample_id)
                    dataset.append(sample)
                    actual_generated += 1
                    time.sleep(1)

        print(f"✅ Generated {actual_generated} QA group(s) for topic: {topic['title']}")

    with open("qa_generator/full_qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Saved {len(dataset)} samples to new_test.json")

if __name__ == "__main__":
    main()
