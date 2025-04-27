import os
import json
import openai
from tqdm import tqdm
from langdetect import detect

# 设置 LiteLLM API
os.environ["LITELLM_API_KEY"] = "sk-xta-CRprnx0q3fdY6t6IBA"
client = openai.OpenAI(
    api_key=os.environ.get("LITELLM_API_KEY"),
    base_url="https://cmu.litellm.ai"
)

def translate_to_english(text):
    prompt = f"""
You are a professional translator. Your task is to translate the following content into fluent, complete English only.
If the text includes Spanish, translate it entirely into English. Do not mix languages. Do not return any Spanish words.
Preserve the structure and bullet points or sections. Here is the content:

{text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a translation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

# 读取原始 JSON
with open("fdic_extracted_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 翻译所有 content 字段
for entry in tqdm(data):
    if "content" in entry:
        original = entry["content"]
        try:
            lang = detect(original[:300])
        except:
            lang = "unknown"
        
        if lang != "en":
            translated = translate_to_english(original)
            # 如果翻译后仍然不是英文，再翻一次
            if detect(translated[:300]) != "en":
                translated = translate_to_english(translated)
            entry["content"] = translated

# 写入文件
with open("fdic_extracted_translated_strict.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 翻译并清洗完成：fdic_extracted_translated_strict.json")
