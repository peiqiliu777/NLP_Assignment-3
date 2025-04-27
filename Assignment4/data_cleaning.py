import json
import re

def clean_text(text: str) -> str:
    # 替换换行符和多空格
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)

    # 删除导航类内容和重复元素
    patterns_to_remove = [
        r"Toggle submenu", r"Next page", r"VIEW MORE", r"Loading\.\.\.",
        r"CONTACT THE FDIC", r"CONTACT US", r"HOW CAN WE HELP YOU\?", r"I am a\.\.\.",
        r"Select the information.*?", r"Footer Secondary Menu.*", r"Follow the FDIC.*?",
        r"Enter your email address Subscribe", r"Policies.*?Inspector General",
        r"Define “I am a.*?”", r"Advanced Search", r"Press Releases.*?", r"Subscribe",
        r"Search FDIC.gov", r"Search", r"usa.gov", r"Privacy", r"No Fear Act Data"
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = text.replace('…', '.')
    text = re.sub(r'\s+\.', '.', text)
    return text.strip()

# 文件路径
input_path = "/Users/a77/Documents/CMU/Sem 2/NLP/AS4/fdic_extracted_data.json"
output_path = "/Users/a77/Documents/CMU/Sem 2/NLP/AS4/fdic_extracted_cleaned.json"

# 加载数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 清洗每一项的 content 字段
for item in data:
    if "content" in item:
        item["content"] = clean_text(item["content"])

# 写回 json，保持原始字段顺序和结构
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"清洗完成，已保存至 {output_path}")
