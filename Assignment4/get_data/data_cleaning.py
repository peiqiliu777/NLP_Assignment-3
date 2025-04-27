import json
import re
def clean_text(text: str) -> str:
    # Replace line breaks and multiple spaces
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    # Delete navigation content and duplicate elements
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

# File paths
input_path = "/Users/a77/Documents/CMU/Sem 2/NLP/AS4/fdic_extracted_data.json"
output_path = "/Users/a77/Documents/CMU/Sem 2/NLP/AS4/fdic_extracted_cleaned.json"
# Load data
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)
# Clean the content field for each item
for item in data:
    if "content" in item:
        item["content"] = clean_text(item["content"])
# Write back to json, maintaining original field order and structure
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Cleaning completed, saved to {output_path}")