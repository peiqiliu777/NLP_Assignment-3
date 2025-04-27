import json
import time
import litellm
from tqdm import tqdm

def detect_and_translate(text, api_key, base_url, model):
    """检测文本语言并在必要时翻译为英语"""
    if not text or len(text.strip()) == 0:
        return text
    
    # 首先检测语言
    try:
        response = litellm.completion(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": "You are a language detection tool. Output only 'en' if the text is in English, or the ISO language code if not English."},
                {"role": "user", "content": f"Detect the language of the following text: {text[:500]}..."}
            ],
            temperature=0
        )
        
        language = response.choices[0].message.content.strip().lower()
        
        # 如果不是英语，进行翻译
        if language != "en" and language != "english":
            response = litellm.completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate the text to English while preserving all meaning, technical terms, and formatting. Do not add or remove information."},
                    {"role": "user", "content": f"Translate this text to English: {text}"}
                ],
                temperature=0
            )
            return response.choices[0].message.content
        else:
            return text
    except Exception as e:
        print(f"Error processing text: {e}")
        return text

def process_json_file(input_file, output_file, api_key, base_url, model):
    """处理JSON文件，翻译所有非英语文本"""
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 遍历JSON数据并翻译
    for item in tqdm(data, desc="Processing items"):
        # 翻译docs中的文本
        if "docs" in item:
            for doc in item["docs"]:
                if "text" in doc:
                    doc["text"] = detect_and_translate(doc["text"], api_key, base_url, model)
                if "summary" in doc:
                    doc["summary"] = detect_and_translate(doc["summary"], api_key, base_url, model)
                if "extraction" in doc:
                    doc["extraction"] = detect_and_translate(doc["extraction"], api_key, base_url, model)
        
        # 翻译annotations中的knowledge内容
        if "annotations" in item:
            for annotation in item["annotations"]:
                if "knowledge" in annotation and annotation["knowledge"]:
                    for knowledge in annotation["knowledge"]:
                        if "content" in knowledge:
                            knowledge["content"] = detect_and_translate(knowledge["content"], api_key, base_url, model)
        
        # 每个项目处理后稍作暂停，避免API速率限制
        time.sleep(0.5)
    
    # 保存翻译后的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate non-English text in JSON to English')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--api_key', type=str, required=True, help='LiteLLM API key')
    parser.add_argument('--base_url', type=str, default="https://cmu.litellm.ai", help='Base URL for LiteLLM API')
    parser.add_argument('--model', type=str, default="openai/gpt-4o", help='Model to use for translation')
    
    args = parser.parse_args()
    
    print(f"Starting translation process from {args.input} to {args.output} using {args.model}")
    process_json_file(args.input, args.output, args.api_key, args.base_url, args.model)
    print(f"Translation complete. Output saved to {args.output}")


    # python translate.py --input /gpfsnyu/scratch/yx2432/Research/Zhuzi/NLP_Project3/self-rag-main/retrieval_lm/eval_data/aml_self_rag_dataset.json --output /gpfsnyu/scratch/yx2432/Research/Zhuzi/NLP_Project3/self-rag-main/retrieval_lm/eval_data/aml_self_rag_dataset_en.json --api_key "sk-rZagUnNw1wA3GksBx0l4pQ" --model openai/gpt-4o