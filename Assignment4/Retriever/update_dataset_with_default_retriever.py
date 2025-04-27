import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import argparse
import numpy as np
from tqdm import tqdm
import copy

# 创建摘要的辅助函数
def create_meaningful_summary(text, max_length=200):
    """创建更有意义的摘要，确保包含完整的句子"""
    # 如果文本已经很短，直接返回
    if len(text) <= max_length:
        return text
    
    # 尝试找到句子边界
    sentences = text.split('. ')
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) + 2 <= max_length:
            if summary:
                summary += ". " + sentence
            else:
                summary = sentence
        else:
            break
    
    # 确保摘要以句号结束
    if summary and not summary.endswith('.'):
        summary += '.'
        
    return summary

class DefaultRetriever:
    def __init__(self, model_name, index_path, meta_path):
        # 加载原始预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 加载FAISS索引
        self.index = faiss.read_index(index_path)
        
        # 加载元数据
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
    
    def retrieve(self, query, top_k=5):
        # 编码查询
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            query_emb = self.model(**inputs).pooler_output.cpu().numpy()
        
        # 归一化向量(用于余弦相似度)
        faiss.normalize_L2(query_emb)
        
        # 搜索top-k文档
        scores, doc_indices = self.index.search(query_emb, top_k)
        
        # 返回检索结果及元数据
        results = []
        for i, idx in enumerate(doc_indices[0]):
            if idx < len(self.metadata["doc_ids"]):  # 安全检查
                text = self.metadata["doc_texts"][idx]
                results.append({
                    "id": self.metadata["doc_ids"][idx],
                    "title": f"Default Retriever Topic {i+1}",
                    "text": text,
                    "score": float(scores[0][i]),
                    "summary": create_meaningful_summary(text, 200),
                    "extraction": text[:400] if len(text) > 400 else text
                })
        
        return results

def convert_and_update_dataset(input_files, output_file, retriever, ndocs=5):
    """
    使用默认retriever更新数据集中的文档
    """
    all_samples = []
    
    # 处理每个输入文件
    for input_file in input_files:
        print(f"处理文件: {input_file}")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # 为每个样本创建新对象（确保保持原始格式）
        for sample in tqdm(data, desc=f"更新 {input_file} 中的检索结果"):
            new_sample = copy.deepcopy(sample)  # 深拷贝保留原始结构
            
            # 获取问题文本
            question = sample.get("question", "")
            
            # 确保问题存在
            if not question:
                continue
                
            # 使用默认retriever获取相关文档
            retrieved_docs = retriever.retrieve(question, top_k=ndocs)
            
            # 更新文档部分
            new_sample["docs"] = retrieved_docs
            
            all_samples.append(new_sample)
    
    # 保存转换后的数据集
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"已更新数据集并保存至 {output_file}")
    return len(all_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用默认retriever更新数据集(比较用)")
    parser.add_argument("--model_name", type=str, default="facebook/contriever", help="默认retriever模型名称")
    parser.add_argument("--index_path", type=str, required=True, help="原始FAISS索引路径") 
    parser.add_argument("--meta_path", type=str, required=True, help="原始索引元数据路径")
    parser.add_argument("--input_files", nargs='+', required=True, help="输入数据集路径列表")
    parser.add_argument("--output_file", type=str, required=True, help="输出数据集路径")
    parser.add_argument("--ndocs", type=int, default=5, help="每个问题检索的文档数量")
    
    args = parser.parse_args()
    
    # 初始化默认retriever
    retriever = DefaultRetriever(
        model_name=args.model_name,
        index_path=args.index_path,
        meta_path=args.meta_path
    )
    
    # 更新数据集
    sample_count = convert_and_update_dataset(
        args.input_files,
        args.output_file,
        retriever,
        args.ndocs
    )
    
    print(f"使用默认retriever更新数据集完成！共处理了 {sample_count} 个样本")