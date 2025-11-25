#inference_rag.py
import argparse
import random
import time
import numpy as np
import torch
import warnings
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
from typing import List, Dict

warnings.filterwarnings('ignore')

@dataclass
class ConversationPair:
    question: str
    answer: str
    metadata: Dict = None

class ConversationRAG:#all-MiniLM-L6-v2
    def __init__(self, embedding_model="shibing624/text2vec-base-chinese"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.conversations = []
        self.index = None
        self.embeddings = None
    
    def load_conversations_from_jsonl(self, jsonl_path: str) -> List[ConversationPair]:
        conversations = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'conversations' in data:
                            conversation_list = data['conversations']
                            user_msg = None
                            assistant_msg = None
                            
                            for msg in conversation_list:
                                if msg.get('role') == 'user':
                                    user_msg = msg.get('content', '').strip()
                                elif msg.get('role') == 'assistant':
                                    assistant_msg = msg.get('content', '').strip()
                                    break
                            
                            if user_msg and assistant_msg:
                                conversations.append(ConversationPair(
                                    question=user_msg,
                                    answer=assistant_msg,
                                    metadata={'line_number': line_num}
                                ))
                    except:
                        continue
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
        
        print(f"Loaded {len(conversations)} conversation pairs")
        return conversations
    
    def build_index(self, jsonl_path: str, index_path: str = "cache/faiss_index.bin", conversations_cache_path: str = "cache/conversations_cache.json"):
        if os.path.exists(index_path) and os.path.exists(conversations_cache_path):
            print("Loading FAISS index and conversations from cache...")
            self.index = faiss.read_index(index_path)
            with open(conversations_cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                self.conversations = [ConversationPair(**item) for item in cached_data]
            print(f"Index loaded with {self.index.ntotal} vectors. {len(self.conversations)} conversations loaded from cache.")
            return True
        
        print("Loading conversations...")
        self.conversations = self.load_conversations_from_jsonl(jsonl_path)
        
        if not self.conversations:
            print("No conversations loaded. Cannot build index.")
            return False
        
        print("Generating embeddings...")
        questions = [conv.question for conv in self.conversations]
        self.embeddings = self.embedding_model.encode(questions, convert_to_tensor=False)
        self.embeddings = np.array(self.embeddings).astype('float32')
        
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
        
        # Save the index and conversations
        faiss.write_index(self.index, index_path)
        with open(conversations_cache_path, 'w', encoding='utf-8') as f:
            json.dump([conv.__dict__ for conv in self.conversations], f, ensure_ascii=False, indent=4)
        print("FAISS index and conversations saved to cache.")
        
        return True
    
    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.3):
        if self.index is None or not self.conversations:
            return []
        
        # 确保query是字符串类型并清理
        if not isinstance(query, str):
            query = str(query)
        
        query = query.strip()
        
        # 如果query为空，返回空结果
        if not query:
            return []
        
        try:
            # 确保传入的是字符串列表
            query_embedding = self.embedding_model.encode(
                [query], 
                convert_to_tensor=False,
                show_progress_bar=False
            )
            query_embedding = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx < len(self.conversations):
                    results.append((self.conversations[idx], float(score)))
            
            return results
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            print(f"Query type: {type(query)}, Query value: '{query}'")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 3, max_context_length: int = 1000, threshold: float = 0.3):
        try:
            retrieved = self.retrieve(query, top_k=top_k, threshold=threshold)
            
            if not retrieved:
                return ""
            
            context_parts = ["以下是知识库中的相关示例：\n"]
            
            for i, (conv_pair, score) in enumerate(retrieved, 1):
                example = f"示例 {i}:\n问：{conv_pair.question}\n答：{conv_pair.answer}\n"
                current_context = "\n".join(context_parts) + example
                if len(current_context) > max_context_length:
                    break
                context_parts.append(example)
            
            return "\n".join(context_parts)
        
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""

def init_qwen_model(args):
    """Initialize Qwen model with LoRA adapters."""
    base_model_name = "Qwen/Qwen3-4B"
    
    print(f"Loading tokenizer from {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with 4-bit quantization
    print("Loading base model with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2"
    )
    # Load LoRA adapters if provided
    if args.lora_path:
        print(f"Loading LoRA adapters from {args.lora_path}...")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        print("No LoRA path provided, using base model only")
        model = base_model
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params / 1e6:.2f}M')
    print(f'Trainable Parameters: {trainable_params / 1e6:.2f}M')
    
    return model, tokenizer


def create_qwen_rag_prompt(query: str, rag_system: ConversationRAG, use_rag: bool = True, rag_threshold: float = 0.3):
    """Create Qwen-formatted prompt with RAG context - 遵循模型训练格式"""
    
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    instruction = """请以路易威登（Louis Vuitton）专业客户服务代表的身份，专注于OnTheGo手袋系列来回答问题。

**服务标准：**
- 保持优雅、专业和细致的奢侈品牌服务态度
- 展现对OnTheGo系列的深入了解（尺寸、材质、保养等）
- 使用专业且温暖的语气

**回答要求：**
1. 根据知识库中的产品信息作答
2. 提供全面的产品细节（尺寸、材质、特点、保养）
3. 主动满足客户需求，预判相关问题
4. 使用正确的产品名称和术语
5. 仅提供有依据的准确信息
6. 以解决方案为导向

**OnTheGo系列要点：**
- 尺寸：PM、MM、GM
- 特点：可翻转设计、宽敞空间、双重携带方式
- 材质：Monogram帆布、压纹皮革、Damier棋盘格等

【OnTheGo 尺寸参考】
- PM：25 × 19 × 11.5 cm（轻盈小巧）
- MM：35 × 27 × 14 cm（容量适中）
- GM：41 × 34 × 19 cm（大容量旅行包）
"""


    
    # 添加RAG上下文
    if use_rag and rag_system:
        context = rag_system.get_context_for_query(query, top_k=3, max_context_length=800, threshold=rag_threshold)
        if context.strip():
            instruction += f"\n**知识库参考信息：**\n{context}\n"
    
    # 组合最终的用户消息
    user_message = f"{instruction}\n**客户问题：**\n{query}\n\n请提供专业、详细的回答："
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    return messages

def clean_response_simple(response: str) -> str:
    """
    简化版本：只保留到最后一个有效句号/感叹号/问号，移除高度相似的重复句子（80%以上），
    移除**word**、[word]、【word】格式的内容及其后面的所有文本，
    移除\n后不以有效标点结尾的段落，
    移除"再见"及其后所有内容，
    移除第二个"您好"及其后所有内容，
    移除包含电话号码(+开头)的句子
    
    Args:
        response: 原始生成的回复文本
        
    Returns:
        清理后的回复文本
    """
    if not response or response == "[NO RESPONSE]":
        return response
    
    import re
    
    # 移除"再见"及其后面的所有内容（包括再见本身）
    goodbye_pos = response.find('再见')
    if goodbye_pos != -1:
        response = response[:goodbye_pos].strip()
    
    # 移除第二个"您好"及其后面的所有内容
    first_hello = response.find('您好')
    if first_hello != -1:
        # 从第一个"您好"之后开始查找第二个
        second_hello = response.find('您好', first_hello + 2)
        if second_hello != -1:
            response = response[:second_hello].strip()
    
    # 移除包含电话号码的句子（+开头的电话号码格式）
    # 匹配模式: +后面跟数字、短横线、空格等
    phone_pattern = r'[^。！？]*\+\d+[0-9\-\s()（）]*[^。！？]*[。！？]?'
    response = re.sub(phone_pattern, '', response)
    
    # 移除 **word** 格式的内容及其后面的所有文本
    asterisk_pattern = r'\*\*.*?\*\*.*'
    response = re.sub(asterisk_pattern, '', response, flags=re.DOTALL)
    
    # 移除 [word] 格式的内容及其后面的所有文本
    bracket_pattern = r'\[.*?\].*'
    response = re.sub(bracket_pattern, '', response, flags=re.DOTALL)
    
    # 移除 【word】 格式的内容及其后面的所有文本
    chinese_bracket_pattern = r'【.*?】.*'
    response = re.sub(chinese_bracket_pattern, '', response, flags=re.DOTALL)
    
    # 处理换行符，移除换行后不以有效标点结尾的段落
    if '\n' in response:
        paragraphs = response.split('\n')
        valid_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if para:  # 非空段落
                # 检查段落是否以有效标点结尾
                if para.endswith(('。', '！', '？', '.', '!', '?')):
                    valid_paragraphs.append(para)
                else:
                    # 找到最后一个有效标点的位置
                    last_valid = -1
                    for ending in ('。', '！', '？', '.', '!', '?'):
                        pos = para.rfind(ending)
                        if pos > last_valid:
                            last_valid = pos
                    
                    if last_valid != -1:
                        # 截取到最后一个有效标点
                        valid_paragraphs.append(para[:last_valid + 1])
        
        response = '\n'.join(valid_paragraphs)
    
    # 找到最后一个有效的句子结束符
    valid_endings = ('。', '！', '？')
    last_valid_pos = -1
    
    for ending in valid_endings:
        pos = response.rfind(ending)
        if pos > last_valid_pos:
            last_valid_pos = pos
    
    # 如果找到有效结束符，截取到该位置（包含结束符）
    if last_valid_pos != -1:
        response = response[:last_valid_pos + 1].strip()
    
    # 移除相似的重复句子（80%以上相似度）
    # 按句子分割（保留标点符号）
    sentences = re.split(r'([。！？])', response)
    
    # 重组句子
    complete_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
            if sentence:
                complete_sentences.append(sentence)
    
    # 计算两个字符串的相似度
    def similarity(s1: str, s2: str) -> float:
        """计算两个字符串的相似度（使用字符级别的比较）"""
        if not s1 or not s2:
            return 0.0
        
        # 移除标点符号进行比较
        s1_clean = s1.rstrip('。！？').strip()
        s2_clean = s2.rstrip('。！？').strip()
        
        if s1_clean == s2_clean:
            return 1.0
        
        # 使用简单的字符匹配计算相似度
        len1, len2 = len(s1_clean), len(s2_clean)
        max_len = max(len1, len2)
        
        if max_len == 0:
            return 0.0
        
        # 计算最长公共子序列的长度
        matches = 0
        for char in set(s1_clean):
            matches += min(s1_clean.count(char), s2_clean.count(char))
        
        return matches / max_len
    
    # 去除相似度超过80%的连续句子
    cleaned_sentences = []
    prev_sentence = None
    
    for sentence in complete_sentences:
        if prev_sentence is None:
            # 第一个句子总是保留
            cleaned_sentences.append(sentence)
            prev_sentence = sentence
        else:
            # 计算与前一句的相似度
            sim = similarity(prev_sentence, sentence)
            
            if sim < 0.8:  # 相似度小于80%才保留
                cleaned_sentences.append(sentence)
                prev_sentence = sentence
            # 否则丢弃（相似度>=80%）
    
    result = ''.join(cleaned_sentences)
    
    return result if result else response



def generate_response(model, tokenizer, query, rag_system, args, use_rag=True):
    """Generate response using Qwen model with RAG."""
    try:
        # 确保query是有效的字符串
        if not isinstance(query, str) or not query.strip():
            return "[Invalid query]"
        
        # Create messages with RAG context
        messages = create_qwen_rag_prompt(query, rag_system, use_rag, args.rag_threshold)
        
        # 使用 apply_chat_template - 推荐方式
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # 方式1: 直接tokenize并返回tensor（推荐）
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=args.enable_thinking 
            ).to(model.device)
            
            # 如果你想看生成的prompt格式（调试用）
            if False:  # 设置为True可以查看prompt
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking 
                )
                print(f"\n生成的Prompt:\n{prompt_text}\n")
        else:
            # Fallback: 如果没有chat_template（理论上Qwen都有）
            print("⚠️ Warning: No chat_template found, using manual formatting")
            prompt = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p if args.temperature > 0 else None,
                do_sample=args.temperature > 0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                # use_cache=False,
                repetition_penalty=1.2  # 防止重复
            )
        
        # Decode only the generated part
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # 清理回复
        response = clean_response_simple(response)
        
        return response if response else "[NO RESPONSE]"
        
    except Exception as e:
        import traceback
        print(f"Error generating response: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return f"[Error: {e}]"

def interactive_chat(model, tokenizer, rag_system, args):
    """Interactive chat mode with RAG."""
    print("\n" + "="*60)
    print("🤖 Qwen Model with RAG - Interactive Chat")
    print("="*60)
    print("\nCommands:")
    print("  'quit' or 'q' - Exit")
    print("  'toggle_rag' - Toggle RAG on/off")
    print("  'search <query>' - Test RAG retrieval")
    print("  'clear' - Clear screen")
    print("-" * 60 + "\n")
    
    use_rag = True
    
    while True:
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        elif user_input.lower() == 'toggle_rag':
            use_rag = not use_rag
            status = "✅ ON" if use_rag else "❌ OFF"
            print(f"🔧 RAG is now {status}")
            continue
        
        elif user_input.lower() == 'clear':
            os.system('clear' if os.name != 'nt' else 'cls')
            continue
        
        elif user_input.lower().startswith('search '):
            query = user_input[7:].strip()
            if rag_system:
                results = rag_system.retrieve(query, top_k=5, threshold=args.rag_threshold)
                print(f"\n🔍 Search results for: '{query}'")
                print(f"Threshold: {args.rag_threshold}")
                if results:
                    for i, (conv_pair, score) in enumerate(results, 1):
                        print(f"\n📌 Result {i} (Score: {score:.3f}):")
                        print(f"   Q: {conv_pair.question[:100]}...")
                        print(f"   A: {conv_pair.answer[:150]}...")
                else:
                    print("❌ No relevant results found (try lowering threshold)")
            else:
                print("❌ RAG system not available")
            print("-" * 60)
            continue
        
        if not user_input:
            continue
        
        # Show RAG status
        rag_icon = "🔍 RAG" if use_rag else "🚫 No RAG"
        print(f"\n{rag_icon} | 🤖 Assistant: ", end="", flush=True)
        
        # Generate response
        start_time = time.time()
        response = generate_response(model, tokenizer, user_input, rag_system, args, use_rag=use_rag)
        elapsed_time = time.time() - start_time
        
        print(response)
        print(f"\n⏱️  Response time: {elapsed_time:.2f}s")
        
        # Show retrieved context if RAG was used
        if use_rag and rag_system:
            retrieved = rag_system.retrieve(user_input, top_k=2, threshold=args.rag_threshold)
            if retrieved:
                print(f"\n📚 Retrieved {len(retrieved)} relevant examples:")
                for i, (conv_pair, score) in enumerate(retrieved, 1):
                    print(f"   {i}. (Score: {score:.3f}) {conv_pair.question[:60]}...")
        
        print("-" * 60 + "\n")

def batch_evaluation(model, tokenizer, rag_system, args):
    """Batch evaluation mode."""
    if args.manual_input:
        print("\n" + "="*60)
        print("📝 Manual Input with RAG Comparison")
        print("="*60)
        print("Enter your questions. Type 'quit' to exit.")
        print("-" * 60 + "\n")
        
        while True:
            user_input = input("👤 Enter your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            print(f"\n❓ Question: {user_input}")
            print("=" * 60)
            
            if args.compare_rag:
                print("\n🚫 Without RAG:")
                response_no_rag = generate_response(model, tokenizer, user_input, rag_system, args, use_rag=False)
                print(f"{response_no_rag}")
                
                print("\n" + "-" * 60)
                print("🔍 With RAG:")
                response_with_rag = generate_response(model, tokenizer, user_input, rag_system, args, use_rag=True)
                print(f"{response_with_rag}")
                
                if rag_system:
                    retrieved = rag_system.retrieve(user_input, top_k=3, threshold=args.rag_threshold)
                    if retrieved:
                        print("\n📚 Retrieved Context:")
                        for j, (conv_pair, score) in enumerate(retrieved, 1):
                            print(f"   {j}. (Score: {score:.3f}) {conv_pair.question[:80]}...")
                    else:
                        print("\n📚 No relevant context found")
            else:
                response = generate_response(model, tokenizer, user_input, rag_system, args, use_rag=True)
                print(f"🤖 Assistant: {response}")
            
            print("=" * 60 + "\n")
    else:
        # Predefined test prompts
        test_prompts = [
            "What are the key HR policies in Malaysia?",
            "How should companies handle employee termination?",
            "What are the EPF contribution requirements?",
            "Explain the process for handling workplace disputes",
            "What are the mandatory employee benefits?",
        ]
        
        print("\n" + "="*60)
        print(f"📊 Batch Evaluation with RAG ({len(test_prompts)} prompts)")
        print("="*60 + "\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"[{i}/{len(test_prompts)}] ❓ {prompt}")
            
            if args.compare_rag:
                print("\n🚫 Without RAG:")
                response_no_rag = generate_response(model, tokenizer, prompt, rag_system, args, use_rag=False)
                print(f"{response_no_rag}")
                
                print("\n🔍 With RAG:")
                response_with_rag = generate_response(model, tokenizer, prompt, rag_system, args, use_rag=True)
                print(f"{response_with_rag}")
                
                if rag_system:
                    retrieved = rag_system.retrieve(prompt, top_k=2, threshold=args.rag_threshold)
                    if retrieved:
                        print("\n📚 Retrieved Context:")
                        for j, (conv_pair, score) in enumerate(retrieved, 1):
                            print(f"   {j}. (Score: {score:.3f}) {conv_pair.question[:80]}...")
            else:
                response = generate_response(model, tokenizer, prompt, rag_system, args, use_rag=True)
                print(f"🤖 Assistant: {response}")
            
            print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Qwen Model with RAG Evaluation")
    
    # Model configuration
    parser.add_argument('--lora_path', type=str, default="out/lora/latest_qwen_4b_zh.pth", 
                        help='Path to LoRA adapters (leave empty to use base model only)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8, help='Top-p sampling parameter')
    parser.add_argument('--max_new_tokens', type=int, default=150, help='Maximum new tokens to generate')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum input sequence length')
    
    # RAG configuration
    parser.add_argument('--rag_dataset', type=str, required=True, 
                        help='Path to JSONL file for RAG knowledge base')
    parser.add_argument('--disable_rag', action='store_true', help='Disable RAG functionality')
    parser.add_argument('--rag_threshold', type=float, default=0.3, 
                        help='Minimum similarity score for RAG retrieval (0.0-1.0)')
    
    # Evaluation mode
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], default='interactive',
                        help='Evaluation mode')
    parser.add_argument('--compare_rag', action='store_true', 
                        help='Compare responses with and without RAG (batch mode)')
    parser.add_argument('--manual_input', action='store_true', 
                        help='Allow manual input in batch mode')
    parser.add_argument('--enable_thinking', action='store_true', default=True)
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    print("🚀 Initializing Qwen model...")
    model, tokenizer = init_qwen_model(args)
    
    # Initialize RAG system
    rag_system = None
    if not args.disable_rag:
        try:
            print("\n📚 Setting up RAG system...")
            if not os.path.exists(args.rag_dataset):
                print(f"❌ RAG dataset not found: {args.rag_dataset}")
                print("Continuing without RAG...")
            else:
                rag_system = ConversationRAG()
                if rag_system.build_index(args.rag_dataset):
                    print("✅ RAG system initialized successfully!")
                else:
                    rag_system = None
                    print("❌ Failed to build RAG index, continuing without RAG...")
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            print("Continuing without RAG...")
            rag_system = None
    else:
        print("RAG system disabled")
    
    # Run evaluation
    if args.mode == 'interactive':
        interactive_chat(model, tokenizer, rag_system, args)
    elif args.mode == 'batch':
        batch_evaluation(model, tokenizer, rag_system, args)

if __name__ == "__main__":
    main()