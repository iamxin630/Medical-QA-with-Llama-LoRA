"""
HW1-Question-Answering: PathoQA with Llama-3.2-1B-Instruct
Training pipeline with LoRA finetuning, CoT prompting, and visualization
"""

import os
import json
import logging
import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================================
# Configuration and Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # output to stream
        logging.FileHandler('train.log', mode='a'), # output to train.log
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Training configuration"""
    # Model and data
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_path: str = "dataset"
    output_dir: str = "saved_models"

    # LoRA hyperparameters
    lora_r: int = 32 # 可訓練參數的維度
    lora_alpha: int = 64 # 負責縮放權重更新，幫助訓練在開始時更穩定
    lora_dropout: float = 0.1 # 隨機丟棄 10% 的 LoRA 權重，防止Overfitting

    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4 # 4 * 4 = 16 effective batch size
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    early_stopping_patience: int = 2

    # Sequence parameters
    max_length: int = 512
    max_new_tokens: int = 64 # 64 個 Token 足以推理 + 答案

    # Data split ratios — 90/10 gives more training data
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    test_ratio: float = 0.0

    # Device and seed
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Final submission strategy
    do_full_data_submission: bool = True
    full_data_epochs: int = 2
    submission_seeds: List[int] = None

    def __post_init__(self):
        if self.submission_seeds is None:
            self.submission_seeds = [42, 52, 62]

def set_seed(seed: int): # 固定隨機性
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# Custom Prompting Strategy (Zero-Shot / Few-Shot / Chain-of-Thought)
# ============================================================================

# Few-shot exemplar for medical QA (one short example to save token budget)
FEW_SHOT_EXAMPLE = {
    "question": "Which cell type is primarily responsible for humoral immunity?",
    "options": ["T cells", "B cells", "Natural killer cells", "Macrophages"],
    "reasoning": "Humoral immunity relies on antibodies produced by B cells (plasma cells). T cells handle cell-mediated immunity, NK cells target infected cells, and macrophages are phagocytes.",
    "answer": "B",
}


class MedicalQAPrompt:
    """
    Custom prompting strategy for medical QA combining:
    - Few-Shot: exemplars with correct reasoning
    - Chain-of-Thought (CoT): step-by-step reasoning prompt
    - Zero-Shot fallback when context budget is tight
    """

    SYSTEM_PROMPT = ( # 定義模型的身分與行為
        "You are a pathology expert. "
        "Analyze the question, reason briefly, then state your final answer as: "
        "The answer is X. (where X is A, B, C, or D)"
    )

    @staticmethod
    def create_prompt(question: str, options: List[str], use_few_shot: bool = True) -> str:
        """Unified prompt used for both training and inference.

        One optional few-shot exemplar is prepended to illustrate the expected
        answer format while staying within the token budget.
        """
        parts = [MedicalQAPrompt.SYSTEM_PROMPT, ""]

        if use_few_shot:
            ex = FEW_SHOT_EXAMPLE
            parts.append(f"Q: {ex['question']}")
            for i, letter in enumerate(["A", "B", "C", "D"]):
                parts.append(f"{letter}) {ex['options'][i]}")
            parts.append(f"Reasoning: {ex['reasoning']}")
            parts.append(f"The answer is {ex['answer']}.")
            parts.append("")

        parts.append(f"Q: {question}")
        for i, letter in enumerate(["A", "B", "C", "D"]):
            parts.append(f"{letter}) {options[i]}")
        return "\n".join(parts)

    @staticmethod
    def extract_answer(generated_text: str) -> str:
        """Parse the predicted option letter from generated text.

        Tries several patterns in order of specificity.  Falls back to the
        first standalone A/B/C/D found.  Returns None only if nothing matches
        so callers can decide on a fallback strategy.
        """
        text = generated_text.upper().strip()
        # Pattern 1: "The answer is B" / "ANSWER: C"
        m = re.search(r"(?:THE\s+)?(?:FINAL\s+)?ANSWER\s*(?:IS)?\s*[:：]?\s*\(?([ABCD])\)?", text)
        if m:
            return m.group(1)
        # Pattern 2: first standalone letter A-D
        m = re.search(r"\b([ABCD])\b", text)
        if m:
            return m.group(1)
        # Return None so callers can apply their own fallback
        return None

# ============================================================================
# Dataset Definition
# ============================================================================

class PathoQADataset(Dataset):
    """PathoQA Medical Question Answering Dataset"""

    def __init__(self, df, tokenizer, max_length=512, include_labels=True, mode='train'): # 儲存 DataFrame、Tokenizer，並設定最大長度限制
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels
        self.mode = mode  # 'train' or 'eval' or 'infer'
        if self.tokenizer.pad_token is None:
            # Llama 模型預設不一定有填充符號，若沒有則使用結束符號 eos_token 代替
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row['question'])
        options = [str(row['opa']), str(row['opb']), str(row['opc']), str(row['opd'])]

        # Use the same prompt template for both training and inference
        # Training: with few-shot exemplar to teach the answer format
        # Inference: same prompt so there's no distribution shift
        prompt = MedicalQAPrompt.create_prompt(question, options, use_few_shot=True)

        if self.include_labels and 'ans' in row:
            ans_raw = str(row['ans']).strip()
            if ans_raw in ('0', '1', '2', '3'):
                answer = ['A', 'B', 'C', 'D'][int(ans_raw)]
            else:
                answer = ans_raw.upper()

            # Training target: short reasoning cue + structured answer
            answer_text = f"\nThe answer is {answer}."

            # Tokenize prompt and answer separately to build proper labels
            prompt_enc = self.tokenizer(
                prompt, add_special_tokens=True, return_tensors='pt'
            )
            answer_enc = self.tokenizer(
                answer_text, add_special_tokens=False, return_tensors='pt'
            )

            # Concat: prompt tokens + answer tokens
            input_ids = torch.cat([prompt_enc['input_ids'].squeeze(0),
                                   answer_enc['input_ids'].squeeze(0)])
            attention_mask = torch.ones_like(input_ids)

            # Labels: mask prompt part with -100, keep answer part
            # 強迫模型只針對給出答案計算損失Loss，而不會浪費梯度去學習如何重複問題，提升微調效果
            labels = torch.full_like(input_ids, -100)
            prompt_len = prompt_enc['input_ids'].size(1)
            labels[prompt_len:] = input_ids[prompt_len:]

            # Truncate / pad to max_length
            if input_ids.size(0) > self.max_length: # 超過 512，就切掉後面的部分
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                labels = labels[:self.max_length]
            elif input_ids.size(0) < self.max_length: # 如果太短，就補上無意義的填充符號，並且標記這些位置不參與計算
                pad_len = self.max_length - input_ids.size(0)
                pad_id = self.tokenizer.pad_token_id
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'label': answer,
                'prompt': prompt,
            }

        # Inference mode (no labels)
        return {
            'prompt': prompt,
        }

# ============================================================================
# Data Loading
# ============================================================================

def load_and_split_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split dataset into train/val/test/benchmark"""
    logger.info("Loading dataset...")
    
    train_df = pd.read_csv(f"{config.dataset_path}/dataset.csv")
    benchmark_df = pd.read_csv(f"{config.dataset_path}/benchmark.csv")
    
    logger.info(f"  Training data: {len(train_df)} samples")
    logger.info(f"  Benchmark data: {len(benchmark_df)} samples")
    
    # Standardize column names
    train_df.columns = train_df.columns.str.lower()
    benchmark_df.columns = benchmark_df.columns.str.lower()
    
    # Split into train/val/test
    if config.test_ratio > 0:
        train_data, temp = train_test_split(
            train_df,
            test_size=(config.val_ratio + config.test_ratio),
            random_state=config.seed,
            stratify=train_df['ans'] # 確保分割後的各個集合中，正確答案 A, B, C, D 的分佈比例與原始數據集完全一致
        )
        val_data, test_data = train_test_split(
            temp,
            test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
            random_state=config.seed,
            stratify=temp['ans']
        )
    else:
        # 90/10 split: use val set as test set too
        train_data, val_data = train_test_split(
            train_df,
            test_size=config.val_ratio,
            random_state=config.seed,
            stratify=train_df['ans']
        )
        test_data = val_data  # report on val set
    
    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data, benchmark_df

# ============================================================================
# Helper Functions for Device Management
# ============================================================================

def get_model_device(model): 
    """Get the device of the model"""
    return next(model.parameters()).device


def generate_predictions(
    model,
    tokenizer,
    prompt_batch: List[str],
    max_new_tokens: int = 64,
    device=None,
) -> List[str]:
    """Generate text from prompts and parse A/B/C/D from the generated output."""
    if device is None:
        device = get_model_device(model)
    inputs = tokenizer( # 將文字轉換為 PyTorch 張量
        prompt_batch,
        max_length=512,
        padding=True, # 確保同一個 Batch 裡的 Prompt 長度一致，讓 GPU 能並行處理多筆題目
        truncation=True, # 太長就切斷（上限 512）
        return_tensors='pt',
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Enable KV-cache for efficient generation
    model.config.use_cache = True # 開啟緩存，儲存模型已經計算過的 Token ，在產生下一個字時不需要從頭計算，提升生成速度
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False, # 使用 Greedy Search
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = False  # restore for training

    preds = []
    prompt_len = input_ids.shape[1]
    for out in output_ids:
        generated = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        ans = MedicalQAPrompt.extract_answer(generated) # 尋找 A/B/C/D
        if ans is None:
            # If parsing fails, pick a random option to avoid systematic bias
            ans = np.random.choice(['A', 'B', 'C', 'D'])
            logger.debug(f"Parse failure, random fallback -> {ans}: {generated[:80]}")
        preds.append(ans)
    return preds

# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, config):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0.0
    valid_steps = 0
    device = get_model_device(model)
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training") # 建立進度條
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
  
        # 將資料餵給模型，得到輸出（包含 Loss）
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if (labels != -100).sum().item() == 0:
            continue

        # Label smoothing: mix CE loss with uniform distribution over A/B/C/D
        loss = outputs.loss
        if config.label_smoothing > 0:
            loss = loss * (1 - config.label_smoothing) + config.label_smoothing * 1.3863  # ln(4)

        if not torch.isfinite(loss):
            logger.warning(f"NaN loss at step {step}, skipping")
            continue

        # 將 Loss 除以累積步數，因為梯度會累加，所以要取平均
        loss = loss / config.gradient_accumulation_steps
        loss.backward() # 計算梯度（反向傳播）

        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm) # 梯度裁剪：防止梯度爆炸（數值突然衝太高）
            optimizer.step() # 根據梯度更新權重
            scheduler.step() # 更新學習率
            optimizer.zero_grad() # 歸零梯度，準備下一輪累積

        # 將累積的 Loss 加回總數（乘回倍數以得到真實數值）
        total_loss += loss.item() * config.gradient_accumulation_steps
        valid_steps += 1
        pbar.set_postfix({'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}'})

        del loss, outputs # 手動刪除變數以釋放空間
        torch.cuda.empty_cache() # 清理 GPU 碎片

    return total_loss / max(valid_steps, 1)

def evaluate_epoch(model, val_loader, config):
    """Evaluate on val/test set using text generation."""
    model.eval()
    total_loss = 0.0
    all_preds = [] # 儲存所有的預測答案 (A/B/C/D)
    all_labels = [] # 儲存所有的正確答案
    device = get_model_device(model)
    tokenizer = config.tokenizer

    pbar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad(): #驗證階段，不需要計算梯度，可以節省記憶體和加速運算
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 1. 計算 Loss (用於了解模型對正確答案的熟悉度)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            # 2. 執行生成預測
            preds = generate_predictions(
                model, tokenizer, list(batch['prompt']),
                max_new_tokens=config.max_new_tokens, device=device,
            )

            all_preds.extend(preds) # 將這批預測結果加進清單
            all_labels.extend(batch['label']) # 將這批正確答案加進清單

            del outputs # 立即刪除輸出張量以釋放顯存
            torch.cuda.empty_cache() # 清理 GPU 碎片

    avg_loss = total_loss / len(val_loader) # 計算平均損失
    accuracy = accuracy_score(all_labels, all_preds) # 計算準確率 (正確題數/總題數)
    return avg_loss, accuracy, all_preds, all_labels


def generate_full_data_submission(config: Config, benchmark_df: pd.DataFrame):
    """Train on all data with multiple seeds, then hard-vote across seeds for submission."""
    logger.info("\n" + "=" * 80)
    logger.info("Stage 2: Full-data finetune for final submission")
    logger.info("=" * 80)

    full_train_df = pd.read_csv(f"{config.dataset_path}/dataset.csv")
    full_train_df.columns = full_train_df.columns.str.lower()

    all_seed_preds: List[List[str]] = []  # [seed_idx][sample_idx] -> letter
    question_ids = benchmark_df['question_id'].tolist()

    for seed in config.submission_seeds:
        logger.info(f"\n[Submission Seed {seed}] Training on full data...")
        set_seed(seed)

        # 重新載入分詞器並設定截斷與填充方向
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        tokenizer.truncation_side = 'left' # 從左側截斷
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # 處理填充符號缺失
        tokenizer.padding_side = 'right' # 向右填充

        # 載入 Llama 基礎模型 (bfloat16 精度以節省顯存)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model.config.use_cache = False # 訓練模式下關閉 KV-cache 緩存

        # 定義 LoRA 低秩適配器參數
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config) # 將 LoRA 應用到模型上

        train_dataset = PathoQADataset(full_train_df, tokenizer, config.max_length, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        # 設定優化器，僅針對 LoRA 可訓練參數進行更新
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), # Frozen parameters will have requires_grad=False and thus be excluded from optimization
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 使用 OneCycleLR 動態調整學習率
        total_steps = max(1, len(train_loader) * config.full_data_epochs // config.gradient_accumulation_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR( # Warmup + Cosine Annealing
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
            anneal_strategy='cos',
        )

        seed_config = deepcopy(config) # 確保了每個種子訓練時，拿到的設定都是乾淨且獨立的
        seed_config.tokenizer = tokenizer

        for epoch in range(config.full_data_epochs): # 進行 2 輪全數據訓練
            loss = train_epoch(model, train_loader, optimizer, scheduler, seed_config)
            logger.info(f"  Seed {seed} Epoch {epoch + 1}/{config.full_data_epochs} train_loss={loss:.4f}")

        # Generate benchmark predictions for this seed
        benchmark_dataset = PathoQADataset(benchmark_df, tokenizer, config.max_length, include_labels=False, mode='infer')
        benchmark_loader = DataLoader(benchmark_dataset, batch_size=config.batch_size, shuffle=False) # 預測時必須按照原本的題目順序

        seed_preds: List[str] = []
        device = get_model_device(model)
        model.eval() # 切換為評估模式
        with torch.no_grad(): # 關閉梯度計算以加速
            for batch in tqdm(benchmark_loader, desc=f"Benchmark seed {seed}"):
                # 讓模型針對 benchmark 寫出答案
                preds = generate_predictions(
                    model, tokenizer, list(batch['prompt']),
                    max_new_tokens=config.max_new_tokens, device=device,
                )
                seed_preds.extend(preds) # 收集此種子模型的預測結果

        logger.info(f"  Seed {seed} prediction distribution: {dict(Counter(seed_preds))}")
        all_seed_preds.append(seed_preds) # 存入總預測矩陣中

        del model, tokenizer # 立即銷毀模型，釋放 GPU 顯存給下一個種子
        torch.cuda.empty_cache() # 清理顯存碎片

    # Hard voting across seeds: majority vote per sample
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    final_preds: List[str] = []
    for i in range(len(question_ids)):
        # 收集所有種子對同一題的答案，例如：['A', 'A', 'B']
        votes = [all_seed_preds[s][i] for s in range(len(all_seed_preds))]
        # 選出出現次數最多的答案，例如：'A'
        winner = Counter(votes).most_common(1)[0][0]
        final_preds.append(winner)

    pred_df = pd.DataFrame({
        'question_id': question_ids,
        'pred': [letter_to_idx[p] for p in final_preds],
    })
    pred_df.to_csv('predictions.csv', index=False) # 產出提交文件
    logger.info(f"Hard-voted multi-seed predictions saved to predictions.csv")
    logger.info(f"Final prediction distribution: {dict(Counter(final_preds))}")

# ============================================================================
# Main Training Function
# ============================================================================

def main(config: Config = None):
    """Main training function with LoRA finetuning"""
    if config is None:
        config = Config()

    logger.info("=" * 80)
    logger.info("PathoQA Training with Llama-3.2-1B-Instruct + LoRA")
    logger.info("=" * 80)

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # Load data
    train_data, val_data, test_data, benchmark_df = load_and_split_data(config)

    # Load tokenizer
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Load base model in bfloat16 for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False

    # ================================================================
    # Apply LoRA (Low-Rank Adaptation) for parameter-efficient finetuning
    # ================================================================
    # 在 Llama 模型中插入小的可訓練層，並凍結原本的大部分參數
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config) # 將 LoRA 應用到模型上
    model.print_trainable_parameters()

    config.tokenizer = tokenizer

    # Create datasets (short prompt for train, rich prompt for eval)
    logger.info("Creating datasets...")
    train_dataset = PathoQADataset(train_data, tokenizer, config.max_length, mode='train')
    val_dataset = PathoQADataset(val_data, tokenizer, config.max_length, mode='eval')
    test_dataset = PathoQADataset(test_data, tokenizer, config.max_length, mode='eval')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Optimizer (only LoRA parameters are trainable)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Linear warmup + cosine schedule
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy='cos',
    )

    # Training history
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': []
    }
    best_val_acc = 0.0
    no_improve_count = 0

    logger.info(f"\nStarting LoRA training for {config.num_epochs} epochs...\n")

    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info("-" * 50)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config)
        logger.info(f"Train Loss: {train_loss:.4f}")

        val_loss, val_acc, val_preds, val_labels = evaluate_epoch(model, val_loader, config)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Log prediction distribution
        pred_dist = Counter(val_preds)
        logger.info(f"Val prediction distribution: {dict(pred_dist)}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            best_path = os.path.join(config.output_dir, "best_lora")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info(f"New best model saved (acc={val_acc:.4f})")
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epoch(s)")

        # Save epoch checkpoint
        ckpt_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

        # Early stopping
        if no_improve_count >= config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Save final model
    final_path = os.path.join(config.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Final model saved to {final_path}")

    # ================================================================
    # Reload best checkpoint for evaluation & benchmark
    # ================================================================
    best_path = os.path.join(config.output_dir, "best_lora")
    if os.path.exists(best_path):
        logger.info(f"\nReloading best checkpoint from {best_path}...")
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            ),
            best_path,
        )
        model.eval()
        logger.info("Best checkpoint loaded.")

    # ================================================================
    # Evaluate on test set
    # ================================================================
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate_epoch(model, test_loader, config)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    test_pred_dist = Counter(test_preds)
    logger.info(f"Test prediction distribution: {dict(test_pred_dist)}")

    # ================================================================
    # Benchmark predictions
    # ================================================================
    logger.info("\nGenerating predictions on benchmark set...")
    benchmark_dataset = PathoQADataset(benchmark_df, tokenizer, config.max_length, include_labels=False, mode='infer')
    benchmark_loader = DataLoader(benchmark_dataset, batch_size=config.batch_size)

    bench_preds: List[str] = []
    device = get_model_device(model)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(benchmark_loader, desc="Benchmark"):
            preds = generate_predictions(
                model, tokenizer, list(batch['prompt']),
                max_new_tokens=config.max_new_tokens, device=device,
            )
            bench_preds.extend(preds)

    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    pred_df = benchmark_df[['question_id']].copy()
    pred_df['pred'] = [letter_to_idx[p] for p in bench_preds]
    pred_df.to_csv('predictions.csv', index=False)
    logger.info("Predictions saved to predictions.csv")
    bench_dist = Counter(bench_preds)
    logger.info(f"Benchmark prediction distribution: {dict(bench_dist)}")

    # ================================================================
    # Visualization
    # ================================================================
    logger.info("\nGenerating visualizations...")
    plot_training_metrics(history)
    plot_test_results(test_preds, test_labels)
    plot_answer_distribution(bench_preds, title="Benchmark Predictions")

    # Save history
    with open(f'{config.output_dir}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info("=" * 80)

    # Stage 2: keep strict split for model selection, then full-data submission training
    if config.do_full_data_submission:
        generate_full_data_submission(config, benchmark_df)

    return model, tokenizer, history


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_metrics(history: Dict):
    """Plot training metrics for each epoch"""
    epochs = history['epoch'] # 取得訓練輪次列表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 建立 2x2 的子圖矩陣，設定整體大小
    fig.suptitle('Training and Validation Metrics per Epoch', fontsize=16, fontweight='bold')

    # Training Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'o-', linewidth=2.5, markersize=10, color='#1f77b4')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss', fontweight='bold'); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(epochs)
    for i, v in enumerate(history['train_loss']):
        axes[0, 0].text(epochs[i], v, f'{v:.3f}', ha='center', va='bottom')

    # Validation Loss
    axes[0, 1].plot(epochs, history['val_loss'], 's-', linewidth=2.5, markersize=10, color='#ff7f0e')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss', fontweight='bold'); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(epochs)
    for i, v in enumerate(history['val_loss']):
        axes[0, 1].text(epochs[i], v, f'{v:.3f}', ha='center', va='bottom')

    # Validation Accuracy
    axes[1, 0].plot(epochs, history['val_accuracy'], '^-', linewidth=2.5, markersize=10, color='#2ca02c')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Validation Accuracy', fontweight='bold'); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(epochs); axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(history['val_accuracy']):
        axes[1, 0].text(epochs[i], v, f'{v:.3f}', ha='center', va='bottom')

    # Training vs Validation Loss
    axes[1, 1].plot(epochs, history['train_loss'], 'o-', linewidth=2.5, markersize=10,
                    color='#1f77b4', label='Train Loss')
    axes[1, 1].plot(epochs, history['val_loss'], 's-', linewidth=2.5, markersize=10,
                    color='#ff7f0e', label='Val Loss')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training vs Validation Loss', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_xticks(epochs)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    logger.info("Saved training_metrics.png")
    plt.close()


def plot_test_results(test_preds: List[str], test_labels: List[str]):
    """Plot confusion matrix and classification metrics for the test set."""
    labels_order = ['A', 'B', 'C', 'D']
    # 計算混淆矩陣：橫軸是模型預測值，縱軸是真實標籤
    cm = confusion_matrix(test_labels, test_preds, labels=labels_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Test Set Evaluation', fontsize=16, fontweight='bold')

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order,
                yticklabels=labels_order, ax=axes[0])
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')

    # Per-class accuracy bar chart
    per_class_acc = []
    for i, lbl in enumerate(labels_order):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class_acc.append(correct / total if total > 0 else 0)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = axes[1].bar(labels_order, per_class_acc, color=colors)
    axes[1].set_xlabel('Option'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-Option Accuracy'); axes[1].set_ylim([0, 1])
    # 畫出一條灰色虛線，代表總體平均準確率，方便對比各選項表現
    axes[1].axhline(y=accuracy_score(test_labels, test_preds), color='grey',
                     linestyle='--', label=f'Overall: {accuracy_score(test_labels, test_preds):.3f}')
    axes[1].legend() # 顯示圖例
    for bar, acc in zip(bars, per_class_acc):
        axes[1].text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f'{acc:.3f}',
                     ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    logger.info("Saved test_results.png")
    plt.close()


def plot_answer_distribution(preds: List[str], title: str = "Predictions"):
    """Plot answer distribution histogram."""
    labels_order = ['A', 'B', 'C', 'D']
    counts = Counter(preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [counts.get(l, 0) for l in labels_order]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(labels_order, vals, color=colors)
    ax.set_xlabel('Answer'); ax.set_ylabel('Count')
    ax.set_title(f'{title} – Answer Distribution', fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    fname = title.lower().replace(' ', '_') + '_distribution.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {fname}")
    plt.close()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    config = Config()

    logger.info("Configuration:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")
    logger.info("")

    model, tokenizer, history = main(config)
