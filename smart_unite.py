"""
SMART Test-Time Framework Implementation
Main script implementing the SMART strategy for selective LLM intervention

Based on euiin/SMART repository:
- Small Language Models (SLMs) perform step-by-step reasoning
- Large Language Models (LLMs) provide guidance only when necessary
- Enables up to 98.9% of LLM accuracy while reducing LLM token usage by up to 90%
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

from smart_config import SMARTConfig, get_smart_config
from smart_best_of_n import smart_best_of_n_batch
from utils.ans_process import gsm_parse_pred_ans, arc_parse_pred_ans, qa_parse_pred_ans
from utils.collate_fun import gsm_collate_fn, arc_collate_fn, piqa_collate_fn, qa_collate_fn
from utils.extract_response import gsm_extract_math_answer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_models(config: SMARTConfig) -> tuple:
    """Setup SLM and LLM models with tokenizers"""
    
    logger.info(f"Loading SLM from {config.slm_path}")
    slm_tokenizer = AutoTokenizer.from_pretrained(config.slm_path)
    slm_model = AutoModelForCausalLM.from_pretrained(
        config.slm_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval()
    
    logger.info(f"Loading LLM from {config.llm_path}")
    llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
    llm_model = AutoModelForCausalLM.from_pretrained(
        config.llm_path,
        torch_dtype=torch.float16,
        device_map="auto", 
        attn_implementation="flash_attention_2"
    ).eval()
    
    # Setup tokenizers
    for tokenizer in [slm_tokenizer, llm_tokenizer]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    
    return slm_tokenizer, slm_model, llm_tokenizer, llm_model


def get_collate_fn(test_set_path: str):
    """Get appropriate collate function based on dataset"""
    test_set_lower = test_set_path.lower()
    
    if 'gsm' in test_set_lower:
        return gsm_collate_fn
    elif 'triviaqa' in test_set_lower or 'nq' in test_set_lower:
        return qa_collate_fn  
    elif 'arc' in test_set_lower:
        return arc_collate_fn
    elif 'piqa' in test_set_lower:
        return piqa_collate_fn
    else:
        # Default to GSM format
        return gsm_collate_fn


def smart_collate_fn(batch):
    """SMART collate function that formats problems for step-by-step reasoning"""
    questions, answers = [], []
    
    for b in batch:
        # Handle different question field names
        if "question" in b:
            ques = b["question"]
        elif "problem" in b:
            ques = b["problem"]
        else:
            # Fallback: use first string field found
            for key, value in b.items():
                if isinstance(value, str) and len(value) > 10:
                    ques = value
                    break
            else:
                ques = str(b)
        
        # Format question for step-by-step reasoning
        prompt_q = f"Question: {ques}\nLet's think step by step.\n"
        questions.append(prompt_q)
        
        # Handle different answer field names  
        if "answer" in b:
            answers.append(b["answer"])
        elif "label" in b:
            answers.append(b["label"])
        else:
            answers.append("")
    
    return questions, answers


def smart_inference(config: SMARTConfig):
    """Run SMART inference on test dataset"""
    
    # Setup accelerator
    accelerator = Accelerator()
    
    # Load models
    slm_tokenizer, slm_model, llm_tokenizer, llm_model = setup_models(config)
    
    # Load dataset
    logger.info(f"Loading dataset from {config.test_set}")
    test_dataset = load_dataset("json", data_files=config.test_set)['train']
    
    # Create dataloader
    ds_loader = DataLoader(
        test_dataset, 
        batch_size=config.per_device_batch_size,
        collate_fn=smart_collate_fn,
        num_workers=2
    )
    ds_loader = accelerator.prepare_data_loader(ds_loader)
    
    # Setup output file
    output_dir = Path(config.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = []
    solution_list, pred_list, label_list, ori_ans_list, question_list = [], [], [], [], []
    
    # Process batches
    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader, desc="SMART Inference")
    else:
        iter_item = ds_loader
    
    for questions, answers in iter_item:
        # Convert batch to format expected by SMART
        batch_examples = {"problem": questions}
        
        # Run SMART best-of-n
        results = smart_best_of_n_batch(
            batch_examples, slm_tokenizer, slm_model,
            llm_tokenizer, llm_model, config
        )
        
        # Process results
        completions = results["completions"]
        predictions = results["pred"]
        smart_stats = results["smart_stats"]
        
        # Extract answers from gold standard
        ans_num = []
        for gold_ans in answers:
            if 'gsm' in config.test_set.lower():
                # Extract numerical answer for GSM8K
                import re
                match = re.search(r"answer is (\d+)", gold_ans, re.IGNORECASE)
                if match:
                    ans_num.append(float(match.group(1)))
                else:
                    # Try to extract number from #### format
                    match = re.search(r"#### (-?\d+)", gold_ans)
                    if match:
                        ans_num.append(float(match.group(1)))
                    else:
                        ans_num.append(float('nan'))
            else:
                ans_num.append(gold_ans)
        
        # Process predictions
        pred_num = []
        ans_list = []
        for pred_text in predictions:
            # Clean prediction text
            if 'Question' in pred_text:
                pred_text = pred_text.split('Question:')[0].strip()
            if 'Explanation' in pred_text:
                pred_text = pred_text.split('Explanation')[0].strip()
            
            ans_list.append(pred_text)
            
            # Extract numerical answer for GSM8K
            if 'gsm' in config.test_set.lower():
                pred_num.append(gsm_extract_math_answer(pred_text))
            else:
                pred_num.append(pred_text)
        
        # Store results
        label_list.extend(ans_num)
        ori_ans_list.extend(answers)  
        pred_list.extend(pred_num)
        solution_list.extend(ans_list)
        question_list.extend(questions)
        
        # Log batch results
        for i, (q, p, l) in enumerate(zip(questions, pred_num, ans_num)):
            logger.debug(f"Q: {q[:50]}... | Pred: {p} | Label: {l}")
    
    # Gather results from all processes
    logger.info("Gathering results from all processes...")
    accelerator.wait_for_everyone()
    
    gather_pred = gather_object(pred_list)
    gather_label = gather_object(label_list)
    gather_solution = gather_object(solution_list)
    gather_ori_solution = gather_object(ori_ans_list)
    gather_qs = gather_object(question_list)
    
    # Write results to file
    if accelerator.is_main_process:
        with open(config.output_file, "w", encoding="utf-8") as fw:
            for qs, pred, label, solution, ori_ans in zip(
                gather_qs, gather_pred, gather_label, gather_solution, gather_ori_solution
            ):
                fw.write(json.dumps({
                    "question": qs,
                    "original_sln": ori_ans,
                    "pred_solution": solution,
                    "pred": pred,
                    "label": label
                }, ensure_ascii=False) + "\n")
        
        logger.info(f"Results written to {config.output_file}")
        
        # Parse and evaluate results
        if 'gsm' in config.test_set.lower():
            gsm_parse_pred_ans(config.output_file)
        elif 'triviaqa' in config.test_set.lower() or 'nq' in config.test_set.lower():
            qa_parse_pred_ans(config.output_file)
        elif 'arc' in config.test_set.lower() or 'piqa' in config.test_set.lower():
            arc_parse_pred_ans(config.output_file)


def main():
    """Main entry point"""
    # Get configuration
    config = get_smart_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    logger.info("="*50)
    logger.info("SMART Test-Time Framework")
    logger.info("="*50)
    logger.info(f"SLM: {config.slm_path}")
    logger.info(f"LLM: {config.llm_path}")
    logger.info(f"Threshold: {config.threshold}")
    logger.info(f"Approach: {config.approach}")
    logger.info(f"Score method: {config.score_method}")
    logger.info(f"N candidates: {config.n}")
    logger.info(f"Dataset: {config.test_set}")
    logger.info("="*50)
    
    # Run inference
    smart_inference(config)
    
    logger.info("SMART inference completed successfully! 🔥")


if __name__ == "__main__":
    main()