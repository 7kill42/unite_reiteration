"""
SMART Best-of-N Implementation
Based on euiin/SMART repository strategy
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from smart_config import SMARTConfig
from smart_utils import (
    SMARTScorer, generate_step, should_intervene, 
    extract_steps, combine_steps, estimate_tokens
)

logger = logging.getLogger(__name__)


def smart_best_of_n_inference(
    problems: List[str], 
    slm_tokenizer: AutoTokenizer,
    slm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer, 
    llm_model: AutoModelForCausalLM,
    config: SMARTConfig
) -> Dict[str, Any]:
    """
    SMART best-of-n inference with selective LLM intervention
    
    Args:
        problems: List of problem prompts
        slm_tokenizer: Small model tokenizer
        slm_model: Small language model
        llm_tokenizer: Large model tokenizer  
        llm_model: Large language model
        config: SMART configuration
        
    Returns:
        Dictionary with completions and predictions
    """
    scorer = SMARTScorer(score_method=config.score_method)
    results = {"completions": [], "pred": [], "smart_stats": []}
    
    for problem in tqdm(problems, desc="Processing problems"):
        # Generate initial completions with SLM
        completions = []
        intervention_stats = []
        
        for candidate_idx in range(config.n):
            completion, stats = _generate_smart_completion(
                problem, slm_tokenizer, slm_model, llm_tokenizer, llm_model, 
                config, scorer, candidate_idx
            )
            completions.append(completion)
            intervention_stats.append(stats)
        
        # Score all completions
        completion_texts = [[comp] for comp in completions]
        all_scores = scorer.score_steps([problem], completion_texts)[0]
        
        # Aggregate scores for each completion
        agg_scores = []
        for completion_scores in all_scores:
            if completion_scores and completion_scores[0]:
                agg_score = scorer.aggregate_scores(completion_scores[0], config.agg_strategy)
            else:
                agg_score = 0.0
            agg_scores.append(agg_score)
        
        # Select best completion
        best_idx = np.argmax(agg_scores)
        best_completion = completions[best_idx]
        
        results["completions"].append(completions)
        results["pred"].append(best_completion)
        results["smart_stats"].append(intervention_stats)
        
        # Log statistics
        total_interventions = sum(len(stats['interventions']) for stats in intervention_stats)
        total_llm_tokens = sum(stats['llm_tokens'] for stats in intervention_stats)
        total_slm_tokens = sum(stats['slm_tokens'] for stats in intervention_stats)
        
        logger.info(f"Problem completed: {total_interventions} interventions, "
                   f"LLM tokens: {total_llm_tokens}, SLM tokens: {total_slm_tokens}")
    
    return results


def _generate_smart_completion(
    problem: str,
    slm_tokenizer: AutoTokenizer,
    slm_model: AutoModelForCausalLM, 
    llm_tokenizer: AutoTokenizer,
    llm_model: AutoModelForCausalLM,
    config: SMARTConfig,
    scorer: SMARTScorer,
    candidate_idx: int
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a single completion using SMART strategy
    
    Returns:
        Tuple of (completion_text, intervention_stats)
    """
    current_completion = ""
    intervention_stats = {
        'interventions': [],
        'llm_tokens': 0,
        'slm_tokens': 0,
        'candidate_idx': candidate_idx
    }
    
    # Generate step by step
    for step_idx in range(config.num_iterations):
        # Generate next step with SLM
        step_result = generate_step(
            slm_tokenizer, slm_model, problem, current_completion,
            config.system_prompt, max_tokens=128, temperature=config.temperature
        )
        
        if not step_result.text.strip():
            break
            
        # Update completion with SLM step
        temp_completion = current_completion
        if temp_completion and not temp_completion.endswith('\n\n'):
            temp_completion += '\n\n'
        temp_completion += step_result.text.strip()
        
        intervention_stats['slm_tokens'] += step_result.tokens_used
        
        # Score the current completion to decide on intervention
        completion_steps = extract_steps(temp_completion)
        if completion_steps:
            step_scores = scorer.score_steps([problem], [[temp_completion]])[0][0]
            
            # Check if intervention is needed
            if should_intervene(step_scores, config.threshold, "last"):
                # LLM intervention needed
                logger.debug(f"Step {step_idx}: LLM intervention triggered (score: {step_scores[-1]:.3f})")
                
                # Generate corrected step with LLM
                llm_step_result = generate_step(
                    llm_tokenizer, llm_model, problem, current_completion,
                    config.system_prompt, max_tokens=256, temperature=config.temperature
                )
                
                if llm_step_result.text.strip():
                    # Use LLM step instead
                    if current_completion and not current_completion.endswith('\n\n'):
                        current_completion += '\n\n'
                    current_completion += llm_step_result.text.strip()
                    
                    intervention_stats['llm_tokens'] += llm_step_result.tokens_used
                    intervention_stats['interventions'].append({
                        'step': step_idx,
                        'slm_text': step_result.text,
                        'llm_text': llm_step_result.text,
                        'slm_score': step_scores[-1] if step_scores else 0.0,
                        'reason': 'low_score'
                    })
                else:
                    # Fall back to SLM step if LLM generates nothing
                    current_completion = temp_completion
            else:
                # Use SLM step
                current_completion = temp_completion
        else:
            # No steps yet, use SLM step
            current_completion = temp_completion
        
        # Check for completion
        if step_result.stop_reason in ['eos', 'empty'] or len(extract_steps(current_completion)) >= 10:
            break
    
    return current_completion, intervention_stats


def format_problem_with_prompt(problem: str, prompt_template: str = None) -> str:
    """Format problem with appropriate prompt template"""
    if prompt_template is None:
        return f"Question: {problem}\nLet's think step by step.\n"
    
    return prompt_template.format(question=problem)


def smart_best_of_n_batch(
    examples: Dict[str, List[str]],
    slm_tokenizer: AutoTokenizer,
    slm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    llm_model: AutoModelForCausalLM, 
    config: SMARTConfig
) -> Dict[str, Any]:
    """
    Process a batch of examples using SMART best-of-n
    
    Args:
        examples: Dictionary with 'problem' key containing list of problems
        slm_tokenizer: Small model tokenizer
        slm_model: Small language model
        llm_tokenizer: Large model tokenizer
        llm_model: Large language model
        config: SMART configuration
        
    Returns:
        Dictionary with results
    """
    problems = examples["problem"]
    
    # Run SMART inference
    results = smart_best_of_n_inference(
        problems, slm_tokenizer, slm_model, 
        llm_tokenizer, llm_model, config
    )
    
    # Calculate overall statistics
    all_stats = results["smart_stats"]
    total_problems = len(problems)
    total_interventions = sum(len(stats['interventions']) for problem_stats in all_stats 
                             for stats in problem_stats)
    total_llm_tokens = sum(stats['llm_tokens'] for problem_stats in all_stats 
                          for stats in problem_stats)
    total_slm_tokens = sum(stats['slm_tokens'] for problem_stats in all_stats 
                          for stats in problem_stats)
    
    intervention_rate = total_interventions / (total_problems * config.n) if total_problems > 0 else 0
    token_efficiency = total_slm_tokens / (total_llm_tokens + total_slm_tokens) if (total_llm_tokens + total_slm_tokens) > 0 else 1.0
    
    logger.info(f"SMART Best-of-N Statistics:")
    logger.info(f"  Problems processed: {total_problems}")
    logger.info(f"  Total candidates: {total_problems * config.n}")
    logger.info(f"  Total interventions: {total_interventions}")
    logger.info(f"  Intervention rate: {intervention_rate:.3f}")
    logger.info(f"  LLM tokens: {total_llm_tokens}")
    logger.info(f"  SLM tokens: {total_slm_tokens}")
    logger.info(f"  Token efficiency (SLM ratio): {token_efficiency:.3f}")
    
    # Add statistics to results
    results["overall_stats"] = {
        "total_problems": total_problems,
        "total_interventions": total_interventions,
        "intervention_rate": intervention_rate,
        "llm_tokens": total_llm_tokens,
        "slm_tokens": total_slm_tokens,
        "token_efficiency": token_efficiency
    }
    
    return results