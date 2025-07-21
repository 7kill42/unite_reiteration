"""
SMART Utilities for Step-wise Reasoning
Based on euiin/SMART repository implementation
"""

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)


@dataclass 
class StepResult:
    """Result of a single reasoning step"""
    text: str
    score: float
    stop_reason: Optional[str] = None
    tokens_used: int = 0


@dataclass
class SMARTBeam:
    """Beam for SMART search with step-wise reasoning"""
    prompt: str
    index: int
    current_text: str = ""
    steps: List[str] = None
    scores: List[float] = None
    step_scores: List[List[float]] = None  # All step scores for PRM
    pruned: bool = False
    completed: bool = False
    stop_reasons: List[str] = None
    
    # SMART-specific tracking
    llm_interventions: List[int] = None  # Steps where LLM intervened
    intervention_details: List[Dict] = None  # Details of each intervention
    total_llm_tokens: int = 0
    total_slm_tokens: int = 0
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.scores is None:
            self.scores = []
        if self.step_scores is None:
            self.step_scores = []
        if self.stop_reasons is None:
            self.stop_reasons = []
        if self.llm_interventions is None:
            self.llm_interventions = []
        if self.intervention_details is None:
            self.intervention_details = []


class SMARTScorer:
    """Simple scoring mechanism for SMART framework"""
    
    def __init__(self, score_method='conf'):
        self.score_method = score_method
        
    def score_steps(self, prompts: List[str], completions: List[List[str]]) -> List[List[List[float]]]:
        """
        Score reasoning steps for each completion
        
        Args:
            prompts: List of problem prompts
            completions: List of completions for each prompt (each completion is list of steps)
            
        Returns:
            List of scores for each completion's steps
        """
        if self.score_method == 'conf':
            return self._confidence_scoring(prompts, completions)
        elif self.score_method == 'prm':
            return self._prm_scoring(prompts, completions)
        else:
            raise ValueError(f"Unknown scoring method: {self.score_method}")
    
    def _confidence_scoring(self, prompts: List[str], completions: List[List[str]]) -> List[List[List[float]]]:
        """Simple confidence-based scoring - placeholder implementation"""
        scores = []
        for completion_list in completions:
            completion_scores = []
            for completion in completion_list:
                steps = completion.split('\n\n') if completion else ['']
                # Simple heuristic: longer steps get higher confidence, but with some randomness
                step_scores = []
                for i, step in enumerate(steps):
                    if step.strip():
                        # Simulate confidence: longer steps = higher confidence, but decay over time
                        base_score = min(0.9, len(step) / 100.0 + 0.3)
                        decay = 0.95 ** i  # Decay confidence over steps
                        score = base_score * decay + np.random.normal(0, 0.05)
                        step_scores.append(max(0.1, min(0.95, score)))
                    else:
                        step_scores.append(0.1)
                completion_scores.append(step_scores)
            scores.append(completion_scores)
        return scores
    
    def _prm_scoring(self, prompts: List[str], completions: List[List[str]]) -> List[List[List[float]]]:
        """PRM-based scoring - placeholder implementation"""
        # For now, use the same as confidence scoring
        # In a real implementation, this would use a trained Process Reward Model
        return self._confidence_scoring(prompts, completions)
    
    def aggregate_scores(self, step_scores: List[float], strategy: str = "last") -> float:
        """Aggregate step scores into a single score"""
        if not step_scores:
            return 0.0
            
        if strategy == "last":
            return step_scores[-1]
        elif strategy == "mean":
            return np.mean(step_scores)
        elif strategy == "min":
            return np.min(step_scores)
        elif strategy == "prod":
            return np.prod(step_scores)
        else:
            return step_scores[-1]


def build_conversation(prompt: str, partial_completion: str, system_prompt: str) -> List[Dict[str, str]]:
    """Build conversation format for model input"""
    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    if partial_completion.strip():
        conv.append({"role": "assistant", "content": partial_completion})
        
    return conv


def generate_step(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, 
                 prompt: str, partial_completion: str, system_prompt: str,
                 max_tokens: int = 256, temperature: float = 0.0, 
                 stop_sequences: List[str] = None) -> StepResult:
    """
    Generate a single reasoning step
    
    Args:
        tokenizer: Model tokenizer
        model: Language model
        prompt: Original problem prompt
        partial_completion: Current partial solution
        system_prompt: System prompt
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        stop_sequences: Stop sequences for generation
        
    Returns:
        StepResult with generated text and metadata
    """
    if stop_sequences is None:
        stop_sequences = ["\n\n"]
    
    # Build conversation
    conv = build_conversation(prompt, partial_completion, system_prompt)
    
    # Apply chat template
    if partial_completion.strip():
        # Continue the assistant message
        formatted_input = tokenizer.apply_chat_template(
            conv, tokenize=False, continue_final_message=True
        )
    else:
        # Start new assistant message
        formatted_input = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
    
    # Tokenize input
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated part
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Apply stop sequences
    for stop_seq in stop_sequences:
        if stop_seq in generated_text:
            generated_text = generated_text.split(stop_seq)[0]
            break
    
    # Check for stop reasons
    stop_reason = None
    if not generated_text.strip():
        stop_reason = "empty"
    elif len(generated_ids) >= max_tokens - 1:
        stop_reason = "length"
    elif tokenizer.eos_token_id in generated_ids:
        stop_reason = "eos"
    
    return StepResult(
        text=generated_text,
        score=0.5,  # Placeholder, will be updated by scorer
        stop_reason=stop_reason,
        tokens_used=len(generated_ids)
    )


def should_intervene(scores: List[float], threshold: float, strategy: str = "last") -> bool:
    """
    Determine if LLM intervention is needed based on scores
    
    Args:
        scores: List of step scores
        threshold: Intervention threshold
        strategy: Strategy for determining intervention ("last", "any", "recent")
        
    Returns:
        True if intervention is needed
    """
    if not scores:
        return True  # Intervene if no scores available
    
    if strategy == "last":
        return scores[-1] < threshold
    elif strategy == "any":
        return any(score < threshold for score in scores)
    elif strategy == "recent":
        # Check last 2 scores
        recent_scores = scores[-2:] if len(scores) >= 2 else scores
        return any(score < threshold for score in recent_scores)
    else:
        return scores[-1] < threshold


def extract_steps(text: str) -> List[str]:
    """Extract reasoning steps from text (split by double newlines)"""
    if not text.strip():
        return []
    
    steps = text.split('\n\n')
    # Filter out empty steps
    steps = [step.strip() for step in steps if step.strip()]
    return steps


def combine_steps(steps: List[str]) -> str:
    """Combine reasoning steps back into text"""
    return '\n\n'.join(steps)


def truncate_to_step(text: str, step_index: int) -> str:
    """Truncate text to a specific step index"""
    steps = extract_steps(text)
    if step_index >= len(steps):
        return text
    
    return combine_steps(steps[:step_index + 1])


def estimate_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    """Estimate token count for text"""
    return len(tokenizer.encode(text))