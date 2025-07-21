"""
SMART Configuration Module
Based on euiin/SMART repository strategy for selective LLM intervention
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class SMARTConfig:
    """Configuration for SMART test-time framework"""
    
    # Model paths
    slm_path: str = "Your small model path"  # Small Language Model path
    llm_path: str = "Your large model path"  # Large Language Model path
    
    # SMART strategy settings
    threshold: float = 0.3  # Confidence/score threshold for LLM intervention
    smart_search: bool = True  # Enable SMART selective intervention
    score_method: str = 'prm'  # 'prm' or 'conf' for scoring method
    
    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    n: int = 8  # Number of candidates for best-of-n
    beam_width: int = 4  # Beam width for beam search
    num_iterations: int = 10  # Max iterations for beam search
    lookahead: int = 1  # Lookahead steps
    
    # Search strategy
    approach: str = "best_of_n"  # 'best_of_n' or 'beam_search'
    agg_strategy: str = "last"  # Score aggregation strategy
    filter_duplicates: bool = True
    sort_completed: bool = True
    
    # System prompt
    system_prompt: str = "You are a helpful assistant that provides step-by-step solutions to mathematical problems."
    custom_chat_template: Optional[str] = None
    
    # Dataset and output settings
    test_set: str = "Your data path"
    prompts: str = "Your prompt path"
    output_file: str = "Your output file path"
    per_device_batch_size: int = 1
    search_batch_size: int = 4
    prm_batch_size: int = 16
    
    # Resource settings
    gpu_memory_utilization: float = 0.8
    seed: int = 42
    
    # PRM settings (for reward model)
    prm_model_path: str = "Your PRM model path"
    prm_threshold: float = 0.3  # Threshold for PRM-based intervention


def get_smart_config() -> SMARTConfig:
    """Parse command line arguments and return SMART configuration"""
    parser = argparse.ArgumentParser(description="SMART test-time framework configuration")
    
    # Model paths
    parser.add_argument("--slm_path", type=str, default="Your small model path",
                       help="Path to Small Language Model")
    parser.add_argument("--llm_path", type=str, default="Your large model path", 
                       help="Path to Large Language Model")
    
    # SMART strategy
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="Threshold for LLM intervention")
    parser.add_argument("--smart_search", action="store_true", default=True,
                       help="Enable SMART selective intervention")
    parser.add_argument("--score_method", type=str, default='prm', choices=['prm', 'conf'],
                       help="Scoring method: prm or confidence")
    
    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=8, help="Number of candidates")
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--lookahead", type=int, default=1)
    
    # Search strategy
    parser.add_argument("--approach", type=str, default="best_of_n", 
                       choices=["best_of_n", "beam_search"])
    parser.add_argument("--agg_strategy", type=str, default="last")
    
    # Dataset and output
    parser.add_argument("--test_set", type=str, default="Your data path")
    parser.add_argument("--prompts", type=str, default="Your prompt path")
    parser.add_argument("--output_file", type=str, default="Your output file path")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--search_batch_size", type=int, default=4)
    parser.add_argument("--prm_batch_size", type=int, default=16)
    
    # Resource settings
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    
    # PRM settings
    parser.add_argument("--prm_model_path", type=str, default="Your PRM model path")
    parser.add_argument("--prm_threshold", type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Convert argparse namespace to SMARTConfig dataclass
    config = SMARTConfig()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    return config