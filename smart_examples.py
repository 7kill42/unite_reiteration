#!/usr/bin/env python3
"""
SMART Usage Example
Shows how to use the SMART framework for different datasets and configurations
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr)
                
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Main function demonstrating different SMART usage scenarios"""
    
    print("🚀 SMART Framework Usage Examples")
    print("=" * 60)
    
    # Example 1: Basic demo
    success = run_command(
        ["python", "smart_demo.py"],
        "Basic SMART Demo (no models required)"
    )
    
    if not success:
        print("\n❌ Basic demo failed. Check installation.")
        return 1
    
    print(f"\n📋 Additional Usage Examples:")
    print("-" * 60)
    
    # Example configurations for different scenarios
    examples = [
        {
            "name": "GSM8K Dataset with SMART Best-of-N",
            "cmd": [
                "python", "smart_unite.py",
                "--slm_path", "microsoft/DialoGPT-small", 
                "--llm_path", "microsoft/DialoGPT-medium",
                "--approach", "best_of_n",
                "--n", "8",
                "--threshold", "0.3",
                "--score_method", "conf",
                "--test_set", "datasets/GSM/test.jsonl",
                "--output_file", "results/gsm_smart.jsonl"
            ]
        },
        {
            "name": "ARC Dataset with Low Intervention Threshold", 
            "cmd": [
                "python", "smart_unite.py",
                "--slm_path", "microsoft/DialoGPT-small",
                "--llm_path", "microsoft/DialoGPT-large", 
                "--approach", "best_of_n",
                "--n", "4",
                "--threshold", "0.2",  # Lower threshold = more interventions
                "--score_method", "prm",
                "--test_set", "datasets/ARC/test.jsonl",
                "--output_file", "results/arc_smart.jsonl"
            ]
        },
        {
            "name": "High Efficiency Mode (Less LLM Usage)",
            "cmd": [
                "python", "smart_unite.py", 
                "--slm_path", "microsoft/DialoGPT-small",
                "--llm_path", "microsoft/DialoGPT-large",
                "--approach", "best_of_n", 
                "--n", "4",
                "--threshold", "0.1",  # High threshold = fewer interventions
                "--score_method", "conf",
                "--temperature", "0.0",
                "--max_tokens", "256",
                "--test_set", "datasets/PIQA/test.jsonl",
                "--output_file", "results/piqa_efficient.jsonl"
            ]
        }
    ]
    
    # Display example commands
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print("   " + " \\\n   ".join(example['cmd']))
    
    print(f"\n📊 Key Parameters:")
    print("-" * 30)
    print("--threshold    : Confidence threshold (0.1-0.9)")  
    print("               : Lower = more LLM interventions")
    print("               : Higher = fewer LLM interventions")
    print("--n           : Number of candidates (2-16)")
    print("--score_method: 'conf' (confidence) or 'prm' (process reward model)")
    print("--approach    : 'best_of_n' or 'beam_search'")
    print("--temperature : Generation randomness (0.0-1.0)")
    
    print(f"\n🎯 Typical Threshold Values:")
    print("-" * 30)
    print("0.1-0.2: High intervention (more LLM usage, higher accuracy)")
    print("0.3-0.4: Balanced (recommended starting point)")  
    print("0.5-0.7: Low intervention (more SLM usage, higher efficiency)")
    print("0.8-0.9: Minimal intervention (maximum efficiency)")
    
    print(f"\n📁 Expected File Structure:")
    print("-" * 30)
    print("datasets/")
    print("  ├── GSM/test.jsonl")
    print("  ├── ARC/test.jsonl") 
    print("  └── PIQA/test.jsonl")
    print("models/")
    print("  ├── small_model/")
    print("  └── large_model/")
    print("results/")
    print("  └── [output files]")
    
    print(f"\n✨ Next Steps:")
    print("-" * 30)
    print("1. Prepare your datasets in JSONL format")
    print("2. Download or specify model paths")
    print("3. Run smart_demo.py to verify installation")
    print("4. Experiment with different thresholds")
    print("5. Compare results with original unite2.py/unite3.py")
    
    print(f"\n🔗 Resources:")
    print("-" * 30) 
    print("- SMART_README.md: Detailed documentation")
    print("- smart_demo.py: Working demo without models")
    print("- smart_config.py: All configuration options")
    print("- Original SMART: https://github.com/euiin/SMART")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())