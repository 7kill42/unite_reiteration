#!/usr/bin/env python3
"""
SMART Demo Script
Demonstrates the SMART test-time framework with minimal setup
"""

import json
import logging
import sys
from pathlib import Path

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_data():
    """Create sample GSM8K-like data for testing"""
    demo_data = [
        {
            "question": "John has 5 apples. He gives 2 to Mary. How many apples does John have left?",
            "answer": "John starts with 5 apples. He gives away 2 apples. So he has 5 - 2 = 3 apples left. The answer is 3"
        },
        {
            "question": "A train travels 60 miles in 2 hours. What is its average speed?",
            "answer": "The train travels 60 miles in 2 hours. Speed = distance / time = 60 / 2 = 30 miles per hour. The answer is 30"
        },
        {
            "question": "If 3 pencils cost $6, how much does 1 pencil cost?",
            "answer": "3 pencils cost $6. So 1 pencil costs $6 / 3 = $2. The answer is 2"
        }
    ]
    
    # Create demo data file
    demo_file = "/tmp/demo_data.jsonl"
    with open(demo_file, "w") as f:
        for item in demo_data:
            f.write(json.dumps(item) + "\n")
    
    return demo_file

def create_demo_prompt():
    """Create sample prompt file"""
    prompt_text = "You are a helpful assistant that solves mathematical problems step by step."
    prompt_file = "/tmp/demo_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(prompt_text)
    return prompt_file

def run_smart_demo():
    """Run SMART demo with simulated models"""
    try:
        from smart_config import SMARTConfig
        from smart_utils import SMARTScorer, extract_steps, combine_steps
        from smart_best_of_n import format_problem_with_prompt
        
        logger.info("🚀 Starting SMART Demo")
        
        # Create demo data
        demo_data_file = create_demo_data()
        demo_prompt_file = create_demo_prompt()
        logger.info(f"Created demo data: {demo_data_file}")
        
        # Setup configuration (without real models)
        config = SMARTConfig()
        config.test_set = demo_data_file
        config.prompts = demo_prompt_file
        config.output_file = "/tmp/smart_demo_output.jsonl"
        config.n = 2  # Generate 2 candidates per problem
        config.threshold = 0.4  # Intervention threshold
        
        logger.info(f"Configuration:")
        logger.info(f"  Threshold: {config.threshold}")
        logger.info(f"  Candidates (n): {config.n}")
        logger.info(f"  Score method: {config.score_method}")
        
        # Test SMART utilities
        logger.info("\n📊 Testing SMART Components:")
        
        # Test scorer
        scorer = SMARTScorer(score_method='conf')
        
        # Test step extraction
        sample_text = "First, I need to understand the problem.\n\nThen I can solve it step by step.\n\nFinally, I get the answer."
        steps = extract_steps(sample_text)
        logger.info(f"  ✓ Step extraction: {len(steps)} steps found")
        
        # Test scoring
        problems = ["What is 2+2?", "Calculate 5*3"]
        completions = [["2+2 equals 4"], ["5*3 equals 15"]]
        scores = scorer.score_steps(problems, completions)
        logger.info(f"  ✓ Scoring: Generated scores for {len(problems)} problems")
        
        # Test step recombination
        combined = combine_steps(steps)
        logger.info(f"  ✓ Step combination: {len(combined)} characters")
        
        # Simulate SMART workflow
        logger.info("\n🔄 Simulating SMART Workflow:")
        
        # Load demo problems
        with open(demo_data_file, "r") as f:
            problems = [json.loads(line)["question"] for line in f]
        
        logger.info(f"  Processing {len(problems)} problems...")
        
        # Simulate processing each problem
        results = []
        for i, problem in enumerate(problems):
            logger.info(f"\n  Problem {i+1}: {problem[:50]}...")
            
            # Simulate multiple candidates with different quality scores
            candidates = []
            for j in range(config.n):
                # Simulate reasoning steps
                steps = [
                    f"Let me understand this problem: {problem[:30]}...",
                    f"I need to calculate: [simulated calculation]",
                    f"Therefore, the answer is: [simulated answer]"
                ]
                
                completion = combine_steps(steps)
                
                # Simulate scores (some below threshold to trigger intervention)
                step_scores = [0.8, 0.6 - j*0.3, 0.7]  # Second candidate gets lower score
                overall_score = scorer.aggregate_scores(step_scores, "last")
                
                intervention_needed = overall_score < config.threshold
                candidates.append({
                    'completion': completion,
                    'scores': step_scores,
                    'overall_score': overall_score,
                    'intervention': intervention_needed
                })
                
                logger.info(f"    Candidate {j+1}: Score={overall_score:.3f}, "
                           f"Intervention={'needed' if intervention_needed else 'not needed'}")
            
            # Select best candidate (highest overall score)
            best_candidate = max(candidates, key=lambda x: x['overall_score'])
            results.append({
                'problem': problem,
                'prediction': best_candidate['completion'],
                'score': best_candidate['overall_score'],
                'interventions': sum(1 for c in candidates if c['intervention'])
            })
        
        # Calculate statistics
        total_interventions = sum(r['interventions'] for r in results)
        total_candidates = len(results) * config.n
        intervention_rate = total_interventions / total_candidates
        
        logger.info(f"\n📈 SMART Demo Results:")
        logger.info(f"  Problems processed: {len(results)}")
        logger.info(f"  Total candidates: {total_candidates}")
        logger.info(f"  Interventions needed: {total_interventions}")
        logger.info(f"  Intervention rate: {intervention_rate:.2%}")
        logger.info(f"  Simulated token efficiency: {(1-intervention_rate)*100:.1f}% SLM usage")
        
        # Write demo output
        with open(config.output_file, "w") as f:
            for result in results:
                f.write(json.dumps({
                    "question": result['problem'],
                    "pred_solution": result['prediction'],
                    "pred": "[simulated answer]",
                    "label": "[simulated label]",
                    "smart_score": result['score']
                }, ensure_ascii=False) + "\n")
        
        logger.info(f"  Results written to: {config.output_file}")
        logger.info("\n✅ SMART Demo completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_smart_demo()
    sys.exit(0 if success else 1)