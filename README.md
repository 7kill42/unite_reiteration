
# unite_reiteration

复现论文：DETERMINE-THEN-ENSEMBLE: NECESSITY OF TOP-K UNION FOR LARGE LANGUAGE MODEL ENSEMBLING

## 🚀 NEW: SMART Strategy Implementation

This repository now also implements the **SMART** (Small Reasons, Large Hints) test-time framework from [euiin/SMART](https://github.com/euiin/SMART), enabling selective LLM intervention for more efficient reasoning.

### SMART vs Original Ensemble Approach

| Feature | Original Ensemble | SMART Framework |
|---------|------------------|----------------|  
| **Level** | Token-by-token | Step-by-step |
| **Model Usage** | All models active always | SLM primary, LLM selective |
| **Efficiency** | Full computation cost | Up to 90% LLM token reduction |
| **Accuracy** | Full ensemble accuracy | Up to 98.9% of LLM accuracy |
| **Strategy** | Weighted averaging | Threshold-based intervention |

### Quick Start with SMART

```bash
# Run SMART framework
python smart_unite.py \
    --slm_path "path/to/small_model" \
    --llm_path "path/to/large_model" \
    --threshold 0.3 \
    --test_set "datasets/GSM/test.jsonl"

# Demo without real models
python smart_demo.py
```

See [SMART_README.md](SMART_README.md) for detailed documentation.

---

## Original Repository Content
