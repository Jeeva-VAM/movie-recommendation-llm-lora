# movie-recommendation-llm-lora
Instruction-tuned Movie Recommendation LLM using TinyLlama + LoRA
# Instruction-Tuned Movie Recommendation LLM

This project fine-tunes a quantized **TinyLlama-1.1B-Chat** model using **LoRA (PEFT)** to generate movie recommendations from instruction-style prompts.

## Highlights
- LoRA fine-tuning with only **0.1% trainable parameters**
- 4-bit quantization for training on **Kaggle free GPU**
- Instruction-style dataset built from IMDb metadata
- Restart-safe deployment using Kaggle Datasets

## Architecture
IMDb CSV → Prompt Builder → Tokenizer → TinyLlama (Frozen) → LoRA Adapters → Movie Recommendations

## Tech Stack
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- bitsandbytes
- Kaggle

## Training Details
- Base model: TinyLlama-1.1B-Chat
- Trainable params: ~1.1M
- Dataset size: ~6k instruction samples
- Training time: ~40 minutes (Kaggle T4)

## Example
- ###Question: Recommend a science fiction movie with a twist.
  ###Answer: "The Martian" (2015) - A film about an astronaut stranded on Mars who must use his ingenuity and resourcefulness to survive. The twist is that the astronaut's wife was killed in a tragic accident before he left Earth, and he had to make his way back to Earth alone. This movie is a great example of how a twist can add depth and complexity to a story.

## How to Load the Fine-Tuned Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(base)

base_model = AutoModelForCausalLM.from_pretrained(
    base,
    load_in_4bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "PATH_TO_ADAPTER"
)

## Limitations
- Not fact-perfect (LLM hallucinations)
- LoRA adapts behavior, not full knowledge
- Best combined with RAG for accuracy

## Future Work
- Add Retrieval-Augmented Generation (RAG)
- Deploy with FastAPI
- Upload adapter to Hugging Face Hub
