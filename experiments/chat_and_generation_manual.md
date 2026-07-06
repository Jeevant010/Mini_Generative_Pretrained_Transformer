# Model Evaluation & Chat Manual

As the model progresses through its training phases (Pre-training $\rightarrow$ SFT $\rightarrow$ DPO), it is critical to evaluate its capabilities mathematically and qualitatively. This manual explains how you can interact with the model yourself and what the mathematical metrics mean.

## 1. Testing the Model Yourself (Interactive Chat)

To see the complete, un-truncated generation of words and talk to the model yourself, we have created an interactive chat script.

### How to run it:
1. Open your terminal.
2. Activate your conda environment (if not already active): `conda activate LLM`
3. Run the chat script:
   ```bash
   python chat.py
   ```
   
**What it does:** 
The script automatically finds the most advanced checkpoint in your `checkpoints/` folder (it prioritizes DPO over SFT). It will open a real-time prompt where you can type questions like *"What is the capital of France?"* and the model will generate the full, un-truncated response back to you.

## 2. Generating Mathematical Metrics

If you want to run the rigorous mathematical evaluations yourself at any phase to see if the model is degrading or improving:

### How to run it:
```bash
python -m evaluation.quality_metrics
```
*(You can also pass a specific checkpoint using `--checkpoint checkpoints/sft/best_sft_model.pt`)*

### Understanding the Metrics

* **Perplexity (PPL):** A measure of how "confused" the model is when reading text. 
  * *Healthy Range:* 20 - 40. 
  * *Danger Zone:* > 100 (Means the model is suffering catastrophic forgetting).
* **Distinct-1, Distinct-2, Distinct-3:** Measures how repetitive the vocabulary is (lexical diversity).
  * *Healthy Range:* D-2 > 0.6 and D-3 > 0.7.
  * *Danger Zone:* < 0.3 (Means the model is repeating the exact same phrases over and over).
* **Output Entropy:** Measures how confident the model is in its predictions.
  * *Healthy Range:* 5 - 10 bits.
  * *Danger Zone:* < 3 bits (The model is overly confident and will likely loop or produce robotic text).

By using `chat.py` and `evaluation/quality_metrics.py`, you have full transparency into the model's brain at every phase of the project!
