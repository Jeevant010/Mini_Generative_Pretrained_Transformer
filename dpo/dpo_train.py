import os
import json
import copy
import time
import torch
import torch.nn.functional as F
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer


class DPODataset:
    """Load and batch preference pairs for DPO training."""

    def __init__(self, data_path, tokenizer, max_length):
        with open(data_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize_pair(self, example):
        """Tokenize a preference pair into (prompt_ids, chosen_ids, rejected_ids)."""
        prompt = example["prompt"]
        chosen = prompt + example["chosen"]
        rejected = prompt + example["rejected"]

        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        chosen_ids = self.tokenizer.encode(chosen, add_bos=True, add_eos=True)
        rejected_ids = self.tokenizer.encode(rejected, add_bos=True, add_eos=True)

        # Create labels (mask prompt portion)
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids[len(prompt_ids):]
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids[len(prompt_ids):]

        # Truncate
        chosen_ids = chosen_ids[:self.max_length]
        rejected_ids = rejected_ids[:self.max_length]
        chosen_labels = chosen_labels[:self.max_length]
        rejected_labels = rejected_labels[:self.max_length]

        return chosen_ids, chosen_labels, rejected_ids, rejected_labels

    def get_batch(self, batch_size):
        """Get a batch of preference pairs."""
        import random
        indices = random.sample(range(len(self.examples)), 
                                min(batch_size, len(self.examples)))

        batch_chosen_ids = []
        batch_chosen_labels = []
        batch_rejected_ids = []
        batch_rejected_labels = []

        for idx in indices:
            c_ids, c_labels, r_ids, r_labels = self._tokenize_pair(
                self.examples[idx]
            )
            batch_chosen_ids.append(c_ids)
            batch_chosen_labels.append(c_labels)
            batch_rejected_ids.append(r_ids)
            batch_rejected_labels.append(r_labels)

        # Pad each batch independently
        def pad_batch(ids_list, labels_list, pad_id=0):
            max_len = max(len(ids) for ids in ids_list)
            padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in ids_list]
            padded_labels = [lbl + [-100] * (max_len - len(lbl)) for lbl in labels_list]
            return (
                torch.tensor(padded_ids, dtype=torch.long),
                torch.tensor(padded_labels, dtype=torch.long),
            )

        chosen_x, chosen_y = pad_batch(batch_chosen_ids, batch_chosen_labels)
        rejected_x, rejected_y = pad_batch(batch_rejected_ids, batch_rejected_labels)

        device = config.device
        return (
            chosen_x.to(device), chosen_y.to(device),
            rejected_x.to(device), rejected_y.to(device),
        )


def compute_log_probs(model, input_ids, labels):
    """Compute sum of log probabilities for response tokens."""
    logits, _ = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    mask = (shift_labels != -100).float()
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp,
             ref_chosen_lp, ref_rejected_lp, beta=0.1):
    """
    Compute DPO loss.
    L = -log(sigma(beta * ((log pi(yw|x) - log ref(yw|x)) 
                          - (log pi(yl|x) - log ref(yl|x)))))
    """
    chosen_reward = beta * (policy_chosen_lp - ref_chosen_lp)
    rejected_reward = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss


def train_dpo():
    device = config.device
    beta = 0.1
    dpo_lr = 1e-5
    dpo_epochs = 1
    batch_size = 2  # DPO batches are memory-heavy (2 forward passes per step)

    # Load tokenizer
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config.TOKENIZER_PATH))
    tokenizer = BytePairTokenizer.load(tokenizer_path)

    # Load SFT model as policy (trainable)
    policy_model = GPTLanguageModel(config).to(device)
    
    sft_ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'sft', 'best_sft_model.pt'))
    print(f"Loading SFT model from {sft_ckpt_path}")
    sft_ckpt = torch.load(
        sft_ckpt_path,
        map_location=device, weights_only=False,
    )
    policy_model.load_state_dict(sft_ckpt["model_state_dict"])

    # Create reference model (frozen copy of SFT model)
    print("Cloning reference model...")
    ref_model = GPTLanguageModel(config).to(device)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load preference dataset
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'preference_pairs.json'))
    print(f"Loading preference dataset from {data_path}")
    dataset = DPODataset(
        data_path, tokenizer, config.block_size
    )

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=dpo_lr)

    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'dpo'))
    os.makedirs(ckpt_dir, exist_ok=True)
    
    policy_model.train()
    step = 0
    steps_per_epoch = len(dataset.examples) // batch_size

    print(f"Starting DPO Training: {dpo_epochs} epochs, {steps_per_epoch} steps per epoch.")

    for epoch in range(dpo_epochs):
        print(f"\n--- DPO Epoch {epoch + 1}/{dpo_epochs} ---")

        for batch_idx in range(steps_per_epoch):
            t0 = time.perf_counter()
            chosen_x, chosen_y, rejected_x, rejected_y = dataset.get_batch(
                batch_size
            )

            # Policy model log probs
            policy_chosen_lp = compute_log_probs(policy_model, chosen_x, chosen_y)
            policy_rejected_lp = compute_log_probs(
                policy_model, rejected_x, rejected_y
            )

            # Reference model log probs (no grad)
            with torch.no_grad():
                ref_chosen_lp = compute_log_probs(ref_model, chosen_x, chosen_y)
                ref_rejected_lp = compute_log_probs(
                    ref_model, rejected_x, rejected_y
                )

            loss = dpo_loss(
                policy_chosen_lp, policy_rejected_lp,
                ref_chosen_lp, ref_rejected_lp, beta,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            
            dt = time.perf_counter() - t0

            step += 1
            if step % 5 == 0:
                # Compute accuracy: how often does policy prefer chosen?
                with torch.no_grad():
                    chosen_r = policy_chosen_lp - ref_chosen_lp
                    rejected_r = policy_rejected_lp - ref_rejected_lp
                    accuracy = ((chosen_r - rejected_r) > 0).float().mean().item()

                print(
                    f"  Step {step:4d} | Loss: {loss.item():.4f} | "
                    f"Accuracy: {accuracy:.2%} | {dt*1000:.0f}ms", flush=True
                )
                
            if step % 500 == 0:
                torch.save({
                    "step": step,
                    "epoch": epoch,
                    "model_state_dict": policy_model.state_dict(),
                }, os.path.join(ckpt_dir, f"dpo_step_{step}.pt"))
                print(f"  [OK] Saved checkpoint step {step}", flush=True)

        # Save after epoch
        torch.save({
            "step": step,
            "epoch": epoch,
            "model_state_dict": policy_model.state_dict(),
        }, os.path.join(ckpt_dir, "best_dpo_model.pt"))
        print(f"  [OK] Saved epoch {epoch+1} checkpoint", flush=True)

    print("\nDPO Training Complete!")


if __name__ == "__main__":
    train_dpo()
