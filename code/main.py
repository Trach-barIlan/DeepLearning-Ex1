from __future__ import annotations

import argparse
import math
import os
import random
import torch

from hyperparams import defaults as hparams


def parse_args():
    parser = argparse.ArgumentParser(description="Train a character-level transformer language model")

    parser.add_argument("--data-path", type=str, default=hparams.DATA_PATH, help="Path containing *.txt training files")
    parser.add_argument("--seq-len", type=int, default=hparams.SEQ_LEN, help="Context length")
    parser.add_argument("--batch-size", type=int, default=hparams.BATCH_SIZE, help="Batch size")
    parser.add_argument("--num-batches", type=int, default=hparams.NUM_BATCHES, help="Number of optimization steps to run")

    parser.add_argument("--n-layers", type=int, default=hparams.N_LAYERS, help="Number of decoder blocks")
    parser.add_argument("--n-heads", type=int, default=hparams.N_HEADS, help="Attention heads per block")
    parser.add_argument("--embed-size", type=int, default=hparams.EMBED_SIZE, help="Embedding size")
    parser.add_argument("--mlp-hidden-size", type=int, default=hparams.MLP_HIDDEN_SIZE, help="MLP hidden size")
    parser.add_argument("--dropout", type=float, default=hparams.DROPOUT, help="Dropout probability inside transformer blocks")

    parser.add_argument("--learning-rate", type=float, default=hparams.LEARNING_RATE, help="AdamW learning rate")
    parser.add_argument("--gradient-clipping", type=float, default=hparams.GRADIENT_CLIPPING, help="Gradient clipping max-norm")

    parser.add_argument("--checkpoint-dir", type=str, default=hparams.CHECKPOINT_DIR, help="Directory to write checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=hparams.CHECKPOINT_EVERY, help="Save checkpoint every N batches (<=0 disables periodic checkpoints)")
    parser.add_argument("--resume-from-checkpoint", type=str, default=hparams.RESUME_FROM_CHECKPOINT, help="Path to a .pt checkpoint to resume from")
    parser.add_argument("--resumed-learning-rate", type=float, default=hparams.RESUMED_LEARNING_RATE, help="Optional LR override when resuming")

    parser.add_argument("--sample-prefix", type=str, default=hparams.SAMPLE_PREFIX, help="Prefix text for periodic qualitative sampling")
    parser.add_argument("--sample-tokens", type=int, default=hparams.SAMPLE_TOKENS, help="Number of tokens to generate for periodic sample")
    parser.add_argument("--sample-every", type=int, default=hparams.SAMPLE_EVERY, help="Run sampling every N batches")
    parser.add_argument("--sample-temperature", type=float, default=hparams.SAMPLE_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--sample-topk", type=int, default=hparams.SAMPLE_TOPK, help="Top-k sampling cutoff")

    parser.add_argument("--validation-split", type=float, default=hparams.VALIDATION_SPLIT, help="Fraction of sequences held out for validation")
    parser.add_argument("--validation-every", type=int, default=hparams.VALIDATION_EVERY, help="Run validation every N batches")
    parser.add_argument("--validation-batches", type=int, default=hparams.VALIDATION_BATCHES, help="Number of validation batches to average")

    return parser.parse_args()


def evaluate_model(model, data_iter, batch_size, device, n_batches):
    if n_batches <= 0:
        return None, None

    model.eval()
    batch_iter = data.batch_items(data_iter, batch_size)
    losses = []

    with torch.no_grad():
        for _ in range(n_batches):
            batch = next(batch_iter)
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device).long()
            batch_y = batch_y.to(device).long()
            logits = model(batch_x)
            losses.append(lm.compute_loss(logits, batch_y).item())

    model.train()
    validation_loss = sum(losses) / len(losses)
    validation_perplexity = math.exp(validation_loss)
    return validation_loss, validation_perplexity


if __name__ == "__main__":
    import lm
    from torch import optim
    from transformer import TransformerLM

    import data

    args = parse_args()

    seq_len = args.seq_len
    batch_size = args.batch_size
    data_path = args.data_path
    n_layers = args.n_layers
    n_heads = args.n_heads
    embed_size = args.embed_size
    mlp_hidden_size = args.mlp_hidden_size
    dropout = args.dropout

    learning_rate = args.learning_rate
    gradient_clipping = args.gradient_clipping
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    num_batches_to_train = args.num_batches

    checkpoint_dir = args.checkpoint_dir
    checkpoint_every = args.checkpoint_every
    resume_from_checkpoint = args.resume_from_checkpoint
    resumed_learning_rate = args.resumed_learning_rate
    validation_split = args.validation_split
    validation_every = args.validation_every
    validation_batches = args.validation_batches

    def save_checkpoint(path, model, optimizer, tokenizer, model_config, seen_batches, loss_value):
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer_state": tokenizer.state_dict(),
            "model_config": model_config,
            "num_batches": seen_batches,
            "last_loss": float(loss_value),
        }
        torch.save(checkpoint, path)

    model_config = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "embed_size": embed_size,
        "max_context_len": seq_len,
        "mlp_hidden_size": mlp_hidden_size,
        "dropout": dropout,
        "with_residuals": True,
        "pre_norm": True,
    }

    start_batch = 0
    if resume_from_checkpoint is not None:
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)

        tokenizer = data.CharTokenizer()
        tokenizer.load_state_dict(checkpoint["tokenizer_state"])
        tokenized_data = data.tokenize_data(data_path, tokenizer)

        model_config = checkpoint["model_config"]
        seq_len = model_config["max_context_len"]
        model = TransformerLM(
            model_config["n_layers"],
            model_config["n_heads"],
            model_config["embed_size"],
            model_config["max_context_len"],
            tokenizer.vocab_size(),
            model_config["mlp_hidden_size"],
            with_residuals=model_config["with_residuals"],
            pre_norm=model_config["pre_norm"],
            dropout=model_config.get("dropout", dropout),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])

        start_batch = int(checkpoint.get("num_batches", 0))
        print(f"Resuming from checkpoint {resume_from_checkpoint} at batch {start_batch}")
    else:
        tokenizer, tokenized_data = data.load_data(data_path)
        model: torch.nn.Module = TransformerLM(
            model_config["n_layers"],
            model_config["n_heads"],
            model_config["embed_size"],
            model_config["max_context_len"],
            tokenizer.vocab_size(),
            model_config["mlp_hidden_size"],
            with_residuals=model_config["with_residuals"],
            pre_norm=model_config["pre_norm"],
            dropout=model_config.get("dropout", dropout),
        ).to(device)

    shuffled_data = list(tokenized_data)
    random.shuffle(shuffled_data)
    validation_count = int(len(shuffled_data) * validation_split)
    if len(shuffled_data) > 1 and validation_split > 0 and validation_count == 0:
        validation_count = 1
    if validation_count >= len(shuffled_data):
        validation_count = max(0, len(shuffled_data) - 1)

    validation_data = shuffled_data[:validation_count]
    train_data = shuffled_data[validation_count:] if validation_count > 0 else shuffled_data

    # NOTE: data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))
    validation_iter = iter(data.RandomOrderDataIterator(validation_data, seq_len + 1)) if len(validation_data) > 0 else None

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    if resume_from_checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if resumed_learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = resumed_learning_rate
            print(f"Overriding resumed learning rate to {resumed_learning_rate}")

    model.train()

    num_batches = start_batch
    os.makedirs(checkpoint_dir, exist_ok=True)
    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device).long()
            batch_y = batch_y.to(device).long()

            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1
            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")

            if validation_iter is not None and validation_every > 0 and num_batches % validation_every == 0:
                validation_loss, validation_perplexity = evaluate_model(
                    model,
                    validation_iter,
                    batch_size,
                    device,
                    validation_batches,
                )
                print(
                    f"Validation loss: {validation_loss:.4f}; validation perplexity: {validation_perplexity:.4f}"
                )

            if args.sample_every > 0 and num_batches % args.sample_every == 0:
                for _ in range(1):
                    model.eval()
                    sampled = tokenizer.detokenize(
                        model.better_sample_continuation(
                            tokenizer.tokenize(args.sample_prefix),
                            args.sample_tokens,
                            temperature=args.sample_temperature,
                            topK=args.sample_topk,
                        )
                    )
                    model.train()
                    print(f"Model sample: '''{sampled}'''")
                print("")

            if checkpoint_every > 0 and num_batches % checkpoint_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{num_batches}.pt")
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    tokenizer,
                    model_config,
                    num_batches,
                    loss.item(),
                )
                print(f"Saved checkpoint: {checkpoint_path}")

        if num_batches >= num_batches_to_train:
            break

    final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        tokenizer,
        model_config,
        num_batches,
        loss.item(),
    )
    print(f"Training complete. Final checkpoint saved: {final_checkpoint_path}")
