from datasets import load_dataset
from torch.utils.data import DataLoader, random_split


from model import build_transformer
from dataset import BilingualDataset
from tokenizer import get_or_build_tokenizer

def get_dataset(config):
    ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    # Build or load tokenizers
    tokenizer_src = get_or_build_tokenizer(
        config, ds_raw, config["lang_src"]
    )
    tokenizer_tgt = get_or_build_tokenizer(
        config, ds_raw, config["lang_tgt"]
    )

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size],
    )

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )

    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )
    max_len_src = max(
        [
            len(tokenizer_src.encode(item['translation'][config["lang_src"]]).ids)
            for item in ds_raw
        ]
    )
    max_len_tgt = max(
        [
            len(tokenizer_tgt.encode(item['translation'][config["lang_tgt"]]).ids)
            for item in ds_raw
        ]
    )
    print(f"Max source sequence length: {max_len_src}")
    print(f"Max target sequence length: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_size_src, vocab_size_tgt):
    transformer = build_transformer(
        src_vocab_size=vocab_size_src,
        tgt_vocab_size=vocab_size_tgt,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
    )

    return transformer
