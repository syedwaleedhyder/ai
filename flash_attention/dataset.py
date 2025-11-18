import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, data, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        self.data = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_target_pair = self.data[idx]
        src_text = src_target_pair['translation'][self.lang_src]
        tgt_text = src_target_pair['translation'][self.lang_tgt]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # for SOS or EOS

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length exceeded maximum limit.")
        
        enc_input = torch.cat([
            self.sos_token,
            torch.Tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(enc_num_padding_tokens)
        ])

        # Add SOS at the beginning for decoder input
        dec_input = torch.cat([
            self.sos_token,
            torch.Tensor(dec_input_tokens, dtype=torch.int64),
            self.pad_token.repeat(dec_num_padding_tokens)
        ])

        # Add EOS at the end for labels
        labels = torch.cat([
            torch.Tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(dec_num_padding_tokens)
        ])

        assert enc_input.shape[0] == self.seq_len
        assert dec_input.shape[0] == self.seq_len
        assert labels.shape[0] == self.seq_len

        return {
            "encoder_input": enc_input, # (seq_len,)
            "decoder_input": dec_input, # (seq_len,)
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            "label": labels, # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, 1, size, size), diagonal=1), dtype=torch.int)
    return mask == 0        