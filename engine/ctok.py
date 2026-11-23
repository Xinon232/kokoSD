"""Token decoding from the cartridge's own tokenizer tables (build/tok_dump on
build/koko.bin), standing in for the training tokenizer file, which is not in
this repository. Koko's C tokenizer was checked identical to the training one
on 65,819 strings (engine/DESIGN.md)."""
import subprocess
from pathlib import Path
R = Path(__file__).resolve().parents[1]


class Tok:
    def __init__(self):
        out = subprocess.run([str(R/'build/tok_dump'), str(R/'build/koko.bin')], capture_output=True, check=True).stdout
        self.text = [bytes.fromhex(l) for l in out.decode().split('\n')[:16384]]
        # The cartridge keeps no text for its special tokens (0 end of text,
        # 1 user, 2 Koko); the training tokenizer names them as below.
        for i, name in enumerate((b'<|endoftext|>', b'<|user|>', b'<|koko|>')): self.text[i] = name
        self.special = {0, 1, 2}

    def decode(self, ids, skip_special_tokens=True):
        return b''.join(self.text[i] for i in ids if not (skip_special_tokens and i in self.special)).decode('utf-8', 'replace')


def tokenizer():
    return Tok()
