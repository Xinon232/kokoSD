"""Take Koko's data block (weights, tables, tokenizer, fact table) out of a
released koko.gba, so the ROM can be rebuilt without the training files.

    python3 tools/blob_from_rom.py path/to/koko.gba [build/koko.bin]

The block is found by its sine table (engine/ref_int.py: 4,096 Q14 values at
O_SIN in gen/koko_layout.h); its size is KOKO_BLOB_BYTES. The model data is
CC BY-SA 4.0 (LICENSE-MODEL); the ROM itself is not kept in git."""
from pathlib import Path
import re, sys
import numpy as np
R = Path(__file__).resolve().parents[1]
lay = (R/'gen/koko_layout.h').read_text()
val = lambda n: int(re.search(rf'#define {n} (\d+)u', lay).group(1))
rom = Path(sys.argv[1]).read_bytes()
out = Path(sys.argv[2]) if len(sys.argv) > 2 else R/'build/koko.bin'
sin = np.round(np.sin(2 * np.pi * np.arange(4096) / 4096) * 16384).astype('<i2').tobytes()
at = rom.find(sin)
if at < 0 or rom.find(sin, at + 1) >= 0: sys.exit('sine table not found once: not a Koko ROM with this layout')
base, size = at - val('O_SIN'), val('KOKO_BLOB_BYTES')
if base < 0 or base + size > len(rom): sys.exit('data block outside the ROM: the layout does not match this ROM')
out.parent.mkdir(exist_ok=True)
out.write_bytes(rom[base:base + size])
print(f'{out}: {size} bytes from ROM offset {base:#x}')
