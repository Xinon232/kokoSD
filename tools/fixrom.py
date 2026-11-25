"""Pad the ROM, put the Nintendo logo in the header (devkitPro's gbafix: the BIOS checks it when a cart or
loader such as SuperFW's "BIOS boot" starts the game through it) and write the header checksum."""
import os, subprocess, sys
d = bytearray(open(sys.argv[1], 'rb').read())
d[0xBD] = (-(sum(d[0xA0:0xBD]) + 0x19)) & 0xFF
while len(d) % 4096:
    d.append(0xFF)
open(sys.argv[1], 'wb').write(d)
subprocess.run([os.path.join(os.environ.get('DEVKITPRO', '/opt/devkitpro'), 'tools/bin/gbafix'), sys.argv[1]], check=True, capture_output=True)
# kokoSD 1.4: the self test's ROM sums, right after the image (which ends on a 4 KB boundary): "KSUM", the image
# size, the number of 1 MB pieces, then per piece the 32-bit sum of its words and the sum of word x (its index
# in the piece + 1), both mod 2^32. The game finds it at __rom_end rounded up to 4 KB.
if len(sys.argv) > 2 and sys.argv[2] == '--sums':
    import numpy as np, struct
    d = bytearray(open(sys.argv[1], 'rb').read())
    size = len(d); w = np.frombuffer(bytes(d), '<u4').astype(np.uint64)
    piece = 1 << 18                                     # words in 1 MB
    tab = [b'KSUM', struct.pack('<II', size, (len(w) + piece - 1) // piece)]
    for i in range(0, len(w), piece):
        c = w[i:i + piece]
        tab.append(struct.pack('<II', int(c.sum()) & 0xFFFFFFFF, int((c * np.arange(1, len(c) + 1, dtype=np.uint64)).sum()) & 0xFFFFFFFF))
    # kokoSD 1.4.1: the words a loader may change when it starts the game (SuperFW: the header's fixed fields and
    # checksum, and every 0x04000204 constant, its WAITCNT patch): their offsets and original values, so the self
    # test can put the originals back before comparing (at most 255).
    may = sorted(set(list(range(0xA0, 0xC0, 4)) + [4 * int(k) for k in np.nonzero(w == 0x04000204)[0]]))[:255]
    tab.append(struct.pack('<I', len(may)))
    tab += [struct.pack('<II', o, int(w[o // 4])) for o in may]
    d += b''.join(tab)
    while len(d) % 4096: d.append(0xFF)
    open(sys.argv[1], 'wb').write(d)
print(f'{sys.argv[1]}: {len(d)} bytes ({len(d) / 1024:.0f} KB)')
