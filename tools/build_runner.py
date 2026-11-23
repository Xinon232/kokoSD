"""Build the emulator runners: build/runner (no SD card) and, with gbaSD's
card models, build/runner-sc, build/runner-ezfo, build/runner-edgba (card
image from the SD_IMAGE environment variable). Needs libmgba and its
headers. GPL-3.0-or-later."""
import subprocess
from pathlib import Path
R = Path(__file__).resolve().parents[1]
(R/'build').mkdir(exist_ok=True)
for name, model in (('runner', None), ('runner-sc', 1), ('runner-ezfo', 2), ('runner-edgba', 3)):
    subprocess.run(['gcc', '-O2', '-Wall', *([f'-DCART_MODEL={model}', '-Wno-unused-function', '-Wno-unused-variable'] if model else []),
                    '-o', str(R/'build'/name), str(R/'test/runner.c'), '-lmgba'], check=True)
    print('built', R/'build'/name)
