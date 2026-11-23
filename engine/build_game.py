"""Build build/koko.gba (the game): now tools/build.py (kokoSD, with gbaSD)."""
import runpy, sys
from pathlib import Path
sys.argv = [sys.argv[0]]
runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/build.py'), run_name='__main__')
