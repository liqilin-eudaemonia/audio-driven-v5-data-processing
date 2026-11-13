from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import adapter
adapter.install()
original=adapter.LEGACY/'evaluate28.py'
exec(compile(original.read_text(),str(original),'exec'),globals())
