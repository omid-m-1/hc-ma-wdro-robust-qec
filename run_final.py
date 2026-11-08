"""Default: the 20-task matched v9-final/seed107 matrix, not the old d3 learner."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
raise SystemExit(subprocess.call([sys.executable,str(ROOT/'study.py'),'run','--suite','reproduce-quoted-matrix','--preset','v9-final','--profile','matched']+sys.argv[1:]))
