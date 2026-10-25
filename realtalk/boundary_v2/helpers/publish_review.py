from pathlib import Path
import shutil
R=Path(__file__).resolve().parents[1];assert (R/'review/data.json').exists();assert (R/'review/review.woff2').exists();shutil.copyfile(R/'code/review.html',R/'review/index.html');print('REVIEW_READY',R/'review/index.html')
