"""Package this project without temporary caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
out=ROOT.parent/'Karishma_Sawant_AI_QE_BFSI_Capstone.zip'
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for f in ROOT.rglob('*'):
        if f.is_file() and not any(x in f.parts for x in ('__pycache__','.pytest_cache','.venv')):
            z.write(f,arcname=Path('Karishma_Sawant_AI_QE_BFSI_Capstone')/f.relative_to(ROOT))
print(out)
