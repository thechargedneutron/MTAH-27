"""Input and working directories. MTAH_DATA: the competition data bundle (with tactical/ and broadcast/);
MTAH_WORK: where intermediate files are written. Both default to folders inside the repository."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("MTAH_DATA", os.path.join(ROOT, "data"))
WORK = os.environ.get("MTAH_WORK", os.path.join(ROOT, "work"))
