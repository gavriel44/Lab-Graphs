import os
import tempfile

# Select the backend before any test imports pyplot; keep font caches writable.
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="labgraphs-mpl-"))
