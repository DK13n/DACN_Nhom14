
apt install gpustat
uv sync
source .venv/bin/activate
uvicorn pvcore.main:app --host 0.0.0.0 --port 8000 --reload