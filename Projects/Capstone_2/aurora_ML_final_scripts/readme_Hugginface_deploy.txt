Hugging Face Space (Docker) Deployment — Aurora API
====================================================

1) Create a Docker Space (if not already created)
   - Name example: aurora-api
   - Type: Docker

2) Clone the Space repo
   git clone https://huggingface.co/spaces/<USER>/<SPACE_NAME>
   cd <SPACE_NAME>

3) Copy files from this project into the Space repo
   - aurora_final_scripts/3_serve.py
   - aurora_final_scripts/Dockerfile
   - aurora_final_scripts/models/
   - aurora_final_scripts/results/

4) Commit & push
   git add .
   git commit -m "Deploy aurora API (FastAPI + Docker)"
   git push

Notes
-----
- Hugging Face Docker Spaces require an open port (use 7860).
- The server already runs on 7860 in Dockerfile and 3_serve.py.
- Update BASE_URL in any test script if you want to hit the HF Space.

Example BASE_URL
----------------
https://huggingface.co/spaces/<USER>/<SPACE_NAME>
