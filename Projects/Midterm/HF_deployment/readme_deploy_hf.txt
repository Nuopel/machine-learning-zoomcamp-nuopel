Crée un Space Docker si pas déjà fait : wine-rating-api, type Docker.

Clone le repo HF :
git clone https://huggingface.co/spaces/Nuopel/wine-rating-api
cd wine-rating-api

Copie ton dossier src + Dockerfile + requirements.txt dans ce repo.

Commit & push :
git add .
git commit -m "Deploy LinearRegression via Gradio"
git push


The port in predict and docker need to be one accepted by huggingface for exemple : 7860


BASE_URL = "https://huggingface.co/spaces/Nuopel/wine-rating-api"
