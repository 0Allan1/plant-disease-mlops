# 🌿 Plant Disease Classification - end-to-end MLOps pipeline

An image-classification system that recognises **15 leaf conditions** (pepper, potato and tomato, healthy or diseased) from a photo.
It covers the whole ML cycle: data acquisition → preprocessing → model creation → testing → **retraining on newly uploaded data** →
API → web UI → Docker → cloud deployment → load testing with Locust.

| | |
|---|---|
| 🎥 **Video demo (YouTube)** | `PASTE_YOUTUBE_LINK_HERE` |
| 🌍 **Live UI** |  |
| 🔌 **Live API (Swagger docs)** | `PASTE_PUBLIC_API_URL_HERE` (example: `http://<VM_IP>:8000/docs`) |
| 💻 **GitHub repo** | `PASTE_GITHUB_REPO_URL_HERE` |



## 1. Project description

* **Problem:** farmers need a fast way to spot crop disease early. A leaf photo goes in, the disease class comes out.
* **Data:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (image data): 20,638 leaf images in 15 classes, split 80/20 into `data/train` (16,516) and `data/test` (4,122). The images are not committed; `python -m src.data_acquisition` recreates them.
* **Model:** MobileNetV2 pre-trained on ImageNet with a new classification head, saved as `models/plant_disease_model.h5`.
* **Optimisation:** transfer learning, two-phase training (frozen head, then fine-tuning the top 80 layers at lr 1e-4), dropout, L2 regularisation, data augmentation, class weights for the imbalanced classes, early stopping and learning-rate reduction.
* **Retraining:** users upload new labelled images (many images, or a zip with one folder per class). Files are recorded in a SQLite database, validated and standardised, and the **saved model is fine-tuned** (it is the pre-trained model). The new model replaces the old one only if its test accuracy is not more than `ACCEPT_TOLERANCE` (default 0.5 points) worse. Retraining starts with the **Retrain button** or automatically when `RETRAIN_THRESHOLD` (default 50) new images are waiting.

## 2. Model results (test set, 4,122 images the model never saw)

| Loss | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---:|---:|---:|---:|---:|
| 0.162 | 0.962 | 0.965 | 0.967 | 0.965 |

* Validation accuracy at the end of training (about 0.96) matches test accuracy, so there is no sign of overfitting.
* Weakest class: **Tomato Target Spot** (precision 0.82, F1 0.88), because other tomato spot diseases (Septoria, spider mites, bacterial spot) are often predicted as Target Spot. Tomato early blight has recall 0.90. All pepper and potato classes have F1 of 0.988 or higher.
* Full preprocessing, plots, learning curves, per-class report and confusion matrix are in `notebook/plant_disease_mlops.ipynb`.
* In production (cloud VM, 240-image test subset) the **Model evaluation** page gives accuracy 0.963, precision 0.967, recall 0.963, F1 0.961 at about 28 ms per image.

## 3. Repository structure
```
plant-disease-mlops/
├── README.md
├── notebook/plant_disease_mlops.ipynb   preprocessing, training, evaluation, prediction
├── src/
│   ├── preprocessing.py                 datasets, cleaning, class weights, image features
│   ├── model.py                         build / train / evaluate / save
│   ├── prediction.py                    loads the model, predicts one image
│   ├── retrain.py                       retraining pipeline + trigger
│   └── data_acquisition.py, train.py, db.py, config.py
├── app/api.py                           FastAPI service
├── app/ui.py                            Streamlit dashboard
├── data/train, data/test, data/uploads  (images are not committed)
├── models/plant_disease_model.h5        trained model + class_names.json
├── locust/                              load test + experiment script
├── scripts/smoke_test.py                quick end-to-end test on synthetic images
├── Dockerfile, Dockerfile.ui, docker-compose.yml, nginx.conf
└── results/                             flood-test results
```

## 4. Setup (local, no Docker needed)

```bash
git clone PASTE_GITHUB_REPO_URL_HERE && cd plant-disease-mlops
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-ui.txt

# 1) data (downloads PlantVillage with kagglehub and splits it)
python -m src.data_acquisition
#    already have the folders?  python -m src.data_acquisition --source /path/to/PlantVillage
#    small subset for quick tests: add  --max-per-class 80

# 2) model: run notebook/plant_disease_mlops.ipynb (use a GPU, e.g. Google Colab), then copy
#    models/plant_disease_model.h5 and models/class_names.json into models/
#    (python -m src.train trains from scratch on CPU and is very slow)

# 3) API  (Swagger UI at http://localhost:8000/docs)
uvicorn app.api:app --port 8000

# 4) UI in a second terminal  (http://localhost:8501)
#    Linux/Mac:           API_URL=http://localhost:8000 streamlit run app/ui.py
#    Windows PowerShell:  $env:API_URL='http://localhost:8000'; streamlit run app/ui.py
```
Check everything without any dataset: `python scripts/smoke_test.py`.

## 5. Run with Docker

```bash
docker compose up --build            # API behind nginx on :8000, UI on :8501
docker compose up -d --scale api=3   # run 3 API containers (nginx load-balances them)
```
`data/` and `models/` are shared volumes, so all containers use the same model and uploads. After a retrain, every container reloads the new model file automatically.

## 6. Using the app

| UI page | What it does |
|---|---|
| Dashboard & up-time | status, up-time, requests, errors, average latency, model version, container name |
| Predict | upload one leaf image → predicted disease + confidence |
| Data insights | class distribution and 3 colour features, each with an interpretation |
| Upload & retrain | bulk upload (many images or a zip), pending counts, **Retrain** button, run history |
| Model evaluation | evaluates the **deployed** model on the test set |

API endpoints: `GET /health`, `POST /predict`, `POST /upload`, `POST /upload_zip`, `POST /retrain`, `GET /retrain/status`, `GET /uploads/summary`, `POST /evaluate`, `GET /data/stats`, `GET /data/features`.

## 7. Cloud deployment (AWS EC2)

* **Machine:** AWS EC2 `m7i-flex.large` (2 vCPUs, 8 GB RAM), Ubuntu 24.04, 30 GB disk, AWS Free plan credits.
* **Ports opened:** 22 (SSH), 8000 (API), 8501 (UI). A 4 GB swap file protects against memory spikes.
* **Deployment:** Docker installed on the VM, repository cloned, then `docker compose up -d --build`. A small dataset subset (80 images per class, `python -m src.data_acquisition --max-per-class 80`) is used on the VM for the evaluation page, retraining and Locust, and `RETRAIN_EPOCHS=1` keeps retraining short.
* **Memory measured:** about 0.8 GB per idle API container and about 3.6 GB peak during retraining. A first attempt on a 1 GB machine swapped heavily (1.9 s per image), so the VM was resized, and evaluation then dropped to 28 ms per image.

## 8. Flood-request simulation (Locust)

Command: `bash locust/run_experiments.sh <users> <spawn rate> 60s`. Each run lasts 60 seconds and sends prediction requests (plus an occasional health check) to the nginx gateway, with 1, 2 and 3 API containers. Locust ran on the same 2 vCPU VM as the API.

**10 users (normal load)**

| API containers | Requests | Failures | Req/s | Avg (ms) | Median (ms) | 95% (ms) | 99% (ms) | Max (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 507 | 0 | 8.6 | 543.8 | 550 | 840 | 910 | 983.8 |
| 2 | 563 | 0 | 9.5 | 410.0 | 370 | 820 | 940 | 1078 |
| 3 | 568 | 0 | 9.6 | 414.1 | 360 | 840 | 980 | 1107.6 |

**50 users (stress test)**

| API containers | Requests | Failures | Req/s | Avg (ms) | Median (ms) | 95% (ms) | 99% (ms) | Max (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 499 | 0 | 8.4 | 4653 | 5100 | 5600 | 5700 | 5792.2 |
| 2 | 635 | 0 | 10.8 | 3534.4 | 3300 | 5600 | 5900 | 6122 |
| 3 | 623 | 0 | 10.5 | 3622.6 | 4000 | 5900 | 6100 | 7011.8 |

**Analysis:** the VM serves about 10 predictions per second. With 10 users the system stayed responsive (median 0.36 to 0.55 s). With 50 users throughput did not increase and latency rose to several seconds because requests queued. A second container improved both cases (median -33% at 10 users, throughput +29% at 50 users), but a third gave no meaningful gain, because the VM has only 2 vCPUs and TensorFlow already uses both, so extra containers compete for the same CPU. More capacity needs more CPU cores or a lighter model, not only more containers. No request failed in any run. Each setting was run once, so small differences (for example 2 vs 3 containers) are within normal run-to-run noise.

## 9. Retraining demonstration (cloud VM)

| Run | Data uploaded | Result |
|---|---|---|
| 1 | 25 images of Tomato early blight (multi-image upload) | 25 used, 0 rejected, accuracy 0.9667 → 0.9583, new model **rejected**, old model kept |
| 2 | 45 images from 3 classes (zip upload) | 45 used, 0 rejected, accuracy 0.9667 → 0.9667, new model **accepted** and saved (the old one is archived in `models/archive/`) |

The rejected run shows the safety check: a worse model never replaces the production model. The test set on the VM is small (about 240 images, so one image is about 0.4 points), and the VM trains on only a small subset of the original data, so these accuracy differences are noise-level and are not claims of improvement.

## 10. Limitations

* Retraining supports only the 15 existing classes. A new disease would need a new output layer.
* Up-time and request counters are kept per container and reset on restart.
* SQLite is enough for this project; a larger system would use PostgreSQL.
* The UI has no login. PlantVillage photos have clean backgrounds, so field photos may give lower accuracy.
