"""Streamlit dashboard. Talks to the API only (set API_URL).   streamlit run app/ui.py"""
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Plant Disease MLOps", page_icon="🌿", layout="wide")


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 120), **kw)
        return r
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API}: {e}")
        st.stop()


page = st.sidebar.radio("Page", ["Dashboard & up-time", "Predict", "Data insights", "Upload & retrain", "Model evaluation"])
st.sidebar.caption(f"API: {API}")

# ------------------------------------------------------------------ 1. up-time
if page == "Dashboard & up-time":
    st.title("🌿 Plant Disease Classifier - monitoring")
    h = api("GET", "/health").json()
    up = h["uptime_seconds"]
    c = st.columns(4)
    c[0].metric("Status", h["status"].upper())
    c[1].metric("Up-time", f"{int(up // 3600)}h {int(up % 3600 // 60)}m {int(up % 60)}s")
    c[2].metric("Requests served", h["requests_served"])
    c[3].metric("Avg latency", f"{h['avg_latency_ms']} ms")
    c = st.columns(4)
    c[0].metric("Predictions", h["predictions_served"])
    c[1].metric("Errors", h["errors"])
    c[2].metric("Model loaded", "yes" if h["model_loaded"] else "not yet")
    c[3].metric("Model version", h["model_version"] or "-")
    st.caption(f"Answered by container: {h['container']} (refresh to see load balancing across containers)")
    if st.button("Refresh"):
        st.rerun()

# ------------------------------------------------------------------ 2. predict
elif page == "Predict":
    st.title("Predict one leaf image")
    f = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    if f:
        left, right = st.columns(2)
        left.image(f, caption=f.name, use_container_width=True)
        r = api("POST", "/predict", files={"file": (f.name, f.getvalue(), f.type)})
        if r.ok:
            res = r.json()
            right.success(f"**{res['label']}**  ({res['confidence']:.1%})")
            right.bar_chart(pd.DataFrame(res["top"]).set_index("label")["confidence"])
        else:
            right.error(r.json().get("detail", r.text))

# ------------------------------------------------------------------ 3. data insights
elif page == "Data insights":
    st.title("Data insights")
    stats = api("GET", "/data/stats").json()
    if not stats["train"]:
        st.warning("No training data found. Run `python -m src.data_acquisition` first.")
        st.stop()
    dist = pd.Series(stats["train"]).sort_values()
    st.subheader("1. Class distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    dist.plot.barh(ax=ax, color="#3b8b5a")
    ax.set_xlabel("training images")
    st.pyplot(fig)
    st.markdown(f"**Story:** the largest class has **{dist.max()}** images, the smallest only **{dist.min()}** "
                f"({dist.max() / max(dist.min(), 1):.0f}x). The data is imbalanced, so accuracy alone can hide "
                "weak classes. That is why training uses class weights and we report macro precision, recall and F1.")

    feats = pd.DataFrame(api("GET", "/data/features", timeout=600).json())
    order = feats.groupby("class")["mean_green"].median().sort_values().index

    def box(col, title):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.boxplot([feats[feats["class"] == c][col] for c in order])
        ax.set_xticklabels(order, rotation=75, ha="right")
        ax.set_title(title)
        st.pyplot(fig)

    st.subheader("2. Average green intensity per class")
    box("mean_green", "Mean green channel (0-255)")
    st.markdown("**Story:** healthy leaves usually keep a stronger green signal, while blight, mould and virus "
                "classes shift towards yellow/brown and lose green. Overlapping boxes warn us which classes the "
                "model will find hard to separate by colour alone.")
    st.subheader("3. Share of non-green pixels per class")
    box("non_green_ratio", "Fraction of pixels where green is not dominant")
    st.markdown("**Story:** lesions, dead tissue and background raise this ratio. Diseases with large damaged areas "
                "(late blight, early blight) separate best; mosaic virus and bacterial spot are subtler, so "
                "texture (learned by the CNN) matters more than colour.")
    st.subheader("4. Brightness per class")
    box("brightness", "Mean brightness (0-255)")
    st.caption("Wide brightness spread inside a class means photos were taken under different light, "
               "which is why the model uses contrast augmentation.")

# ------------------------------------------------------------------ 4. upload & retrain
elif page == "Upload & retrain":
    st.title("Upload new data & retrain")
    summary = api("GET", "/uploads/summary").json()
    c = st.columns(4)
    c[0].metric("Pending uploads", summary["pending"])
    c[1].metric("Processed", summary["processed"])
    c[2].metric("Rejected", summary["rejected"])
    c[3].metric("Auto-retrain threshold", summary["retrain_threshold"])

    classes = api("GET", "/classes").json()["classes"]
    t1, t2 = st.tabs(["Images of one class", "Zip with class folders"])
    with t1:
        label = st.selectbox("Class of these images", classes)
        files = st.file_uploader("Select many images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files and st.button("Upload images"):
            r = api("POST", "/upload", data={"label": label},
                    files=[("files", (f.name, f.getvalue(), f.type)) for f in files])
            st.success(r.json()) if r.ok else st.error(r.text)
    with t2:
        st.caption("Zip layout: any_folder/<class_name>/image.jpg")
        z = st.file_uploader("Zip file", type=["zip"])
        if z and st.button("Upload zip"):
            r = api("POST", "/upload_zip", files={"file": (z.name, z.getvalue(), "application/zip")})
            st.success(r.json()) if r.ok else st.error(r.text)

    st.divider()
    st.subheader("Retraining")
    if st.button("🔁 Retrain model now", type="primary"):
        r = api("POST", "/retrain")
        st.info(r.json().get("message") or r.json().get("detail"))
    status = api("GET", "/retrain/status").json()
    run = status["latest_run"]
    if run:
        st.write(f"Latest run #{run['id']} - **{run['status']}** (trigger: {run['trigger']})")
        if run["message"]:
            st.write(run["message"])
        if run["status"] == "running":
            st.caption("Training in the background...")
            time.sleep(5)
            st.rerun()
    if status["history"]:
        st.dataframe(pd.DataFrame(status["history"])[["id", "trigger", "status", "n_new_images", "old_accuracy",
                                                      "new_accuracy", "model_replaced", "message"]])

# ------------------------------------------------------------------ 5. evaluation in production
elif page == "Model evaluation":
    st.title("Evaluate the deployed model")
    n = st.slider("Test images to use", 50, 1000, 300, 50)
    if st.button("Run evaluation on the production model"):
        with st.spinner("Evaluating..."):
            r = api("POST", f"/evaluate?max_samples={n}", timeout=900)
        if r.ok:
            m = r.json()
            c = st.columns(4)
            c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
            c[1].metric("Precision (macro)", f"{m['precision_macro']:.3f}")
            c[2].metric("Recall (macro)", f"{m['recall_macro']:.3f}")
            c[3].metric("F1 (macro)", f"{m['f1_macro']:.3f}")
            st.caption(f"{m['samples']} images, {m['ms_per_image']} ms/image, model version {m['model_version']}")
        else:
            st.error(r.json().get("detail", r.text))
