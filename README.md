# ML Clustering Project

A Streamlit-based clustering analytics app with KMeans++ tuning, PCA visualization, and a polished dashboard experience.

## Features

- Interactive Streamlit dashboard in `frontend/app.py`
- Data loading and preprocessing pipeline in `backend/ml_engine.py`
- K-fold model tuning with silhouette analysis
- Final model training, PCA projections, and cluster insights
- Seed persistence database support for deployment

## Local setup

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd ML_Clustering_Project
```

2. Create and activate a virtual environment:

```bash
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run frontend/app.py
```

## Deployment

This app is configured to deploy on Streamlit Community Cloud. The main entrypoint is `frontend/app.py`.

## App link

https://mlclusteringproject-nif9dyc2etmkpfsfdgphmd.streamlit.app