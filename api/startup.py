"""
api/startup.py
--------------
Downloads ML model and RAG docs from GCS when Cloud Run
container starts. Called once on FastAPI startup.

Locally: skips download (files already exist)
On GCP:  downloads from gs://caris-cpchem-assets/
"""

import os
import sys

GCP_PROJECT  = os.getenv("GCP_PROJECT_ID", "")
GCS_BUCKET   = os.getenv("GCS_BUCKET", "caris-cpchem-assets")
MODEL_DIR    = "ml/saved_models"
CHROMA_DIR   = "chroma_db"
RAG_DOCS_DIR = "rag/docs"


def download_from_gcs():
    """Download model + RAG assets from GCS on Cloud Run startup."""
    if not GCP_PROJECT:
        print("[Startup] Local mode — skipping GCS download")
        return

    try:
        from google.cloud import storage
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(GCS_BUCKET)

        dirs_to_download = [
            (f"ml/saved_models/", MODEL_DIR),
            (f"chroma_db/",       CHROMA_DIR),
            (f"rag/docs/",        RAG_DOCS_DIR),
        ]

        for gcs_prefix, local_dir in dirs_to_download:
            os.makedirs(local_dir, exist_ok=True)
            blobs = list(bucket.list_blobs(prefix=gcs_prefix))

            if not blobs:
                print(f"[Startup] No files found at gs://{GCS_BUCKET}/{gcs_prefix}")
                continue

            for blob in blobs:
                local_path = blob.name
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                if not os.path.exists(local_path):
                    blob.download_to_filename(local_path)
                    print(f"[Startup] Downloaded {blob.name}")
                else:
                    print(f"[Startup] Already exists: {local_path}")

        print("[Startup] GCS download complete")

    except Exception as e:
        print(f"[Startup] GCS download error: {e}")
        print("[Startup] Continuing without GCS assets...")