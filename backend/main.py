"""
SENTINEL-H2 Simulation Dashboard — FastAPI Backend

No Redis, no Celery. Uses ThreadPoolExecutor for background jobs.
Serves the frontend as static files.
"""

import os
import sys
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Ensure the backend directory is on the path so pillar imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from job_manager import manager
from pillars import pillar1_electrolyte
from pillars import pillar2_hydrate
from pillars import pillar3_gdl
from pillars import pillar4_ae
from pillars import pillar5_ep
from pillars import pillar6_hdc
import orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SENTINEL-H2 Simulation Console")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Map pillar number to simulation module
PILLAR_MODULES = {
    1: pillar1_electrolyte,
    2: pillar2_hydrate,
    3: pillar3_gdl,
    4: pillar4_ae,
    5: pillar5_ep,
    6: pillar6_hdc,
}

PILLAR_NAMES = {
    1: "Solid-Acid Electrolyte",
    2: "Clathrate-Hydrate Storage",
    3: "Gas Diffusion Layer (LBM)",
    4: "Acoustic-Emission Inversion",
    5: "Exceptional-Point Sensor",
    6: "HDC Fusion Engine",
}


class SimRequest(BaseModel):
    params: Dict[str, Any] = {}


# --- Pillar endpoints ---

@app.post("/api/pillar/{pillar_id}/run")
def run_pillar(pillar_id: int, req: SimRequest):
    if pillar_id not in PILLAR_MODULES:
        raise HTTPException(status_code=404, detail=f"Pillar {pillar_id} not found")

    module = PILLAR_MODULES[pillar_id]
    job_id = manager.submit(
        task_fn=module.run_simulation,
        params=req.params,
        pillar=PILLAR_NAMES[pillar_id],
    )
    logger.info("Submitted pillar %d job: %s", pillar_id, job_id)
    return {"job_id": job_id, "pillar": pillar_id}


# --- System endpoint ---

@app.post("/api/system/run")
def run_system(req: SimRequest):
    job_id = manager.submit(
        task_fn=orchestrator.run_simulation,
        params=req.params,
        pillar="System Integration",
    )
    logger.info("Submitted system job: %s", job_id)
    return {"job_id": job_id, "pillar": "system"}


import math
import json

def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None so JSON serialization works."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str):
    status = manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return sanitize_for_json(status)


# --- Serve frontend ---

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

# Mount static directories
if os.path.isdir(os.path.join(FRONTEND_DIR, "css")):
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
if os.path.isdir(os.path.join(FRONTEND_DIR, "js")):
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
if os.path.isdir(os.path.join(FRONTEND_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")


@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
