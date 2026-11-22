from contextlib import asynccontextmanager
import os
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session

from app.db.database import get_session, init_db
from app.schemas.clinical import (
    HealthResponse,
    HematologyInput,
    PredictionRecord,
    PredictionResponse,
)
from app.services.predictor import HematologyPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes PostgreSQL tables and loads the .joblib model once at startup."""
    init_db()
    model_path = os.getenv("MODEL_PATH", "models/multilabel_model.joblib")
    app.state.predictor = HematologyPredictor(model_path=model_path)
    yield


app = FastAPI(
    title="Hematology Multi-Label Classification API",
    description="FastAPI service for multi-label hematology predictions with PostgreSQL logging.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse, tags=["System"])
def read_root(request: Request):
    predictor: HematologyPredictor | None = getattr(request.app.state, "predictor", None)
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "model_path": predictor.model_path if predictor is not None else "",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    predictor: HematologyPredictor | None = getattr(request.app.state, "predictor", None)
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "model_path": predictor.model_path if predictor is not None else "",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_hematology(
    payload: HematologyInput,
    request: Request,
    session: Session = Depends(get_session),
):
    predictor: HematologyPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        result = await run_in_threadpool(predictor.predict, payload)
        preds = result["predictions"]

        record = PredictionRecord(
            gender=payload.GENDER,
            gender_label=payload.GENDER_LABEL,
            wbc=payload.WBC,
            ne=payload.NE,
            ly=payload.LY,
            mo=payload.MO,
            eo=payload.EO,
            rbc=payload.RBC,
            hgb=payload.HGB,
            hct=payload.HCT,
            mcv=payload.MCV,
            mch=payload.MCH,
            mchc=payload.MCHC,
            rdw=payload.RDW,
            plt=payload.PLT,
            ferritin=payload.FERRITIN,
            mentzer_index=payload.MENTZER_INDEX,
            leukemia_suspect=preds["LEUKEMIA_SUSPECT"],
            viral_inf=preds["VIRAL_INF"],
            bacterial_inf=preds["BACTERIAL_INF"],
            monocytosis=preds["MONOCYTOSIS"],
            eosinophilia=preds["EOSINOPHILIA"],
            chronic_disease_anemia=preds["CHRONIC_DISEASE_ANEMIA"],
            thalassemia_trait=preds["THALASSEMIA_TRAIT"],
            iron_deficiency_anemia=preds["IRON_DEFICIENCY_ANEMIA"],
            thrombocytopenia=preds["THROMBOCYTOPENIA"],
            thrombocytosis=preds["THROMBOCYTOSIS"],
        )
        session.add(record)
        session.commit()
        session.refresh(record)

        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        session.rollback()
        raise HTTPException(
            status_code=500, detail=f"Inference or database error: {exc}"
        ) from exc