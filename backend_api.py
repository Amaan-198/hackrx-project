from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from decision_service import DecisionServiceError, decision_service
from tech_lab import build_tech_demo_manifest, resolve_tech_file

app = FastAPI(
    title="Decision Co-Pilot API",
    version="1.0.0",
    description="REST wrapper around the existing claim decision engine.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/tech-demos")
def list_tech_demos():
    return {"items": build_tech_demo_manifest()}


@app.post("/api/decision")
async def evaluate_claim(
    claimQuery: str = Form(...),
    policyPdf: UploadFile = File(...),
):
    filename = policyPdf.filename or "policy.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF policy document.")

    file_bytes = await policyPdf.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded policy PDF is empty.")

    try:
        return decision_service.evaluate_claim(file_bytes, filename, claimQuery)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DecisionServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/tech", include_in_schema=False)
def technology_index_redirect():
    return RedirectResponse(url="/tech/")


@app.get("/tech/", include_in_schema=False)
def technology_index():
    return FileResponse(resolve_tech_file("index.htm"))


@app.get("/tech/{resource_path:path}", include_in_schema=False)
def technology_asset(resource_path: str):
    try:
        return FileResponse(resolve_tech_file(resource_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Technology asset not found.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend_api:app", host="0.0.0.0", port=8502, reload=False)
