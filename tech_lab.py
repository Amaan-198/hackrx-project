from pathlib import Path
from typing import Dict, List

TECH_LAB_DIR = Path(__file__).resolve().parent / "tech"

TECH_DEMOS = [
    {
        "id": "case-study",
        "title": "HTML + CSS Case Study",
        "description": "A clean static page that frames the repo and links to the local demos.",
        "path": "index.htm",
        "technology": "HTML + CSS",
    },
    {
        "id": "calculator",
        "title": "Scientific Calculator",
        "description": "A small JavaScript calculator with scientific functions and keyboard support.",
        "path": "scientific-calculator.html",
        "technology": "JavaScript",
    },
    {
        "id": "angularjs-auth",
        "title": "AngularJS Registration/Login",
        "description": "A minimal AngularJS form flow that runs locally with a seeded demo user.",
        "path": "angularjs-auth.html",
        "technology": "AngularJS",
    },
    {
        "id": "xml-workflow",
        "title": "XML + DTD + XSL Workflow",
        "description": "A browser-rendered XML document validated by DTD and transformed with XSL.",
        "path": "xml/claim-workflow.xml",
        "technology": "XML / DTD / XSL",
    },
]


def build_tech_demo_manifest(base_path: str = "/tech") -> List[Dict[str, str]]:
    normalized_base = base_path.rstrip("/") or "/tech"
    items = []
    for demo in TECH_DEMOS:
        url = f"{normalized_base}/{demo['path']}"
        if demo["path"] == "index.htm":
            url = f"{normalized_base}/"
        items.append({**demo, "url": url})
    return items


def resolve_tech_file(relative_path: str | None = None) -> Path:
    normalized_path = (relative_path or "index.htm").strip().lstrip("/").replace("\\", "/")
    candidate = (TECH_LAB_DIR / normalized_path).resolve()
    base_dir = TECH_LAB_DIR.resolve()

    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise FileNotFoundError("Technology asset not found.") from exc

    if not candidate.is_file():
        raise FileNotFoundError("Technology asset not found.")

    return candidate
