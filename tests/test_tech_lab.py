import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import backend_api
from tech_lab import TECH_LAB_DIR, build_tech_demo_manifest, resolve_tech_file

client = TestClient(backend_api.app)


def test_demo_manifest_has_expected_entries():
    demos = build_tech_demo_manifest()
    titles = {demo["title"] for demo in demos}
    assert titles == {
        "HTML + CSS Case Study",
        "Scientific Calculator",
        "AngularJS Registration/Login",
        "XML + DTD + XSL Workflow",
    }


def test_resolve_tech_file_blocks_path_traversal():
    with pytest.raises(FileNotFoundError):
        resolve_tech_file("../app.py")


def test_api_returns_tech_demo_manifest():
    response = client.get("/api/tech-demos")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 4
    assert payload["items"][0]["url"] == "/tech/"


def test_tech_index_redirects_and_serves_html():
    redirected = client.get("/tech", follow_redirects=False)
    assert redirected.status_code in {307, 308}

    response = client.get("/tech/")
    assert response.status_code == 200
    assert "Technology Lab" in response.text
    assert "Website evaluation in one screen" in response.text


def test_calculator_page_serves_expected_markup():
    response = client.get("/tech/scientific-calculator.html")
    assert response.status_code == 200
    assert 'data-calculator-display' in response.text
    assert "evaluateScientificExpression" in response.text


def test_angular_page_serves_expected_markup():
    response = client.get("/tech/angularjs-auth.html")
    assert response.status_code == 200
    assert "ng-app=\"techLabAuth\"" in response.text
    assert "demo@decision.local" in response.text


def test_xml_workflow_references_dtd_and_xsl():
    response = client.get("/tech/xml/claim-workflow.xml")
    assert response.status_code == 200
    assert "<?xml-stylesheet type=\"text/xsl\" href=\"claim-workflow.xsl\"?>" in response.text
    assert "<!DOCTYPE claimWorkflow SYSTEM \"claim-workflow.dtd\">" in response.text


def test_xsl_file_is_well_formed_xml():
    xsl_path = resolve_tech_file("xml/claim-workflow.xsl")
    tree = ElementTree.parse(xsl_path)
    root = tree.getroot()
    assert root.tag.endswith("stylesheet")


def test_missing_tech_asset_returns_404():
    response = client.get("/tech/missing-file.html")
    assert response.status_code == 404
    assert response.json()["detail"] == "Technology asset not found."


def test_node_tech_lab_suite_passes():
    result = subprocess.run(
        ["node", "--test", str(TECH_LAB_DIR / "tests" / "tech_lab.test.cjs")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_all_declared_demo_files_exist():
    for demo in build_tech_demo_manifest():
        if demo["url"] == "/tech/":
            file_path = TECH_LAB_DIR / "index.htm"
        else:
            file_path = TECH_LAB_DIR / demo["path"]
        assert file_path.is_file(), f"Missing demo file: {file_path}"


def test_shared_assets_exist():
    assert (TECH_LAB_DIR / "assets" / "tech-lab.css").is_file()
    assert (TECH_LAB_DIR / "assets" / "tech-lab.js").is_file()
