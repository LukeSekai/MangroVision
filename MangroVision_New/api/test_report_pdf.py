"""Regression checks for readable, complete PDFs and the protected download route."""

from io import BytesIO
from pathlib import Path
import re
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.report_pdf import ReportPdfRequest, render_restoration_report_pdf
from api.routes import dashboard, export


def payload(**updates):
    result = {
        "report_type": "planting", "title": "Planting accomplishment report",
        "description": "Seedlings planted within the reporting period.",
        "period_from": "2026-01-01", "period_to": "2026-10-04", "site": "Nasugban",
        "generated_at": "Oct 4, 2026, 12:00 PM", "prepared_by": "LGU Staff",
        "stats": [{"label": "Seedlings planted", "value": "N/A", "hint": "Includes replacement seedlings"}],
        "sections": [{"title": "Monthly planting", "columns": [
            {"key": "date", "label": "Month", "type": "month"},
            {"key": "planted", "label": "Seedlings planted", "type": "count"},
        ], "rows": [["Sep 2026", "12"]]}],
        "notes": ["Assigning a point alone does not count as planting a seedling."],
        "remarks": "Inspect the seedlings next week.",
    }
    result.update(updates)
    return result


def read_pdf(body):
    return PdfReader(BytesIO(render_restoration_report_pdf(ReportPdfRequest.model_validate(body))))


def test_metadata_remarks_na_and_embedded_font():
    reader = read_pdf(payload(site="Nasugban & Peña <b>Site</b>", remarks="<img src='https://invalid'>\nInspect erosion."))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    for value in ["MangroVision", "N/A", "Nasugban & Peña <b>Site</b>", "Inspect erosion.", "Page 1 of 1"]:
        assert value in text
    assert "<img src='https://invalid'>" in text
    assert reader.metadata.title == "Planting accomplishment report"
    assert reader.metadata.author == "LGU Staff"
    fonts = list(reader.pages[0]["/Resources"]["/Font"].get_object().values())
    assert any("/FontFile2" in font.get_object().get("/FontDescriptor", {}) for font in fonts)
    names = [str(font.get_object()["/BaseFont"]) for font in fonts]
    assert any("MangroVisionInter-Regular" in name for name in names)
    assert any("MangroVisionInter-Bold" in name for name in names)


def test_table_pagination_repeats_section_and_column_headers_without_losing_rows():
    body = payload()
    body["sections"][0]["rows"] = [[f"Record {index:03}", str(index)] for index in range(100)]
    body["notes"] = [f"Explanation {index:03}: " + "Recorded counts describe the selected period and should be read with their stated scope. " * 4 for index in range(16)]
    reader = read_pdf(body)
    assert len(reader.pages) > 2
    text = "\n".join(page.extract_text() for page in reader.pages)
    for index in range(100):
        assert text.count(f"Record {index:03}") == 1
    for index in range(16):
        assert text.count(f"Explanation {index:03}") == 1
    for number, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        assert "MangroVision" in page_text
        assert f"Page {number} of {len(reader.pages)}" in page_text
        if "Record " in page_text:
            assert "Monthly planting" in page_text
            assert "Month" in page_text
            assert "Seedlings planted" in page_text
        if "Explanation " in page_text:
            assert "Reading these figures" in page_text


def test_wide_table_is_landscape_and_long_rows_can_span_pages():
    body = payload(report_type="monitoring")
    body["sections"] = [{"title": "Organization visits", "columns": [
        {"key": "organization_name", "label": "Organization"},
        {"key": "actions_taken", "label": "Actions taken"},
        *[{"key": f"count{i}", "label": f"Count {i}", "type": "count"} for i in range(5)],
    ], "rows": [["Coastal Community", "Inspect erosion and seedlings. " * 180 + "End of actions.", *["10"] * 5]]}]
    reader = read_pdf(body)
    assert len(reader.pages) > 1
    assert reader.pages[0].mediabox.width > reader.pages[0].mediabox.height
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "End of actions." in re.sub(r"\s+", " ", text)
    assert "Inspect the seedlings next week." in text


def test_empty_table_and_multiline_long_remarks_render_completely():
    body = payload(remarks="\n".join(f"Remark {index:03}: follow up the recorded activities." for index in range(85)))
    body["sections"][0]["rows"] = []
    reader = read_pdf(body)
    assert reader.pages[0].mediabox.height > reader.pages[0].mediabox.width
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "No records for this selection." in text
    for index in range(85):
        assert text.count(f"Remark {index:03}") == 1


def test_invalid_dimensions_dates_and_unrecognized_fields_rejected():
    body = payload()
    body["sections"][0]["rows"] = [["Too few cells"]]
    with pytest.raises(ValidationError, match="one cell"):
        ReportPdfRequest.model_validate(body)
    with pytest.raises(ValidationError, match="precede"):
        ReportPdfRequest.model_validate(payload(period_to="2025-12-01"))
    with pytest.raises(ValidationError, match="Extra inputs"):
        ReportPdfRequest.model_validate(payload(image_url="https://invalid"))


def test_download_requires_lgu_session_and_returns_pdf_attachment(monkeypatch):
    app = FastAPI()
    app.include_router(export.router, prefix="/api/export")
    client = TestClient(app)
    monkeypatch.setattr(dashboard, "get_user_by_session_token", lambda _token: None)
    assert client.post("/api/export/report/pdf", json=payload()).status_code == 401
    monkeypatch.setattr(dashboard, "get_user_by_session_token", lambda _token: {"role": "planter"})
    assert client.post("/api/export/report/pdf", json=payload()).status_code == 403
    monkeypatch.setattr(dashboard, "get_user_by_session_token", lambda _token: {"role": "lgu"})
    response = client.post("/api/export/report/pdf", json=payload())
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-")
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"] == 'attachment; filename="mangrovision-planting-2026-01-01-to-2026-10-04.pdf"'
    assert len(PdfReader(BytesIO(response.content)).pages) == 1
    assert client.post("/api/export/report/pdf", json=payload(period_to="2025-12-01")).status_code == 422
