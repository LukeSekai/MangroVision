"""Render the displayed LGU report snapshot as a paginated PDF.

The browser supplies already formatted cells so the preview, its N/A values and
the downloaded document agree. Rendering is stateless and does not query or
modify planting records. All supplied text is escaped before PDF layout.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
from xml.sax.saxutils import escape

from pydantic import BaseModel, ConfigDict, Field, model_validator
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import BaseDocTemplate, Frame, LongTable, PageTemplate, Paragraph, Spacer, Table, TableStyle


Text = Annotated[str, Field(max_length=20000)]
Label = Annotated[str, Field(min_length=1, max_length=500)]
CellType = Literal["text", "date", "month", "count", "percent", "decimal", "coordinate"]


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportColumn(ReportModel):
    key: Label
    label: Label
    type: CellType = "text"


class ReportStat(ReportModel):
    label: Label
    value: Label
    hint: Text = ""


class ReportSection(ReportModel):
    title: Label
    columns: list[ReportColumn] = Field(min_length=1, max_length=20)
    rows: list[list[Text]] = Field(max_length=20000)

    @model_validator(mode="after")
    def matching_cells(self):
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("Every row must have one cell for each column.")
        return self


class ReportPdfRequest(ReportModel):
    report_type: Literal["planting", "monitoring", "mortality", "organizations"]
    title: Label
    description: Text
    period_from: date
    period_to: date
    site: Label
    generated_at: Label
    prepared_by: Label
    remarks: str = Field(default="", max_length=5000)
    stats: list[ReportStat] = Field(min_length=1, max_length=8)
    sections: list[ReportSection] = Field(min_length=1, max_length=30)
    notes: list[Text] = Field(max_length=30)

    @model_validator(mode="after")
    def bounded_report(self):
        if self.period_from > self.period_to:
            raise ValueError("The reporting period start must precede its end.")
        if sum(len(section.rows) for section in self.sections) > 50000:
            raise ValueError("The report exceeds 50,000 rows. Select a shorter period.")
        if sum(len(cell) for section in self.sections for row in section.rows for cell in row) > 5000000:
            raise ValueError("The report is too large. Select a shorter period.")
        return self

    @property
    def filename(self) -> str:
        return f"mangrovision-{self.report_type}-{self.period_from}-to-{self.period_to}.pdf"


ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = Path(__file__).resolve().parents[1] / "client" / "public" / "logo-icon.png"
FONT_LOCK = Lock()
GREEN = colors.HexColor("#14532d")
INK = colors.HexColor("#172a20")
MUTED = colors.HexColor("#52675a")
BORDER = colors.HexColor("#dbe5df")
TINT = colors.HexColor("#f1f6f3")


def _register_fonts():
    with FONT_LOCK:
        if "MVInter" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("MVInter", str(ASSETS / "fonts" / "Inter-Regular.ttf")))
            pdfmetrics.registerFont(TTFont("MVInterBold", str(ASSETS / "fonts" / "Inter-Bold.ttf")))


def _plain(text: str) -> str:
    return text.translate(str.maketrans({"\u2013": "-", "\u2014": "-", "\u2011": "-"}))


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_plain(text)).replace("\n", "<br/>"), style)


def _date_label(value: date) -> str:
    return f"{value.strftime('%b')} {value.day}, {value.year}"


def _column_weight(column: ReportColumn) -> float:
    if column.type == "coordinate":
        return 1.7
    if column.type in {"date", "month"}:
        return 1.8
    if column.type != "text":
        return 1.1
    if column.key in {"actions_taken", "notes", "remarks"}:
        return 4.5
    if any(word in column.key for word in ("name", "title", "species", "status", "label", "inspection")):
        return 2.8
    return 2.1


class _NumberedCanvas(Canvas):
    """Defer page output until the total count is known."""

    def __init__(self, *args, footer_label: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._page_states = []
        self._footer_label = footer_label

    def showPage(self):
        self._page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._page_states)
        for state in self._page_states:
            self.__dict__.update(state)
            width, _ = self._pagesize
            self.setStrokeColor(BORDER)
            self.line(12 * mm, 12 * mm, width - 12 * mm, 12 * mm)
            self.setFillColor(MUTED)
            self.setFont("MVInter", 8)
            label = _plain(self._footer_label)
            while label and pdfmetrics.stringWidth(label, "MVInter", 8) > width - 52 * mm:
                label = label[:-1]
            if label != _plain(self._footer_label):
                label = label[:-3] + "..."
            self.drawString(12 * mm, 7.5 * mm, label)
            self.drawRightString(width - 12 * mm, 7.5 * mm, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def render_restoration_report_pdf(report: ReportPdfRequest) -> bytes:
    _register_fonts()
    wide = max(len(section.columns) for section in report.sections) > 6
    page_size = landscape(A4) if wide else A4
    output = BytesIO()
    doc = BaseDocTemplate(
        output, pagesize=page_size, leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=25 * mm, bottomMargin=18 * mm,
        title=report.title, author=report.prepared_by, creator="MangroVision",
    )
    base = ParagraphStyle("body", fontName="MVInter", fontSize=9.5, leading=14, textColor=INK,
                          splitLongWords=True, spaceAfter=6)
    styles = {
        "body": base,
        "title": ParagraphStyle("title", parent=base, fontName="MVInterBold", fontSize=19,
                                leading=25, textColor=GREEN, spaceAfter=5, keepWithNext=True),
        "section": ParagraphStyle("section", parent=base, fontName="MVInterBold", fontSize=11,
                                  leading=15, textColor=GREEN, spaceAfter=8, keepWithNext=True),
        "label": ParagraphStyle("label", parent=base, fontName="MVInterBold", fontSize=7.5,
                                leading=11, textColor=MUTED, spaceAfter=3),
        "value": ParagraphStyle("value", parent=base, fontName="MVInterBold", fontSize=20,
                                leading=27, textColor=GREEN, spaceAfter=4),
        "hint": ParagraphStyle("hint", parent=base, fontSize=8, leading=12, textColor=MUTED, spaceAfter=0),
        "cell": ParagraphStyle("cell", parent=base, fontSize=9, leading=12.5, spaceAfter=0),
        "number": ParagraphStyle("number", parent=base, fontSize=9, leading=12.5, spaceAfter=0,
                                 alignment=TA_RIGHT),
        "heading": ParagraphStyle("heading", parent=base, fontName="MVInterBold", fontSize=8,
                                  leading=11, textColor=GREEN, spaceAfter=0),
    }
    story = [_paragraph(report.title, styles["title"]), _paragraph(report.description, base), Spacer(1, 8)]
    metadata = [
        ("REPORTING PERIOD", f"{_date_label(report.period_from)} - {_date_label(report.period_to)}"),
        ("PROJECT SITE", report.site), ("GENERATED - ASIA/MANILA", report.generated_at),
        ("PREPARED BY", report.prepared_by),
    ]
    meta_cells = [[_paragraph(label, styles["label"]), _paragraph(value, styles["cell"])] for label, value in metadata]
    meta_columns = 4 if wide else 2
    meta_rows = [meta_cells[i:i + meta_columns] for i in range(0, len(meta_cells), meta_columns)]
    meta = Table(meta_rows, colWidths=[doc.width / meta_columns] * meta_columns, hAlign="LEFT")
    meta.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10), ("LINEABOVE", (0, 0), (-1, 0), .5, BORDER),
        ("LINEBELOW", (0, -1), (-1, -1), .5, BORDER),
    ]))
    story.extend([meta, Spacer(1, 14)])
    cards = [[_paragraph(stat.label, styles["label"]), _paragraph(stat.value, styles["value"]),
              _paragraph(stat.hint, styles["hint"])] for stat in report.stats]
    for index in range(0, len(cards), 4):
        row = cards[index:index + 4]
        stats = Table([row], colWidths=[doc.width / len(row)] * len(row), hAlign="LEFT")
        stats.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), TINT), ("BOX", (0, 0), (-1, -1), .5, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), .5, BORDER), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.extend([stats, Spacer(1, 8)])

    for section in report.sections:
        weights = [_column_weight(column) for column in section.columns]
        widths = [doc.width * weight / sum(weights) for weight in weights]
        rows = [[_paragraph(section.title, styles["section"])] + [""] * (len(widths) - 1),
                [_paragraph(column.label, styles["heading"]) for column in section.columns]]
        for row in section.rows:
            rows.append([_paragraph(cell, styles["cell"] if column.type in {"text", "date", "month"} else styles["number"])
                         for column, cell in zip(section.columns, row)])
        if not section.rows:
            rows.append([_paragraph("No records for this selection.", styles["hint"])] + [""] * (len(widths) - 1))
        table = LongTable(rows, colWidths=widths, repeatRows=2, hAlign="LEFT", splitByRow=1, splitInRow=1)
        commands = [
            ("SPAN", (0, 0), (-1, 0)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
            ("LEFTPADDING", (0, 0), (-1, 0), 0), ("RIGHTPADDING", (0, 0), (-1, 0), 0),
            ("TOPPADDING", (0, 0), (-1, 0), 12), ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
            ("BACKGROUND", (0, 1), (-1, 1), TINT),
            ("GRID", (0, 1), (-1, -1), .4, BORDER),
            ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#fafcfb")]),
        ]
        if not section.rows:
            commands.append(("SPAN", (0, 2), (-1, 2)))
        for index, column in enumerate(section.columns):
            if column.type not in {"text", "date", "month"}:
                commands.append(("ALIGN", (index, 1), (index, 1), "RIGHT"))
                rows[1][index].style = ParagraphStyle(f"numeric-heading-{index}", parent=styles["heading"], alignment=TA_RIGHT)
        table.setStyle(TableStyle(commands))
        story.append(table)

    if report.remarks.strip():
        story.extend([Spacer(1, 16), _paragraph("Report remarks", styles["section"]),
                      _paragraph(report.remarks.strip(), base)])
    note_rows = [[_paragraph("Reading these figures", styles["section"])]]
    note_rows.extend([_paragraph(f"- {note}", styles["hint"])] for note in report.notes)
    # Explanations can grow as report terminology is clarified. Repeat their
    # heading when they continue on another page, just like the data tables.
    reading_notes = LongTable(note_rows, colWidths=[doc.width], repeatRows=1,
                              splitByRow=1, splitInRow=1, hAlign="LEFT")
    reading_notes.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([Spacer(1, 16), reading_notes])

    def draw_header(canvas, _doc):
        width, height = page_size
        canvas.saveState()
        if LOGO.is_file():
            canvas.drawImage(str(LOGO), 12 * mm, height - 19 * mm, width=11 * mm, height=11 * mm,
                             preserveAspectRatio=True, mask="auto")
        canvas.setFillColor(GREEN)
        canvas.setFont("MVInterBold", 12)
        canvas.drawString(26 * mm, height - 12 * mm, "MangroVision")
        canvas.setFillColor(MUTED)
        canvas.setFont("MVInter", 7.5)
        canvas.drawString(26 * mm, height - 17 * mm, "Restoration program records")
        canvas.drawRightString(width - 12 * mm, height - 13 * mm, "LGU RESTORATION REPORT")
        canvas.setStrokeColor(GREEN)
        canvas.line(12 * mm, height - 21 * mm, width - 12 * mm, height - 21 * mm)
        canvas.restoreState()

    # Zero frame padding aligns the tables with the brand header and page rule.
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=draw_header))
    doc.build(story,
              canvasmaker=lambda *args, **kwargs: _NumberedCanvas(*args, footer_label=f"MangroVision | {report.site}", **kwargs))
    return output.getvalue()
