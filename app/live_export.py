
from pathlib import Path
from datetime import datetime
from html import escape
import hashlib
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    LongTable,
    KeepTogether
)


PROJECT = Path(__file__).resolve().parents[1]


def _encounter_dir(encounter_id):

    path = (
        PROJECT
        / "results"
        / "live_encounters"
        / encounter_id
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Encounter not found: {encounter_id}"
        )

    return path


def _safe(value):

    if value is None:
        return ""

    return escape(
        str(value)
    )


def _sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _transcript_chunks(text, max_chars=1800):

    words = str(text).split()

    chunks = []
    current = []

    current_length = 0

    for word in words:

        addition = len(word) + 1

        if (
            current
            and
            current_length + addition > max_chars
        ):

            chunks.append(
                " ".join(current)
            )

            current = []
            current_length = 0

        current.append(word)
        current_length += addition

    if current:
        chunks.append(
            " ".join(current)
        )

    return chunks


def _page_footer(canvas, doc):

    canvas.saveState()

    width, height = A4

    canvas.setFont(
        "Helvetica",
        8
    )

    canvas.setFillColor(
        colors.HexColor("#666666")
    )

    canvas.drawString(
        18 * mm,
        10 * mm,
        "Medical Consultation AI - Prototype Record"
    )

    canvas.drawRightString(
        width - 18 * mm,
        10 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


def export_approved_json(
    encounter_id
):

    encounter_dir = (
        _encounter_dir(
            encounter_id
        )
    )

    approved_path = (
        encounter_dir
        / "review"
        / "approved_final_record.json"
    )

    if not approved_path.exists():

        raise RuntimeError(
            "Approved final record not found. "
            "Complete review and approval first."
        )

    exports_dir = (
        encounter_dir
        / "exports"
    )

    exports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    record = json.loads(
        approved_path.read_text(
            encoding="utf-8"
        )
    )

    export_path = (
        exports_dir
        / "approved_medical_record.json"
    )

    export_path.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return export_path


def export_approved_pdf(
    encounter_id
):

    encounter_dir = (
        _encounter_dir(
            encounter_id
        )
    )

    approved_path = (
        encounter_dir
        / "review"
        / "approved_final_record.json"
    )

    if not approved_path.exists():

        raise RuntimeError(
            "Approved final record not found. "
            "Complete review and approval first."
        )

    record = json.loads(
        approved_path.read_text(
            encoding="utf-8"
        )
    )

    exports_dir = (
        encounter_dir
        / "exports"
    )

    exports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_path = (
        exports_dir
        / "approved_medical_record.pdf"
    )


    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Medical Consultation AI - Approved Record",
        author="Medical Consultation AI Prototype"
    )


    # --------------------------------------------------------
    # STYLES
    # --------------------------------------------------------

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MedicalTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=TA_CENTER,
        spaceAfter=6 * mm
    )

    subtitle_style = ParagraphStyle(
        "MedicalSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=5 * mm
    )

    warning_style = ParagraphStyle(
        "PrototypeWarning",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#8A1C1C"),
        borderColor=colors.HexColor("#D7A6A6"),
        borderWidth=0.7,
        borderPadding=7,
        backColor=colors.HexColor("#FFF4F4"),
        spaceAfter=6 * mm
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#183153"),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm
    )

    normal_style = ParagraphStyle(
        "MedicalNormal",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=2 * mm
    )

    small_style = ParagraphStyle(
        "MedicalSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11
    )

    empty_style = ParagraphStyle(
        "NotDocumented",
        parent=normal_style,
        fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#666666")
    )


    story = []


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Medical Consultation AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Reviewed Medical Consultation Record",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "PROTOTYPE - HUMAN REVIEW WORKFLOW DEMONSTRATION - "
            "NOT FOR CLINICAL USE",
            warning_style
        )
    )


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    approved_at = record.get(
        "approved_at",
        "Not available"
    )

    metadata_table = Table(
        [
            [
                Paragraph(
                    "<b>Encounter ID</b>",
                    small_style
                ),
                Paragraph(
                    _safe(
                        record.get(
                            "encounter_id",
                            encounter_id
                        )
                    ),
                    small_style
                )
            ],
            [
                Paragraph(
                    "<b>Workflow status</b>",
                    small_style
                ),
                Paragraph(
                    "Human reviewed and approved in prototype workflow",
                    small_style
                )
            ],
            [
                Paragraph(
                    "<b>Approved at</b>",
                    small_style
                ),
                Paragraph(
                    _safe(approved_at),
                    small_style
                )
            ]
        ],
        colWidths=[
            42 * mm,
            128 * mm
        ]
    )

    metadata_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F3F5F7")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#CCCCCC")
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor("#DDDDDD")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    story.append(metadata_table)

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )


    # --------------------------------------------------------
    # SOAP
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "SOAP Note",
            section_style
        )
    )

    soap = record.get(
        "soap",
        {}
    )

    for section in [
        "subjective",
        "objective",
        "assessment",
        "plan"
    ]:

        story.append(
            Paragraph(
                section.title(),
                ParagraphStyle(
                    f"{section}_heading",
                    parent=normal_style,
                    fontName="Helvetica-Bold",
                    fontSize=11,
                    textColor=colors.HexColor("#222222"),
                    spaceBefore=3 * mm,
                    spaceAfter=1.5 * mm
                )
            )
        )

        items = soap.get(
            section,
            []
        )

        if not items:

            story.append(
                Paragraph(
                    "Not documented.",
                    empty_style
                )
            )

            continue

        for item in items:

            if isinstance(
                item,
                dict
            ):

                statement = item.get(
                    "statement",
                    ""
                )

                source_ids = item.get(
                    "source_fact_ids",
                    []
                )

            else:

                statement = str(item)
                source_ids = []

            source_text = ""

            if source_ids:

                source_text = (
                    " <font color='#666666'>"
                    "[Source facts: "
                    + _safe(
                        ", ".join(
                            source_ids
                        )
                    )
                    + "]</font>"
                )

            story.append(
                Paragraph(
                    "- "
                    + _safe(statement)
                    + source_text,
                    normal_style
                )
            )


    # --------------------------------------------------------
    # REVIEWED CLINICAL FACTS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Reviewed Clinical Facts",
            section_style
        )
    )

    facts = record.get(
        "clinical_facts",
        []
    )

    fact_rows = [
        [
            Paragraph(
                "<b>ID</b>",
                small_style
            ),
            Paragraph(
                "<b>Category</b>",
                small_style
            ),
            Paragraph(
                "<b>Clinical fact</b>",
                small_style
            ),
            Paragraph(
                "<b>Evidence</b>",
                small_style
            )
        ]
    ]

    for fact in facts:

        fact_rows.append([
            Paragraph(
                _safe(
                    fact.get(
                        "fact_id",
                        ""
                    )
                ),
                small_style
            ),
            Paragraph(
                _safe(
                    fact.get(
                        "category",
                        ""
                    )
                ),
                small_style
            ),
            Paragraph(
                _safe(
                    fact.get(
                        "fact",
                        ""
                    )
                ),
                small_style
            ),
            Paragraph(
                '"'
                + _safe(
                    fact.get(
                        "evidence_quote",
                        ""
                    )
                )
                + '"',
                small_style
            )
        ])

    facts_table = LongTable(
        fact_rows,
        repeatRows=1,
        colWidths=[
            12 * mm,
            27 * mm,
            54 * mm,
            77 * mm
        ]
    )

    facts_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#E9EEF4")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#BFC7D1")
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor("#D8DDE3")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            )
        ])
    )

    story.append(
        facts_table
    )


    # --------------------------------------------------------
    # TRANSCRIPT APPENDIX
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Appendix - Reviewed Transcript",
            section_style
        )
    )

    story.append(
        Paragraph(
            "This transcript is the working transcript accepted "
            "during the prototype human-review workflow.",
            small_style
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm
        )
    )

    transcript = record.get(
        "reviewed_transcript",
        ""
    )

    if transcript:

        for chunk in _transcript_chunks(
            transcript
        ):

            story.append(
                Paragraph(
                    _safe(chunk),
                    normal_style
                )
            )

            story.append(
                Spacer(
                    1,
                    2 * mm
                )
            )

    else:

        story.append(
            Paragraph(
                "No reviewed transcript available.",
                empty_style
            )
        )


    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    document.build(
        story,
        onFirstPage=_page_footer,
        onLaterPages=_page_footer
    )

    return pdf_path


def export_approved_record(
    encounter_id
):

    json_path = (
        export_approved_json(
            encounter_id
        )
    )

    pdf_path = (
        export_approved_pdf(
            encounter_id
        )
    )

    exports_dir = (
        _encounter_dir(
            encounter_id
        )
        / "exports"
    )

    manifest = {
        "encounter_id":
            encounter_id,

        "exported_at":
            datetime.now().isoformat(),

        "prototype_notice":
            (
                "Human review workflow demonstration. "
                "Not for clinical use."
            ),

        "files": {
            "json": {
                "path":
                    str(
                        json_path.relative_to(
                            PROJECT
                        )
                    ),

                "sha256":
                    _sha256(
                        json_path
                    )
            },

            "pdf": {
                "path":
                    str(
                        pdf_path.relative_to(
                            PROJECT
                        )
                    ),

                "sha256":
                    _sha256(
                        pdf_path
                    )
            }
        }
    }

    manifest_path = (
        exports_dir
        / "export_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return {
        "json_path":
            json_path,

        "pdf_path":
            pdf_path,

        "manifest_path":
            manifest_path
    }
