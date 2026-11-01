"""Build exactly two A4 text pages plus a separate A4 reference page from Markdown."""
import argparse
import html
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer, Table, TableStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="black">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="9">\1</font>', text)
    return text


def build(source, output):
    parts = source.read_text(encoding="utf-8").split("<!-- PAGE BREAK -->")
    if len(parts) != 3:
        raise ValueError("Markdown must contain exactly three explicit page sections")
    styles = {
        "body": ParagraphStyle("Body", fontName="Times-Roman", fontSize=11.2, leading=15.2,
                               spaceAfter=8, alignment=TA_LEFT),
        "title": ParagraphStyle("Title", fontName="Times-Bold", fontSize=17, leading=20, spaceAfter=13),
        "section": ParagraphStyle("Section", fontName="Times-Bold", fontSize=11.8, leading=15,
                                  spaceBefore=8, spaceAfter=5, keepWithNext=True),
        "reference": ParagraphStyle("Reference", fontName="Times-Roman", fontSize=10.5, leading=14.1, spaceAfter=13),
        "cell": ParagraphStyle("Cell", fontName="Times-Roman", fontSize=9.5, leading=12),
    }
    story = []
    for page_index, part in enumerate(parts):
        lines = part.strip().splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1; continue
            if line.startswith("# "):
                story.append(Paragraph(inline(line[2:]), styles["title"]))
            elif line.startswith("## "):
                story.append(Paragraph(inline(line[3:]), styles["section"]))
            elif line.startswith("|"):
                table_rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r"[:\-\s]+", c) for c in cells):
                        table_rows.append([Paragraph(inline(c), styles["cell"]) for c in cells])
                    index += 1
                width = A4[0] - 116
                widths = [width * 0.34] + [width * 0.66 / (len(table_rows[0])-1)] * (len(table_rows[0])-1)
                table = Table(table_rows, colWidths=widths, hAlign="LEFT")
                table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story.extend([table, Spacer(1, 9)])
                continue
            else:
                paragraph = [line]
                while index+1 < len(lines) and lines[index+1].strip() and not lines[index+1].startswith(("#", "|")):
                    index += 1; paragraph.append(lines[index].strip())
                style = styles["reference"] if page_index == 2 else styles["body"]
                story.append(Paragraph(inline(" ".join(paragraph)), style))
            index += 1
        if page_index != 2:
            story.append(PageBreak())
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=58, leftMargin=58,
                            topMargin=48, bottomMargin=49, title="CLIP Background Bias Research Brief",
                            author="AIST1000 seminar project", pageCompression=1)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.black); canvas.setLineWidth(0.3)
        canvas.line(58, 36, A4[0]-58, 36)
        canvas.setFont("Times-Roman", 9)
        canvas.drawString(58, 23, "AIST1000 | Object-centric processing and background sensitivity")
        canvas.drawRightString(A4[0]-58, 23, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf = PdfReader(output)
    if len(pdf.pages) != 3:
        raise ValueError(f"Expected exactly 3 pages, got {len(pdf.pages)}; adjust source layout")
    for page in pdf.pages:
        if abs(float(page.mediabox.width) - A4[0]) > 1 or abs(float(page.mediabox.height) - A4[1]) > 1:
            raise ValueError("PDF is not A4")
    assert "References" in pdf.pages[2].extract_text()
    print(f"Created three A4 pages: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/research_brief.md")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/research_brief.pdf")
    args = parser.parse_args()
    build(args.source, args.output)
