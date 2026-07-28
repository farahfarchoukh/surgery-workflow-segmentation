"""Pure-Python PDF export for the Technical Architecture Report - no
pandoc/node/graphviz dependency (see README). Strips the raw Mermaid code
fence (which renders fine on GitHub but not through this HTML->PDF path)
and replaces it with a pointer note, since the PNG immediately below it in
the Markdown source is the guaranteed-to-render diagram for the PDF.
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

REPORT_DIR = Path(__file__).parent
SOURCE = REPORT_DIR / "technical_architecture_report.md"
OUTPUT = REPORT_DIR / "technical_architecture_report.pdf"

CSS = """
<style>
  @page { size: A4; margin: 1.5cm; }
  body { font-family: Helvetica, Arial, sans-serif; font-size: 9pt; line-height: 1.34; color: #0b0b0b; }
  h1 { font-size: 17pt; margin-bottom: 3pt; }
  h2 { font-size: 12.5pt; margin-top: 11pt; border-bottom: 1pt solid #ccc; padding-bottom: 2pt; }
  h3 { font-size: 10.5pt; margin-top: 7pt; }
  p { margin: 3pt 0; text-align: justify; }
  code { background: #f0f0ee; padding: 1pt 3pt; font-family: Courier, monospace; font-size: 8.5pt; }
  pre {
    background: #f0f0ee; padding: 6pt 8pt; font-family: Courier, monospace;
    font-size: 7.5pt; line-height: 1.25; white-space: pre;
  }
  pre code { background: transparent; padding: 0; }
  table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.5pt; }
  th, td { border: 0.5pt solid #999; padding: 4pt 6pt; text-align: left; }
  th { background: #f0f0ee; }
  img { max-width: 100%; }
  hr { border: none; border-top: 0.5pt solid #ccc; margin: 8pt 0; }
  li { margin: 1.5pt 0; }
  blockquote { color: #52514e; font-style: italic; }
</style>
"""


def strip_mermaid_fence(md_text: str) -> str:
    note = (
        "*(Mermaid diagram source: `report/diagrams/aws_architecture.mmd` - "
        "rendered as a static image immediately below.)*"
    )
    return re.sub(r"```mermaid.*?```", note, md_text, flags=re.DOTALL)


def resolve_relative_uri(uri: str, _rel: str) -> str:
    """xhtml2pdf's `path` kwarg alone doesn't resolve relative <img src>
    paths (tested directly - it silently drops the image and logs a
    warning); an explicit link_callback resolving against REPORT_DIR is the
    documented way to make relative image references work."""
    if uri.startswith(("http://", "https://")):
        return uri
    return str((REPORT_DIR / uri).resolve())


def build_pdf() -> None:
    md_text = strip_mermaid_fence(SOURCE.read_text())
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    full_html = f"<html><head>{CSS}</head><body>{html_body}</body></html>"

    with open(OUTPUT, "wb") as f:
        result = pisa.CreatePDF(full_html, dest=f, link_callback=resolve_relative_uri)
    if result.err:
        raise RuntimeError(f"PDF generation failed with {result.err} error(s)")
    print(f"PDF saved to {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
