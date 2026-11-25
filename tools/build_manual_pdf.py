#!/usr/bin/env python3
# Halim Jarrar, 2026. GPL-3.0-or-later.
"""Builds the instruction manual kokoSD-manual.pdf from docs/manual.md and the
screenshots in docs/screens/ (480 x 320: the GBA's screen doubled, taken in the
emulator). Needs ReportLab, pypdf, Pillow and the DejaVu Sans fonts
(/usr/share/fonts/truetype/dejavu). Made from gbavideo's tools/build_manual_pdf.py.

  python3 tools/build_manual_pdf.py [OUTPUT.pdf]

docs/manual.md is Markdown with one addition: a fenced block "screens" lists
screenshots, one per line: FILE CAPTION. Every paragraph of the Markdown is
checked to be in the PDF's text."""
from pathlib import Path
import io
import os
import re
import sys
from xml.sax.saxutils import escape

from reportlab import rl_config
rl_config.invariant = 1   # reproducible output
from PIL import Image as PILImage  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (CondPageBreak, Flowable, Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

root = Path(__file__).resolve().parents[1]
output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'kokoSD-manual.pdf'
fontdir = Path(os.environ.get('GBV_DEJAVU_DIR', '/usr/share/fonts/truetype/dejavu'))
for name, filename in [('Body', 'DejaVuSans.ttf'), ('BodyBold', 'DejaVuSans-Bold.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(fontdir / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
BLUE = colors.HexColor('#174a8b')
WIDTH = 595.28 - 2 * 56
styles = getSampleStyleSheet()
S = {
    'body': ParagraphStyle('BodyUTF', fontName='Body', fontSize=9, leading=13, spaceAfter=6),
    'bullet': ParagraphStyle('BulletUTF', fontName='Body', fontSize=9, leading=13, spaceAfter=4, leftIndent=12,
                             bulletIndent=2),
    'h1': ParagraphStyle('SectionUTF', fontName='BodyBold', fontSize=14, leading=18, textColor=BLUE, spaceBefore=10,
                         spaceAfter=8, keepWithNext=True),
    'h2': ParagraphStyle('SubUTF', fontName='BodyBold', fontSize=10.5, leading=14, textColor=BLUE, spaceBefore=8,
                         spaceAfter=5, keepWithNext=True),
    'title': ParagraphStyle('TitleUTF', fontName='BodyBold', fontSize=24, leading=29, textColor=BLUE, spaceAfter=4),
    'cell': ParagraphStyle('CellUTF', fontName='Body', fontSize=8, leading=10.5),
    'caption': ParagraphStyle('CaptionUTF', fontName='Body', fontSize=7.5, leading=9.5, alignment=1,
                              textColor=colors.HexColor('#526174')),
    'code': ParagraphStyle('CodeUTF', fontName='Mono', fontSize=7.5, leading=10, leftIndent=6,
                           backColor=colors.HexColor('#f3f6fa'), borderPadding=4, spaceAfter=9),
}

def shot(name):
    return PILImage.open(root / 'docs' / 'screens' / name).convert('RGB')


def image_flowable(pil, width):
    buf = io.BytesIO()
    pil.save(buf, 'PNG')
    buf.seek(0)
    return Image(buf, width=width, height=width * pil.height / pil.width)


def screens_block(lines):
    cells = []
    for line in lines:
        name, caption = line.split(' ', 1)
        cells.append([image_flowable(shot(name), 150), Paragraph(markup(caption), S['caption'])])
    rows = [cells[i:i + 3] for i in range(0, len(cells), 3)]
    rows = [r + [''] * (3 - len(r)) for r in rows]
    t = Table(rows, colWidths=[WIDTH / 3] * 3, hAlign='CENTER')
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    return [KeepTogether(t), Spacer(1, 4)]


def markup(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<![\w*])\*([^*]+)\*(?![\w*])', r'<i>\1</i>', text)
    return re.sub(r'`([^`]*)`', r'<font name="Mono">\1</font>', text)


def table(lines):
    rows = []
    for ln in lines:
        row = [c.strip() for c in re.split(r'(?<!\\)\|', ln.strip().strip('|'))]
        if not all(re.fullmatch(r'[-:]+', c) for c in row):
            rows.append([Paragraph(markup(c), S['cell']) for c in row])
    n = len(rows[0])
    widths = {2: [0.3, 0.7], 3: [0.2, 0.32, 0.48]}.get(n, [1 / n] * n)
    t = Table(rows, colWidths=[w * WIDTH for w in widths], repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eaf1fa')),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.6, BLUE),
                           ('LINEBELOW', (0, 1), (-1, -1), 0.25, colors.HexColor('#d5dde8')),
                           ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                           ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    return [KeepTogether(t) if len(rows) <= 12 else t, Spacer(1, 8)]


source = (root / 'docs' / 'manual.md').read_text(encoding='utf-8')
version = re.search(r'^# kokoSD (\d+\.\d+)$', source, re.M).group(1)
lines = source.splitlines()
story: list[Flowable] = []
checks = []                       # texts that must be in the PDF
i = 0
para: list[str] = []


def flush():
    if para:
        text = ' '.join(para)
        story.append(Paragraph(markup(text), S['body']))
        checks.append(text)
        para.clear()


while i < len(lines):
    line = lines[i].rstrip()
    if line.startswith('```'):
        flush()
        kind = line[3:].strip()
        block = []
        i += 1
        while not lines[i].startswith('```'):
            block.append(lines[i])
            i += 1
        i += 1
        story.extend(screens_block(block) if kind == 'screens' else [Preformatted('\n'.join(block), S['code'])])
        continue
    if line.startswith('|'):
        flush()
        block = []
        while i < len(lines) and lines[i].startswith('|'):
            block.append(lines[i])
            i += 1
        story.extend(table(block))
        continue
    if line.startswith('!['):
        flush()
        m = re.match(r'!\[(.*)\]\((.*)\)', line)
        pil = PILImage.open(root / 'docs' / m.group(2)).convert('RGB')
        story.extend([Spacer(1, 4), image_flowable(pil, WIDTH), Paragraph(markup(m.group(1)), S['caption']),
                      Spacer(1, 8)])
        i += 1
        continue
    if line.startswith('# '):
        flush()
        story.append(Paragraph(markup(line[2:]), S['title']))
    elif line.startswith('## '):
        flush()
        head = line[3:]
        if head == 'Instruction manual':
            story.append(Paragraph('Instruction manual · Halim Jarrar', S['h1']))
        else:
            if head in ('About / licence',):
                story.append(PageBreak())
            else:
                story.append(CondPageBreak(160))
            story.append(Paragraph(markup(head), S['h1']))
    elif line.startswith('### '):
        flush()
        story.append(CondPageBreak(120))
        story.append(Paragraph(markup(line[4:]), S['h2']))
    elif line.startswith('- '):
        flush()
        item = [line[2:]]
        i += 1
        while i < len(lines) and lines[i].startswith('  ') and lines[i].strip():
            item.append(lines[i].strip())
            i += 1
        text = ' '.join(item)
        story.append(Paragraph(markup(text), S['bullet'], bulletText='•'))
        checks.append(text)
        continue
    elif re.match(r'\d+\. ', line):
        flush()
        num, rest = line.split(' ', 1)
        item = [rest]
        i += 1
        while i < len(lines) and lines[i].startswith('   ') and lines[i].strip():
            item.append(lines[i].strip())
            i += 1
        text = ' '.join(item)
        story.append(Paragraph(markup(text), S['bullet'], bulletText=num))
        checks.append(text)
        continue
    elif not line.strip():
        flush()
    else:
        para.append(line.strip())
    i += 1
flush()


def footer(canvas, doc):
    canvas.setFont('Body', 8)
    canvas.setFillColor(colors.HexColor('#526174'))
    canvas.drawString(56, 28, f'kokoSD {version} · Instruction manual · Halim Jarrar')
    canvas.drawRightString(595.28 - 56, 28, str(doc.page))


SimpleDocTemplate(str(output), pagesize=(595.28, 841.89), rightMargin=56, leftMargin=56, topMargin=46,
                  bottomMargin=50, title=f'kokoSD {version} — Instruction manual', author='Halim Jarrar',
                  subject='kokoSD: Koko the Chatbot on the Game Boy Advance, with an SD card',
                  creator='tools/build_manual_pdf.py').build(story, onFirstPage=footer, onLaterPages=footer)

# Everything of the Markdown is in the PDF.
pdf = PdfReader(str(output))
text = ''.join(p.extract_text() for p in pdf.pages)
text = re.sub(r'kokoSD [0-9.]+ · Instruction manual · Halim Jarrar\n\d+\n', '', text)   # the footers (a paragraph may run across pages)
seen = re.sub(r'\s+', '', text)


def plain(t):
    t = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', t)
    return re.sub(r'\s+', '', t.replace('**', '').replace('`', '').replace('*', ''))


missing = [c for c in checks if plain(c) not in seen]
if missing:
    sys.exit('not in the PDF: ' + ' | '.join(m[:60] for m in missing))
print(f'{output}: {len(pdf.pages)} pages; every paragraph checked')
