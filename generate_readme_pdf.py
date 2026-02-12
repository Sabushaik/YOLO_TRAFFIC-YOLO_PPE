"""
Script to convert README.md to PDF with all text preserved.

Usage:
    python generate_readme_pdf.py

Requirements:
    pip install fpdf2 markdown
"""

import os

import markdown
from fpdf import FPDF
from fpdf.html import FontFace

# Emoji-to-text mapping for characters unsupported by standard PDF fonts
EMOJI_REPLACEMENTS = {
    "\U0001f534": "[!]",       # 🔴
    "\u2705": "[OK]",          # ✅
    "\u26a0\ufe0f": "[WARN]",  # ⚠️
    "\U0001f6a8": "[ALERT]",   # 🚨
}

# Font paths (DejaVu Sans ships with most Linux distributions)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONTS = {
    "DejaVu": {
        "": os.path.join(FONT_DIR, "DejaVuSans.ttf"),
        "B": os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"),
        "I": os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf"),
        "BI": os.path.join(FONT_DIR, "DejaVuSans-BoldOblique.ttf"),
    },
    "DejaVuMono": {
        "": os.path.join(FONT_DIR, "DejaVuSansMono.ttf"),
        "B": os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf"),
        "I": os.path.join(FONT_DIR, "DejaVuSansMono-Oblique.ttf"),
        "BI": os.path.join(FONT_DIR, "DejaVuSansMono-BoldOblique.ttf"),
    },
}


def generate_pdf(md_path="README.md", output_path="README.pdf"):
    """Read a Markdown file and write it as a PDF."""
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Replace emoji characters that are not in the font
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        md_text = md_text.replace(emoji, replacement)

    # Convert Markdown to HTML
    html = markdown.markdown(md_text, extensions=["fenced_code", "tables"])

    # Build the PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Register Unicode-capable fonts
    for family, styles in FONTS.items():
        for style, path in styles.items():
            pdf.add_font(family, style, path)

    pdf.set_font("DejaVu", size=12)

    pdf.write_html(
        html,
        tag_styles={
            "h1": FontFace(family="DejaVu", size_pt=20),
            "h2": FontFace(family="DejaVu", size_pt=16),
            "h3": FontFace(family="DejaVu", size_pt=13),
            "code": FontFace(family="DejaVuMono", size_pt=10),
            "pre": FontFace(family="DejaVuMono", size_pt=10),
            "blockquote": FontFace(family="DejaVu", size_pt=11, color="#555555"),
        },
    )

    pdf.output(output_path)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(script_dir, "README.md")
    output_path = os.path.join(script_dir, "README.pdf")
    generate_pdf(md_path, output_path)
