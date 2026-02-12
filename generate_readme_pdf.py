"""
Script to convert README.md to PDF with all text preserved.

Usage:
    python generate_readme_pdf.py

Requirements:
    pip install fpdf2 markdown
"""

import os
import sys

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

# Common DejaVu Sans font directories across operating systems
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu",          # Debian / Ubuntu
    "/usr/share/fonts/dejavu-sans-fonts",         # Fedora / RHEL
    "/usr/share/fonts/dejavu",                    # Arch / openSUSE
    "/usr/local/share/fonts/dejavu",              # manual install on Linux
    "/opt/homebrew/share/fonts/dejavu",           # macOS Homebrew (Apple Silicon)
    "/usr/local/share/fonts",                     # macOS Homebrew (Intel)
    os.path.expanduser("~/.local/share/fonts"),   # user-level install
    "C:\\Windows\\Fonts",                          # Windows
]

# Required font file basenames per family
_FONT_FILES = {
    "DejaVu": {
        "": "DejaVuSans.ttf",
        "B": "DejaVuSans-Bold.ttf",
        "I": "DejaVuSans-Oblique.ttf",
        "BI": "DejaVuSans-BoldOblique.ttf",
    },
    "DejaVuMono": {
        "": "DejaVuSansMono.ttf",
        "B": "DejaVuSansMono-Bold.ttf",
        "I": "DejaVuSansMono-Oblique.ttf",
        "BI": "DejaVuSansMono-BoldOblique.ttf",
    },
}


def _find_font_dir():
    """Return the first directory that contains all required font files."""
    required = {name for styles in _FONT_FILES.values() for name in styles.values()}
    for directory in _FONT_SEARCH_PATHS:
        if os.path.isdir(directory) and required.issubset(os.listdir(directory)):
            return directory
    return None


def generate_pdf(md_path="README.md", output_path="README.pdf"):
    """Read a Markdown file and write it as a PDF."""
    if not os.path.isfile(md_path):
        print(f"Error: Markdown file not found: {md_path}", file=sys.stderr)
        sys.exit(1)

    font_dir = _find_font_dir()
    if font_dir is None:
        print(
            "Error: DejaVu Sans fonts not found. Install them with:\n"
            "  Debian/Ubuntu: sudo apt install fonts-dejavu-core\n"
            "  Fedora/RHEL:   sudo dnf install dejavu-sans-fonts\n"
            "  macOS:         brew install font-dejavu\n"
            "  Windows:       download from https://dejavu-fonts.github.io/",
            file=sys.stderr,
        )
        sys.exit(1)

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
    for family, styles in _FONT_FILES.items():
        for style, filename in styles.items():
            pdf.add_font(family, style, os.path.join(font_dir, filename))

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
