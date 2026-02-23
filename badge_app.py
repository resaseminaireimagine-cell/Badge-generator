import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageOps

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================
# CONFIG / CHARTE IMAGINE
# =========================
APP_TITLE = "Badge generator — Imagine (A4, Recto/Verso)"
IMAGINE_ROSE = "#C4007A"
BLACK = "#0B1220"
WHITE = "#FFFFFF"
BAND_DEFAULT = "#C9C4A6"

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
CENTER_IMAGE_PATH = ASSETS / "center_image.jpg"

FONT_REG = ASSETS / "Montserrat-Regular.ttf"
FONT_BOLD = ASSETS / "Montserrat-Bold.ttf"
FONT_REG_URL = "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Regular.ttf"
FONT_BOLD_URL = "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Bold.ttf"


# =========================
# FONTS
# =========================
def _download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def ensure_fonts() -> Tuple[Dict[str, str], str]:
    """
    Retourne (fonts_map, status_str)
    """
    fonts = {"regular": "Helvetica", "bold": "Helvetica-Bold"}
    status = "Helvetica (fallback)"

    if ASSETS.exists() and ASSETS.is_dir():
        if not FONT_REG.exists():
            _download(FONT_REG_URL, FONT_REG)
        if not FONT_BOLD.exists():
            _download(FONT_BOLD_URL, FONT_BOLD)

    try:
        if FONT_REG.exists() and FONT_BOLD.exists():
            pdfmetrics.registerFont(TTFont("Montserrat", str(FONT_REG)))
            pdfmetrics.registerFont(TTFont("Montserrat-Bold", str(FONT_BOLD)))
            fonts["regular"] = "Montserrat"
            fonts["bold"] = "Montserrat-Bold"
            status = "Montserrat ✅"
    except Exception:
        pass

    return fonts, status


# =========================
# IMAGE HELPERS
# =========================
def pil_to_reader(img: Image.Image) -> ImageReader:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return ImageReader(bio)


def remove_near_black_to_transparent(img: Image.Image, threshold: int = 18) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")
    px = img.getdata()
    out = []
    for r, g, b, a in px:
        if r < threshold and g < threshold and b < threshold:
            out.append((r, g, b, 0))
        else:
            out.append((r, g, b, a))
    img.putdata(out)
    return img


def make_circle_image(img: Image.Image, diameter_px: int, remove_black: bool, threshold: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")
    if remove_black:
        img = remove_near_black_to_transparent(img, threshold=threshold)

    # Fit (pas de crop sauvage)
    w, h = img.size
    scale = min(diameter_px / w, diameter_px / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)

    canvas_img = Image.new("RGBA", (diameter_px, diameter_px), (0, 0, 0, 0))
    ox = (diameter_px - nw) // 2
    oy = (diameter_px - nh) // 2
    canvas_img.paste(img, (ox, oy), img)

    mask = Image.new("L", (diameter_px, diameter_px), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((0, 0, diameter_px - 1, diameter_px - 1), fill=255)

    out = Image.new("RGBA", (diameter_px, diameter_px), (0, 0, 0, 0))
    out.paste(canvas_img, (0, 0), mask=mask)
    return out


def draw_image_fit(c: canvas.Canvas, img: Image.Image, x: float, y: float, w: float, h: float):
    img = ImageOps.exif_transpose(img).convert("RGBA")
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return
    scale = min(w / iw, h / ih)
    nw, nh = iw * scale, ih * scale
    ox, oy = x + (w - nw) / 2, y + (h - nh) / 2
    c.drawImage(pil_to_reader(img), ox, oy, width=nw, height=nh, mask="auto")


# =========================
# TEXT HELPERS
# =========================
def fit_font_size(c: canvas.Canvas, text: str, font: str, max_w: float, start: int, min_size: int = 7) -> int:
    size = start
    while size > min_size and c.stringWidth(text, font, size) > max_w:
        size -= 1
    return max(size, min_size)


def lines(text: str) -> List[str]:
    return [t.strip() for t in (text or "").split("\n") if t.strip()]


# =========================
# DESIGN
# =========================
@dataclass
class BadgeDesign:
    # "badge" de base = A5 (148x210)
    badge_w_mm: float = 148
    badge_h_mm: float = 210
    band_ratio: float = 0.28

    bg_black: str = BLACK
    band_color: str = BAND_DEFAULT
    accent: str = IMAGINE_ROSE

    margin_mm: float = 8  # un poil plus serré
    circle_d_mm: float = 62
    circle_y_offset_mm: float = -6  # mieux centré

    # tailles
    date_big: int = 22
    date_small: int = 12
    edition_size: int = 10
    title_size: int = 20
    tagline_size: int = 10
    organised_by_size: int = 7
    role_size: int = 16


# =========================
# LAYOUT A4 UNIQUEMENT
# =========================
def layout_positions(layout_name: str, margin_mm: float, gap_mm: float) -> Tuple[Tuple[float, float], Tuple[float, float], List[Tuple[float, float]]]:
    """
    A4 uniquement.
    - "A4 paysage — 2 badges (GRAND)" => 2 x A5 (148x210) côte à côte, nickel.
    - "A4 portrait — 4 badges (PETIT)" => 4 x A6 (105x148).
    """
    m = margin_mm * mm
    g = gap_mm * mm

    if layout_name == "A4 paysage — 2 badges (GRAND)":
        page_w, page_h = landscape(A4)
        badge_w, badge_h = 148 * mm, 210 * mm
        # 2 colonnes, 1 ligne
        positions = [
            (m, page_h - m - badge_h),
            (m + badge_w + g, page_h - m - badge_h),
        ]
        return (page_w, page_h), (badge_w, badge_h), positions

    # 4-up (A6) sur A4 portrait
    page_w, page_h = A4
    badge_w, badge_h = 105 * mm, 148 * mm
    positions = [
        (m, page_h - m - badge_h),
        (m + badge_w + g, page_h - m - badge_h),
        (m, page_h - m - 2 * badge_h - g),
        (m + badge_w + g, page_h - m - 2 * badge_h - g),
    ]
    return (page_w, page_h), (badge_w, badge_h), positions


# =========================
# DRAW RECTO
# =========================
def draw_recto(
    c: canvas.Canvas,
    fonts: Dict[str, str],
    design: BadgeDesign,
    x: float,
    y: float,
    badge_w: float,
    badge_h: float,
    cfg: dict,
    person: dict,
    center_img: Image.Image,
    sponsor_imgs: List[Image.Image],
):
    # proportions band bas
    band_h = badge_h * design.band_ratio
    top_h = badge_h - band_h
    pad = 6 * mm
    margin = design.margin_mm * mm

    # fonds
    c.setFillColor(HexColor(design.bg_black))
    c.rect(x, y + band_h, badge_w, top_h, stroke=0, fill=1)
    c.setFillColor(HexColor(design.band_color))
    c.rect(x, y, badge_w, band_h, stroke=0, fill=1)

    # ---- date top-left (vraiment en haut)
    if cfg["show_date"]:
        bx = x + pad
        by = y + badge_h - pad - 18 * mm  # plus haut

        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], design.date_big)
        c.drawString(bx, by + 9 * mm, cfg["date_l1"])

        c.setFont(fonts["regular"], design.date_small)
        c.drawString(bx, by + 3.5 * mm, cfg["date_l2"])
        c.drawString(bx, by - 2.0 * mm, cfg["date_l3"])

        c.setFillColor(HexColor(design.accent))
        c.rect(bx, by + 1.2 * mm, 10 * mm, 1.2 * mm, stroke=0, fill=1)

    # ---- édition top-right (vraiment en haut à droite)
    if cfg["show_edition"] and cfg["edition_text"].strip():
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["regular"], design.edition_size)
        c.drawRightString(x + badge_w - pad, y + badge_h - pad - 5 * mm, cfg["edition_text"].strip())

    # ---- cercle image (centré)
    circle_d = cfg["circle_d_mm"] * mm
    ccx = x + badge_w / 2
    ccy = y + band_h + top_h * 0.55 + (cfg["circle_y_offset_mm"] * mm)

    dpx = int(cfg["circle_d_mm"] * 12)
    circ = make_circle_image(center_img, dpx, cfg["remove_black_bg"], cfg["black_threshold"])
    c.drawImage(
        pil_to_reader(circ),
        ccx - circle_d / 2,
        ccy - circle_d / 2,
        width=circle_d,
        height=circle_d,
        mask="auto",
    )

    # ---- titre & tagline (centrés, fit)
    if cfg["show_title"]:
        title = cfg["event_title"].strip()
        title_lines = lines(title)[:3]
        c.setFillColor(HexColor(WHITE))

        # largeur max = 0.9 du cercle
        max_w = circle_d * 0.9
        base_size = design.title_size

        # Ajuste taille sur la ligne la plus longue
        longest = max(title_lines, key=len) if title_lines else ""
        tsize = fit_font_size(c, longest, fonts["bold"], max_w, base_size, min_size=12)

        c.setFont(fonts["bold"], tsize)
        lh = (tsize + 3) * 0.9
        total_h = lh * len(title_lines)
        ty = ccy + total_h / 2 - lh

        for ln in title_lines:
            c.drawCentredString(ccx, ty, ln)
            ty -= lh

        if cfg["show_tagline"] and cfg["event_tagline"].strip():
            tag = cfg["event_tagline"].strip()
            c.setFillColor(HexColor(design.accent))
            tag_size = min(design.tagline_size, max(8, tsize - 10))
            tag_size = fit_font_size(c, tag, fonts["bold"], max_w, tag_size, min_size=7)
            c.setFont(fonts["bold"], tag_size)
            c.drawCentredString(ccx, ccy - circle_d * 0.28, tag)

    # ---- sponsors
    if cfg["show_sponsors"] and sponsor_imgs:
        if cfg["show_organised_by"] and cfg["organised_by_label"].strip():
            c.setFillColor(HexColor(WHITE))
            c.setFont(fonts["regular"], design.organised_by_size)
            c.drawString(x + margin, y + band_h + 20 * mm, cfg["organised_by_label"].strip())

        logos = sponsor_imgs[: cfg["max_sponsors"]]
        n = len(logos)
        gap = 2.5 * mm
        box_h = 12 * mm
        avail_w = badge_w - 2 * margin - gap * (n - 1)
        box_w = avail_w / n
        box_y = y + band_h + 6 * mm

        for i, lg in enumerate(logos):
            bx = x + margin + i * (box_w + gap)
            c.setFillColor(HexColor(WHITE))
            c.rect(bx, box_y, box_w, box_h, stroke=0, fill=1)
            draw_image_fit(c, lg, bx + 1.2 * mm, box_y + 1.2 * mm, box_w - 2.4 * mm, box_h - 2.4 * mm)

    # ---- bande basse : catégorie (fit pour éviter débordement)
    if cfg["show_role_band"]:
        if cfg["role_source"] == "Texte fixe":
            role = cfg["role_fixed"].strip().upper()
        else:
            role = str(person.get(cfg["role_col"], "")).strip().upper()

        if role:
            max_w = badge_w - 2 * margin
            c.setFillColor(HexColor("#111111"))
            size = fit_font_size(c, role, fonts["bold"], max_w, design.role_size, min_size=8)
            c.setFont(fonts["bold"], size)
            c.drawCentredString(x + badge_w / 2, y + band_h / 2 - 3 * mm, role)


# =========================
# DRAW VERSO (tout doit rentrer)
# =========================
def draw_verso(c: canvas.Canvas, fonts: Dict[str, str], design: BadgeDesign, x: float, y: float, badge_w: float, badge_h: float, cfg: dict):
    c.setFillColor(HexColor(design.bg_black))
    c.rect(x, y, badge_w, badge_h, stroke=0, fill=1)

    margin = design.margin_mm * mm
    usable_top = y + badge_h - margin
    usable_bottom = y + margin + (18 * mm if cfg["back_show_hashtag"] else 0)

    # compile sections
    enabled_sections = []
    for s in cfg["back_sections"]:
        if s["show"]:
            title = (s["title"] or "").strip().upper()
            content_lines = lines(s["lines"])
            if title or content_lines:
                enabled_sections.append((title, content_lines))

    # compute needed height with base sizes
    title_size = 14
    line_size = 11
    title_gap = 8 * mm
    line_gap = 6 * mm
    section_gap = 8 * mm

    # optional event name
    header_h = 0
    if cfg["back_show_event_name"] and cfg["back_event_name"].strip():
        header_h = 18 * mm

    def estimate_height(ts: int, ls: int) -> float:
        h = header_h
        for t, lns in enabled_sections:
            if t:
                h += (ts + 3) * 0.9 / 72 * 25.4 * mm  # approximate -> overkill not needed
            # simpler fixed mm:
            h += title_gap
            h += len(lns) * line_gap
            h += section_gap
        return h

    available = usable_top - usable_bottom - header_h

    # scale down if needed
    # (on garde simple : si trop long, on baisse les tailles jusqu'à rentrer)
    while True:
        needed = 0
        for t, lns in enabled_sections:
            if t:
                needed += (title_size + 6) * 0.6 * mm
            needed += len(lns) * (line_size + 5) * 0.45 * mm
            needed += 7 * mm
        if needed <= available or (title_size <= 11 and line_size <= 9):
            break
        title_size -= 1
        line_size -= 1

    cursor_y = usable_top

    # header event name
    if cfg["back_show_event_name"] and cfg["back_event_name"].strip():
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], 12)
        c.drawCentredString(x + badge_w / 2, cursor_y - 6 * mm, cfg["back_event_name"].strip()[:80])
        cursor_y -= header_h

    # sections
    for t, lns in enabled_sections:
        if cursor_y < usable_bottom + 10 * mm:
            break

        if t:
            c.setFillColor(HexColor(WHITE))
            c.setFont(fonts["bold"], title_size)
            c.drawCentredString(x + badge_w / 2, cursor_y, t)
            cursor_y -= 9 * mm

        c.setFont(fonts["regular"], line_size)
        for ln in lns:
            c.drawCentredString(x + badge_w / 2, cursor_y, ln[:90])
            cursor_y -= 6 * mm

        cursor_y -= 8 * mm

    # hashtag
    if cfg["back_show_hashtag"] and cfg["back_hashtag"].strip():
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], 14)
        c.drawCentredString(x + badge_w / 2, y + 18 * mm, cfg["back_hashtag"].strip())


def cut_marks(c: canvas.Canvas, x: float, y: float, w: float, h: float):
    c.setStrokeColor(HexColor("#9CA3AF"))
    c.setLineWidth(0.3)
    L = 5 * mm
    c.line(x, y, x + L, y); c.line(x, y, x, y + L)
    c.line(x + w, y, x + w - L, y); c.line(x + w, y, x + w, y + L)
    c.line(x, y + h, x + L, y + h); c.line(x, y + h, x, y + h - L)
    c.line(x + w, y + h, x + w - L, y + h); c.line(x + w, y + h, x + w, y + h - L)


# =========================
# PDF GEN
# =========================
def generate_pdf(df: pd.DataFrame, cfg: dict, design: BadgeDesign, sponsor_imgs: List[Image.Image]) -> bytes:
    fonts, _ = ensure_fonts()

    if not CENTER_IMAGE_PATH.exists():
        raise RuntimeError("assets/center_image.jpg introuvable")

    center_img = Image.open(CENTER_IMAGE_PATH)

    (page_w, page_h), (badge_w, badge_h), positions = layout_positions(cfg["layout"], cfg["page_margin_mm"], cfg["gap_mm"])

    # si layout 4-up, badge = A6 : on redimensionne le design (propre)
    local_design = design
    if cfg["layout"] == "A4 portrait — 4 badges (PETIT)":
        local_design = BadgeDesign(
            badge_w_mm=105,
            badge_h_mm=148,
            band_ratio=design.band_ratio,
            bg_black=design.bg_black,
            band_color=design.band_color,
            accent=design.accent,
            margin_mm=6,
            circle_d_mm=max(42, design.circle_d_mm * 0.72),
            circle_y_offset_mm=design.circle_y_offset_mm,
            date_big=14,
            date_small=9,
            edition_size=8,
            title_size=14,
            tagline_size=9,
            organised_by_size=6,
            role_size=12,
        )

    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=(page_w, page_h))

    rows = df.to_dict(orient="records")
    step = len(positions)  # 2 ou 4 badges par page

    def draw_page_recto(start_idx: int):
        idx = start_idx
        for p, (px, py) in enumerate(positions):
            if idx >= len(rows):
                break
            draw_recto(c, fonts, local_design, px, py, badge_w, badge_h, cfg, rows[idx], center_img, sponsor_imgs)
            if cfg["cut_marks"]:
                cut_marks(c, px, py, badge_w, badge_h)
            idx += 1

    def draw_page_verso(_start_idx: int):
        # verso identique pour tous (contacts), pas dépendant des personnes
        for (px, py) in positions:
            draw_verso(c, fonts, local_design, px, py, badge_w, badge_h, cfg)
            if cfg["cut_marks"]:
                cut_marks(c, px, py, badge_w, badge_h)

    i = 0
    while i < len(rows):
        if cfg["mode"] == "Recto seul":
            draw_page_recto(i)
            c.showPage()
        elif cfg["mode"] == "Verso seul":
            draw_page_verso(i)
            c.showPage()
        else:  # Recto puis Verso
            draw_page_recto(i)
            c.showPage()
            draw_page_verso(i)
            c.showPage()

        i += step

    c.save()
    return out.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# sanity assets
if ASSETS.exists() and not ASSETS.is_dir():
    st.error("`assets` est un fichier. Il doit être un dossier `assets/`.")
    st.stop()

if not CENTER_IMAGE_PATH.exists():
    st.error("Fichier obligatoire manquant : `assets/center_image.jpg` (upload-le dans GitHub).")
    st.stop()

fonts, font_status = ensure_fonts()
st.caption(f"Police utilisée pour le PDF : **{font_status}**")

with st.expander("Image recto (fixe) utilisée pour tous les badges", expanded=False):
    st.image(Image.open(CENTER_IMAGE_PATH), use_container_width=True)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("1) Données participants")
    file = st.file_uploader("CSV ou Excel", type=["csv", "xlsx"])
    df = None
    if file is not None:
        df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        st.dataframe(df.head(20), use_container_width=True)

with right:
    st.subheader("2) Logos sponsors (optionnel)")
    sponsors_up = st.file_uploader("Logos (PNG conseillé) — multiple", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    sponsor_imgs = [Image.open(f) for f in sponsors_up] if sponsors_up else []

    st.subheader("3) Couleurs")
    bg_black = st.color_picker("Fond haut", BLACK)
    band_color = st.color_picker("Bande basse", BAND_DEFAULT)
    accent = st.color_picker("Accent Imagine", IMAGINE_ROSE)

st.divider()

st.subheader("4) Mise en page A4 + export")
layout = st.selectbox("A4 uniquement : choisis la taille", ["A4 paysage — 2 badges (GRAND)", "A4 portrait — 4 badges (PETIT)"], index=0)
mode = st.selectbox("Export", ["Recto puis Verso", "Recto seul", "Verso seul"], index=0)
page_margin_mm = st.number_input("Marge page (mm)", 0, 20, 6)
gap_mm = st.number_input("Espace entre badges (mm)", 0, 20, 6)
cutm = st.checkbox("Traits de coupe", True)

st.divider()
st.subheader("5) Recto — blocs")

c1, c2, c3 = st.columns(3)
with c1:
    show_date = st.checkbox("Afficher la date (haut gauche)", True)
    date_l1 = st.text_input("Date L1", "Mardi 18")
    date_l2 = st.text_input("Date L2", "Février")
    date_l3 = st.text_input("Date L3", "2026")

    remove_black_bg = st.checkbox("Retirer fond noir de l'image", True)
    black_threshold = st.slider("Seuil noir→transparent", 5, 60, 18)

with c2:
    show_edition = st.checkbox("Afficher édition (haut droite)", True)
    edition_text = st.text_input("Texte édition", "1ere édition")

    show_title = st.checkbox("Afficher titre au centre", True)
    event_title = st.text_area("Titre (1 ligne = 1 ligne)", "Grand Patrie", height=80)

    show_tagline = st.checkbox("Afficher tagline", True)
    event_tagline = st.text_input("Tagline", "L'avenir de demain")

with c3:
    show_sponsors = st.checkbox("Afficher logos sponsors", True)
    show_organised_by = st.checkbox("Afficher “Organisé par”", True)
    organised_by_label = st.text_input("Libellé", "Organisé par")
    max_sponsors = st.slider("Max logos", 1, 8, 4)

    circle_d_mm = st.slider("Diamètre image (mm)", 40, 85, 62)
    circle_y_offset_mm = st.slider("Décalage vertical image (mm)", -30, 30, -6)

st.divider()
st.subheader("6) Bande basse (catégorie) — plus de débordement")

role_source = st.selectbox("Source du texte bande basse", ["Colonne du fichier", "Texte fixe"], index=0)
role_fixed = st.text_input("Texte fixe (ex: INTERVENANT)", "INTERVENANT")
show_role_band = st.checkbox("Afficher la bande basse", True)

st.divider()
st.subheader("7) Verso — sections (ce que tu saisis = ce qui sort)")

back_show_event_name = st.checkbox("Verso : afficher nom évènement en haut", False)
back_event_name = st.text_input("Nom évènement (verso)", "Grand Patrie")

sec1, sec2 = st.columns(2)
default_sections = [
    ("ORGANISATION", "Ambroise Lelevé\n06 18 87 26 73"),
    ("LOGISTIQUE", "Anne TISLER\n+33 7 81 18 40 94\nAurélie MESTELAN\n+33 6 38 22 26 36"),
    ("PARTENAIRES", "Justine COUTEILLE\n+33 6 75 58 75 32\nCharlotte M’NASRI\n+33 6 74 47 11 99"),
    ("ÉDITORIAL", "Romain GONZALEZ\n+33 7 69 26 95 89"),
]

back_sections = []
for i, (t, content) in enumerate(default_sections):
    col = sec1 if i % 2 == 0 else sec2
    with col:
        show = st.checkbox(f"Afficher {t}", value=True, key=f"sec_show_{i}")
        title = st.text_input(f"Titre {i+1}", t, key=f"sec_title_{i}")
        text = st.text_area(f"Contenu {i+1}", content, height=120, key=f"sec_text_{i}")
        back_sections.append({"show": show, "title": title, "lines": text})

back_show_hashtag = st.checkbox("Verso : afficher hashtag", True)
back_hashtag = st.text_input("Hashtag", "#Demain c'est top")

st.divider()
st.subheader("8) Mapping colonnes (pour la catégorie si tu la prends du fichier)")

if df is None:
    st.info("Charge un fichier pour activer la génération.")
    st.stop()

cols = list(df.columns)
role_col = st.selectbox("Colonne catégorie (ex: INTERVENANT)", cols, index=cols.index("categorie") if "categorie" in cols else 0)

st.divider()

cfg = {
    "layout": layout,
    "mode": mode,
    "page_margin_mm": float(page_margin_mm),
    "gap_mm": float(gap_mm),
    "cut_marks": cutm,

    "bg_black": bg_black,
    "band_color": band_color,
    "accent": accent,

    "show_date": show_date,
    "date_l1": date_l1,
    "date_l2": date_l2,
    "date_l3": date_l3,

    "remove_black_bg": remove_black_bg,
    "black_threshold": int(black_threshold),

    "show_edition": show_edition,
    "edition_text": edition_text,

    "show_title": show_title,
    "event_title": event_title,
    "show_tagline": show_tagline,
    "event_tagline": event_tagline,

    "show_sponsors": show_sponsors,
    "show_organised_by": show_organised_by,
    "organised_by_label": organised_by_label,
    "max_sponsors": int(max_sponsors),

    "circle_d_mm": float(circle_d_mm),
    "circle_y_offset_mm": float(circle_y_offset_mm),

    "show_role_band": show_role_band,
    "role_source": role_source,
    "role_fixed": role_fixed,
    "role_col": role_col,

    "back_show_event_name": back_show_event_name,
    "back_event_name": back_event_name,
    "back_sections": back_sections,
    "back_show_hashtag": back_show_hashtag,
    "back_hashtag": back_hashtag,
}

design = BadgeDesign(
    bg_black=bg_black,
    band_color=band_color,
    accent=accent,
    circle_d_mm=float(circle_d_mm),
    circle_y_offset_mm=float(circle_y_offset_mm),
)

if st.button("Générer PDF"):
    try:
        pdf_bytes = generate_pdf(df=df, cfg=cfg, design=design, sponsor_imgs=sponsor_imgs)

        st.success("PDF généré.")
        st.download_button("Télécharger le PDF", data=pdf_bytes, file_name="badges_A4_recto_verso.pdf", mime="application/pdf")

        # affichage intégré (si downloads bloqués)
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        components.html(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="900"></iframe>', height=920)

    except Exception as e:
        st.error(f"Erreur: {e}")
