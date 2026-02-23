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
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================
# CONFIG / CHARTE IMAGINE
# =========================
APP_TITLE = "Badge generator — Imagine (A4, 4×A6, Recto/Verso)"
IMAGINE_ROSE = "#C4007A"
BLACK = "#0B1220"
WHITE = "#FFFFFF"
BAND_DEFAULT = "#C9C4A6"

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
CENTER_IMAGE_PATH = ASSETS / "center_image.jpg"  # image microscope fixe

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

    # FIT (pas de crop)
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGBA", (diameter_px, diameter_px), (0, 0, 0, 0))

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


def split_lines(text: str) -> List[str]:
    return [t.strip() for t in (text or "").split("\n") if t.strip()]


def safe_str(v) -> str:
    return "" if v is None else str(v).strip()


# =========================
# DESIGN
# =========================
@dataclass
class Design:
    bg_black: str = BLACK
    band_color: str = BAND_DEFAULT
    accent: str = IMAGINE_ROSE

    band_ratio: float = 0.28
    inner_margin_mm: float = 6

    # base font sizes (A6 print-safe)
    date_big: int = 14
    date_small: int = 9
    edition_size: int = 8

    event_title_size: int = 18
    baseline_size: int = 10

    organised_by_size: int = 6

    band_text_size: int = 12

    # image
    circle_d_mm: float = 58
    circle_y_offset_mm: float = -2


# =========================
# A4 -> 4 badges (A6 ratio) print-safe
# =========================
def compute_a4_4up_layout(page_margin_mm: float, gap_mm: float) -> Tuple[Tuple[float, float], Tuple[float, float], List[Tuple[float, float]]]:
    """
    A4 portrait.
    On calcule une taille badge basée sur le ratio A6 (105×148),
    en garantissant que ça rentre avec marge + gap.
    """
    page_w, page_h = A4
    m = page_margin_mm * mm
    g = gap_mm * mm

    # contraintes : 2 colonnes, 2 lignes
    max_w_per_col = (page_w - 2 * m - g) / 2
    max_h_per_row = (page_h - 2 * m - g) / 2

    # ratio A6
    ratio_w_h = 105 / 148

    # derive badge size from both constraints
    w_from_width = max_w_per_col
    h_from_width = w_from_width / ratio_w_h

    h_from_height = max_h_per_row
    w_from_height = h_from_height * ratio_w_h

    # choose the limiting size
    badge_w = min(w_from_width, w_from_height)
    badge_h = badge_w / ratio_w_h

    # positions 2x2
    x1 = m
    x2 = m + badge_w + g
    y1 = page_h - m - badge_h
    y2 = page_h - m - 2 * badge_h - g

    positions = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    return (page_w, page_h), (badge_w, badge_h), positions


def draw_cut_marks(c: canvas.Canvas, x: float, y: float, w: float, h: float):
    c.setStrokeColor(HexColor("#9CA3AF"))
    c.setLineWidth(0.3)
    L = 4 * mm
    c.line(x, y, x + L, y); c.line(x, y, x, y + L)
    c.line(x + w, y, x + w - L, y); c.line(x + w, y, x + w, y + L)
    c.line(x, y + h, x + L, y + h); c.line(x, y + h, x, y + h - L)
    c.line(x + w, y + h, x + w - L, y + h); c.line(x + w, y + h, x + w, y + h - L)


# =========================
# DRAW RECTO
# =========================
def draw_recto(
    c: canvas.Canvas,
    fonts: Dict[str, str],
    d: Design,
    x: float,
    y: float,
    bw: float,
    bh: float,
    cfg: dict,
    person: dict,
    center_img: Image.Image,
    sponsor_imgs: List[Image.Image],
):
    band_h = bh * d.band_ratio
    top_h = bh - band_h
    pad = 4 * mm
    inner = d.inner_margin_mm * mm

    # backgrounds
    c.setFillColor(HexColor(d.bg_black))
    c.rect(x, y + band_h, bw, top_h, stroke=0, fill=1)
    c.setFillColor(HexColor(d.band_color))
    c.rect(x, y, bw, band_h, stroke=0, fill=1)

    # date top-left (bien en haut)
    if cfg["show_date"]:
        bx = x + pad
        by = y + bh - pad - 16 * mm

        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], d.date_big)
        c.drawString(bx, by + 7 * mm, cfg["date_l1"])

        c.setFont(fonts["regular"], d.date_small)
        c.drawString(bx, by + 2.5 * mm, cfg["date_l2"])
        c.drawString(bx, by - 2.0 * mm, cfg["date_l3"])

        c.setFillColor(HexColor(d.accent))
        c.rect(bx, by + 0.7 * mm, 10 * mm, 1.1 * mm, stroke=0, fill=1)

    # edition top-right (bien en haut à droite)
    if cfg["show_edition"] and cfg["edition_text"].strip():
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["regular"], d.edition_size)
        c.drawRightString(x + bw - pad, y + bh - pad - 3.5 * mm, cfg["edition_text"].strip())

    # title + baseline (plus gros, lisible, AU-DESSUS de l'image)
    ccx = x + bw / 2
    title_lines = split_lines(cfg["event_title"])[:2]
    baseline = cfg["baseline"].strip()

    # zone titre : entre le haut et l'image
    title_top = y + band_h + top_h - 22 * mm
    title_max_w = bw - 2 * inner

    if cfg["show_event_title"] and title_lines:
        # ajuste taille sur ligne la plus longue
        longest = max(title_lines, key=len)
        tsize = fit_font_size(c, longest, fonts["bold"], title_max_w, cfg["event_title_size"], min_size=10)

        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], tsize)
        lh = (tsize + 3) * 0.85
        ty = title_top

        for ln in title_lines:
            c.drawCentredString(ccx, ty, ln)
            ty -= lh

        if cfg["show_baseline"] and baseline:
            c.setFillColor(HexColor(d.accent))
            bsize = fit_font_size(c, baseline, fonts["bold"], title_max_w, cfg["baseline_size"], min_size=7)
            c.setFont(fonts["bold"], bsize)
            c.drawCentredString(ccx, ty - 2 * mm, baseline)

    # microscope circle (plus gros, centré)
    circle_d = cfg["circle_d_mm"] * mm
    ccy = y + band_h + top_h * 0.50 + (cfg["circle_y_offset_mm"] * mm)

    dpx = int(cfg["circle_d_mm"] * 14)  # un peu plus net
    circ = make_circle_image(center_img, dpx, cfg["remove_black_bg"], cfg["black_threshold"])
    c.drawImage(
        pil_to_reader(circ),
        ccx - circle_d / 2,
        ccy - circle_d / 2,
        width=circle_d,
        height=circle_d,
        mask="auto",
    )

    # sponsors (optionnel)
    if cfg["show_sponsors"] and sponsor_imgs:
        if cfg["show_organised_by"] and cfg["organised_by_label"].strip():
            c.setFillColor(HexColor(WHITE))
            c.setFont(fonts["regular"], d.organised_by_size)
            c.drawString(x + inner, y + band_h + 18 * mm, cfg["organised_by_label"].strip())

        logos = sponsor_imgs[: cfg["max_sponsors"]]
        n = len(logos)
        gap = 2.0 * mm
        box_h = 10 * mm
        avail_w = bw - 2 * inner - gap * (n - 1)
        box_w = avail_w / n
        box_y = y + band_h + 6 * mm

        for i, lg in enumerate(logos):
            bx = x + inner + i * (box_w + gap)
            c.setFillColor(HexColor(WHITE))
            c.rect(bx, box_y, box_w, box_h, stroke=0, fill=1)
            draw_image_fit(c, lg, bx + 1.0 * mm, box_y + 1.0 * mm, box_w - 2.0 * mm, box_h - 2.0 * mm)

    # band text (ROLE ou NOM COMPLET)
    if cfg["band_show"]:
        if cfg["band_mode"] == "Catégorie (colonne)":
            txt = safe_str(person.get(cfg["role_col"], "")).upper()
        elif cfg["band_mode"] == "Nom complet":
            txt = f"{safe_str(person.get(cfg['first_col'], ''))} {safe_str(person.get(cfg['last_col'], ''))}".strip().upper()
        else:
            txt = cfg["band_custom"].strip().upper()

        if txt:
            max_w = bw - 2 * inner
            c.setFillColor(HexColor("#111111"))
            size = fit_font_size(c, txt, fonts["bold"], max_w, cfg["band_text_size"], min_size=7)
            c.setFont(fonts["bold"], size)
            c.drawCentredString(x + bw / 2, y + band_h / 2 - 2 * mm, txt)


# =========================
# DRAW VERSO (toujours visible)
# =========================
def draw_verso(c: canvas.Canvas, fonts: Dict[str, str], d: Design, x: float, y: float, bw: float, bh: float, cfg: dict):
    c.setFillColor(HexColor(d.bg_black))
    c.rect(x, y, bw, bh, stroke=0, fill=1)

    inner = d.inner_margin_mm * mm
    top = y + bh - inner
    bottom = y + inner + (14 * mm if cfg["back_show_hashtag"] else 0)

    # sections actives
    sections = []
    for s in cfg["back_sections"]:
        if s["show"]:
            title = (s["title"] or "").strip().upper()
            content = split_lines(s["lines"])
            if title or content:
                sections.append((title, content))

    # header event name (optionnel)
    cursor = top
    if cfg["back_show_event_name"] and cfg["back_event_name"].strip():
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], 11)
        c.drawCentredString(x + bw / 2, cursor - 4 * mm, cfg["back_event_name"].strip()[:80])
        cursor -= 14 * mm

    # auto-fit simple : baisse tailles si trop long
    title_size = 12
    line_size = 9
    while True:
        needed = 0
        for t, lns in sections:
            needed += 8 * mm  # titre
            needed += len(lns) * 5.5 * mm
            needed += 6 * mm
        if needed <= (cursor - bottom) or (title_size <= 10 and line_size <= 8):
            break
        title_size -= 1
        line_size -= 1

    for t, lns in sections:
        if cursor < bottom + 10 * mm:
            break
        if t:
            c.setFillColor(HexColor(WHITE))
            c.setFont(fonts["bold"], title_size)
            c.drawCentredString(x + bw / 2, cursor, t)
            cursor -= 8 * mm

        c.setFont(fonts["regular"], line_size)
        for ln in lns:
            c.drawCentredString(x + bw / 2, cursor, ln[:90])
            cursor -= 5.5 * mm

        cursor -= 6 * mm

    if cfg["back_show_hashtag"] and cfg["back_hashtag"].strip():
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], 12)
        c.drawCentredString(x + bw / 2, y + 12 * mm, cfg["back_hashtag"].strip())


# =========================
# PDF GENERATION
# =========================
def generate_pdf(df: pd.DataFrame, cfg: dict, sponsor_imgs: List[Image.Image]) -> bytes:
    fonts, _ = ensure_fonts()

    center_img = Image.open(CENTER_IMAGE_PATH)

    (page_w, page_h), (bw, bh), positions = compute_a4_4up_layout(cfg["page_margin_mm"], cfg["gap_mm"])

    d = Design(
        bg_black=cfg["bg_black"],
        band_color=cfg["band_color"],
        accent=cfg["accent"],
        circle_d_mm=cfg["circle_d_mm"],
        circle_y_offset_mm=cfg["circle_y_offset_mm"],
    )

    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=(page_w, page_h))

    rows = df.to_dict(orient="records")
    step = 4

    def draw_recto_page(start_idx: int):
        idx = start_idx
        for pos in positions:
            if idx >= len(rows):
                break
            px, py = pos
            draw_recto(c, fonts, d, px, py, bw, bh, cfg, rows[idx], center_img, sponsor_imgs)
            if cfg["cut_marks"]:
                draw_cut_marks(c, px, py, bw, bh)
            idx += 1

    def draw_verso_page():
        for px, py in positions:
            draw_verso(c, fonts, d, px, py, bw, bh, cfg)
            if cfg["cut_marks"]:
                draw_cut_marks(c, px, py, bw, bh)

    i = 0
    while i < len(rows):
        if cfg["export_mode"] == "Recto seul":
            draw_recto_page(i)
            c.showPage()
        elif cfg["export_mode"] == "Verso seul":
            draw_verso_page()
            c.showPage()
        else:  # Recto puis verso
            draw_recto_page(i)
            c.showPage()
            draw_verso_page()
            c.showPage()
        i += step

    c.save()
    return out.getvalue()


# =========================
# STREAMLIT UI
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
st.caption(f"Police PDF : **{font_status}**")

with st.expander("Image microscope utilisée (fixe)", expanded=False):
    st.image(Image.open(CENTER_IMAGE_PATH), use_container_width=True)

left, right = st.columns([1.15, 1])

with left:
    st.subheader("1) Participants")
    f = st.file_uploader("CSV ou Excel", type=["csv", "xlsx"])
    df = None
    if f is not None:
        df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
        st.dataframe(df.head(20), use_container_width=True)

with right:
    st.subheader("2) Logos sponsors (optionnel)")
    logos_up = st.file_uploader("Upload logos (PNG conseillé)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    sponsor_imgs = [Image.open(x) for x in logos_up] if logos_up else []

    st.subheader("3) Couleurs")
    bg_black = st.color_picker("Fond haut", BLACK)
    band_color = st.color_picker("Bande basse", BAND_DEFAULT)
    accent = st.color_picker("Accent Imagine", IMAGINE_ROSE)

st.divider()

st.subheader("4) Export A4 (4×A6) — taille print-safe")
export_mode = st.selectbox("Export", ["Recto puis verso", "Recto seul", "Verso seul"], index=0)
page_margin_mm = st.number_input("Marge page (mm) — mets 0 si tu veux maxi", 0, 12, 2)
gap_mm = st.number_input("Espace entre badges (mm)", 0, 12, 2)
cut_marks = st.checkbox("Traits de coupe", True)

st.divider()
st.subheader("5) Recto — contenu & mise en page")

c1, c2, c3 = st.columns(3)
with c1:
    show_date = st.checkbox("Date (haut gauche)", True)
    date_l1 = st.text_input("Date L1", "Mardi 18")
    date_l2 = st.text_input("Date L2", "Février")
    date_l3 = st.text_input("Date L3", "2026")

    show_edition = st.checkbox("Edition (haut droite)", True)
    edition_text = st.text_input("Texte édition", "1ere édition")

with c2:
    show_event_title = st.checkbox("Nom évènement (gros, lisible)", True)
    event_title = st.text_area("Nom évènement (1 ligne = 1 ligne)", "Grand Patrie", height=70)
    event_title_size = st.slider("Taille nom évènement", 12, 28, 18)

    show_baseline = st.checkbox("Baseline", True)
    baseline = st.text_input("Baseline", "L'avenir de demain")
    baseline_size = st.slider("Taille baseline", 7, 16, 10)

with c3:
    circle_d_mm = st.slider("Diamètre image microscope (mm)", 40, 80, 58)
    circle_y_offset_mm = st.slider("Décalage vertical image (mm)", -25, 25, -2)

    remove_black_bg = st.checkbox("Retirer fond noir de l'image", True)
    black_threshold = st.slider("Seuil noir→transparent", 5, 60, 18)

st.divider()
st.subheader("6) Bande basse (catégorie / nom / texte fixe) + Mapping colonnes")

if df is None:
    st.info("Charge un fichier participants pour activer le mapping et la génération.")
    st.stop()

cols = list(df.columns)

m1, m2 = st.columns(2)
with m1:
    first_col = st.selectbox("Colonne prénom", cols, index=cols.index("prenom") if "prenom" in cols else 0)
    last_col = st.selectbox("Colonne nom", cols, index=cols.index("nom") if "nom" in cols else 0)
with m2:
    role_col = st.selectbox("Colonne catégorie (INTERVENANT/...) ", cols, index=cols.index("categorie") if "categorie" in cols else 0)

band_show = st.checkbox("Afficher texte dans la bande basse", True)
band_mode = st.selectbox("Contenu de la bande basse", ["Catégorie (colonne)", "Nom complet", "Texte fixe"], index=0)
band_custom = st.text_input("Texte fixe", "INTERVENANT")
band_text_size = st.slider("Taille texte bande", 9, 18, 12)

st.divider()
st.subheader("7) Sponsors (optionnel)")

show_sponsors = st.checkbox("Afficher logos sponsors", True)
show_organised_by = st.checkbox("Afficher “Organisé par”", True)
organised_by_label = st.text_input("Libellé", "Organisé par")
max_sponsors = st.slider("Max logos", 1, 8, 3)

st.divider()
st.subheader("8) Verso — sections (ça sort exactement comme saisi)")

back_show_event_name = st.checkbox("Verso : afficher nom évènement en haut", False)
back_event_name = st.text_input("Verso : nom évènement", "Grand Patrie")

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
        show = st.checkbox(f"Afficher {t}", True, key=f"sec_show_{i}")
        title = st.text_input(f"Titre {i+1}", t, key=f"sec_title_{i}")
        txt = st.text_area(f"Contenu {i+1}", content, height=110, key=f"sec_txt_{i}")
        back_sections.append({"show": show, "title": title, "lines": txt})

back_show_hashtag = st.checkbox("Verso : afficher hashtag", True)
back_hashtag = st.text_input("Hashtag", "#Demain c'est top")

# Build cfg
cfg = {
    "export_mode": export_mode,
    "page_margin_mm": float(page_margin_mm),
    "gap_mm": float(gap_mm),
    "cut_marks": cut_marks,

    "bg_black": bg_black,
    "band_color": band_color,
    "accent": accent,

    "show_date": show_date,
    "date_l1": date_l1,
    "date_l2": date_l2,
    "date_l3": date_l3,

    "show_edition": show_edition,
    "edition_text": edition_text,

    "show_event_title": show_event_title,
    "event_title": event_title,
    "event_title_size": int(event_title_size),

    "show_baseline": show_baseline,
    "baseline": baseline,
    "baseline_size": int(baseline_size),

    "circle_d_mm": float(circle_d_mm),
    "circle_y_offset_mm": float(circle_y_offset_mm),

    "remove_black_bg": remove_black_bg,
    "black_threshold": int(black_threshold),

    "show_sponsors": show_sponsors,
    "show_organised_by": show_organised_by,
    "organised_by_label": organised_by_label,
    "max_sponsors": int(max_sponsors),

    "band_show": band_show,
    "band_mode": band_mode,
    "band_custom": band_custom,
    "band_text_size": int(band_text_size),

    "first_col": first_col,
    "last_col": last_col,
    "role_col": role_col,

    "back_show_event_name": back_show_event_name,
    "back_event_name": back_event_name,
    "back_sections": back_sections,
    "back_show_hashtag": back_show_hashtag,
    "back_hashtag": back_hashtag,
}

if st.button("Générer PDF"):
    try:
        pdf_bytes = generate_pdf(df=df, cfg=cfg, sponsor_imgs=sponsor_imgs)

        st.success("PDF généré.")
        st.download_button("Télécharger le PDF", data=pdf_bytes, file_name="badges_A4_4xA6_recto_verso.pdf", mime="application/pdf")

        # Affichage inline (utile si download bloqué)
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        components.html(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="900"></iframe>', height=920)

    except Exception as e:
        st.error(f"Erreur : {e}")
