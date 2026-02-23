import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageOps

import requests
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "Badge generator — Institut Imagine (Recto/Verso)"
IMAGINE_ROSE = "#C4007A"
BLACK = "#0B1220"  # un noir un peu plus "print"
WHITE = "#FFFFFF"
BAND_DEFAULT = "#C9C4A6"  # proche de ton exemple

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"

CENTER_IMAGE_PATH = ASSETS / "center_image.jpg"  # <-- obligatoire (ton image fixe)
# (si tu préfères PNG, renomme et change ici)

FONT_REG = ASSETS / "Montserrat-Regular.ttf"
FONT_BOLD = ASSETS / "Montserrat-Bold.ttf"

# sources publiques (ttf) pour auto-download si besoin
FONT_REG_URL = "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Regular.ttf"
FONT_BOLD_URL = "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Bold.ttf"


# ============================================================
# FONTS
# ============================================================
def assets_dir_ok() -> bool:
    return ASSETS.exists() and ASSETS.is_dir()


def try_download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def ensure_fonts() -> Dict[str, str]:
    """
    - Si assets/ est un dossier : tente de télécharger Montserrat si absent.
    - Enregistre les polices dans reportlab.
    - Fallback Helvetica si ça rate.
    """
    fonts = {"regular": "Helvetica", "bold": "Helvetica-Bold"}

    if assets_dir_ok():
        if not FONT_REG.exists():
            try_download(FONT_REG_URL, FONT_REG)
        if not FONT_BOLD.exists():
            try_download(FONT_BOLD_URL, FONT_BOLD)

    try:
        if FONT_REG.exists() and FONT_BOLD.exists():
            pdfmetrics.registerFont(TTFont("Montserrat", str(FONT_REG)))
            pdfmetrics.registerFont(TTFont("Montserrat-Bold", str(FONT_BOLD)))
            fonts["regular"] = "Montserrat"
            fonts["bold"] = "Montserrat-Bold"
    except Exception:
        pass

    return fonts


# ============================================================
# IMAGE HELPERS
# ============================================================
def pil_to_reader(img: Image.Image) -> ImageReader:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return ImageReader(bio)


def remove_near_black_to_transparent(img: Image.Image, threshold: int = 18) -> Image.Image:
    """
    Convertit les pixels quasi noirs en transparents.
    Ça règle ton “bout sur fond noir” pour l'image microscope.
    """
    img = ImageOps.exif_transpose(img).convert("RGBA")
    px = img.getdata()
    new_px = []
    for r, g, b, a in px:
        if r < threshold and g < threshold and b < threshold:
            new_px.append((r, g, b, 0))
        else:
            new_px.append((r, g, b, a))
    img.putdata(new_px)
    return img


def make_circle_image(
    img: Image.Image,
    diameter_px: int,
    remove_black_bg: bool = True,
    black_threshold: int = 18,
) -> Image.Image:
    """
    - garde l'image entière (pas de crop “au hasard”)
    - scale to fit dans le cercle
    - fond noir -> transparent (optionnel)
    """
    img = ImageOps.exif_transpose(img).convert("RGBA")

    if remove_black_bg:
        img = remove_near_black_to_transparent(img, threshold=black_threshold)

    # resize en conservant le ratio, pour rentrer dans un carré diamètre x diamètre
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGBA", (diameter_px, diameter_px), (0, 0, 0, 0))

    scale = min(diameter_px / w, diameter_px / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)

    # canvas transparent + centrage
    canvas_img = Image.new("RGBA", (diameter_px, diameter_px), (0, 0, 0, 0))
    ox = (diameter_px - nw) // 2
    oy = (diameter_px - nh) // 2
    canvas_img.paste(img, (ox, oy), img)

    # masque cercle
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


# ============================================================
# TEXT HELPERS
# ============================================================
def fit_font_size(c: canvas.Canvas, text: str, font: str, max_w: float, start: int, min_size: int = 7) -> int:
    size = start
    while size > min_size and c.stringWidth(text, font, size) > max_w:
        size -= 1
    return max(size, min_size)


def split_lines_soft(text: str) -> List[str]:
    """
    Multi-line à partir de \n (simple et fiable).
    """
    return [t.strip() for t in (text or "").split("\n") if t.strip()]


# ============================================================
# DESIGN
# ============================================================
@dataclass
class BadgeDesign:
    badge_w_mm: float = 148   # A5
    badge_h_mm: float = 210   # A5
    band_ratio: float = 0.28

    # couleurs
    bg_black: str = BLACK
    band_color: str = BAND_DEFAULT
    accent: str = IMAGINE_ROSE

    # cercle
    circle_diameter_mm: float = 62
    circle_y_offset_mm: float = -8

    # marges
    margin_mm: float = 10

    # tailles
    date_big: int = 22
    date_small: int = 12
    title_size: int = 20
    tagline_size: int = 10
    edition_size: int = 9
    organised_by_size: int = 7
    role_size: int = 15


def draw_date_block(c: canvas.Canvas, fonts: Dict[str, str], x: float, y: float, d: BadgeDesign, cfg: dict):
    # top-left
    pad = 6 * mm
    bx = x + pad
    by = y + (d.badge_h_mm * mm) - pad - 24 * mm

    c.setFillColor(HexColor(WHITE))
    c.setFont(fonts["bold"], d.date_big)
    c.drawString(bx, by + 10 * mm, cfg["date_l1"])

    c.setFont(fonts["regular"], d.date_small)
    c.drawString(bx, by + 4 * mm, cfg["date_l2"])
    c.drawString(bx, by - 2 * mm, cfg["date_l3"])

    # accent underline
    c.setFillColor(HexColor(d.accent))
    c.rect(bx, by + 2 * mm, 10 * mm, 1.2 * mm, stroke=0, fill=1)


def draw_front_badge(
    c: canvas.Canvas,
    fonts: Dict[str, str],
    design: BadgeDesign,
    x: float,
    y: float,
    cfg: dict,
    person: dict,
    center_img: Image.Image,
    sponsor_imgs: List[Image.Image],
):
    bw = design.badge_w_mm * mm
    bh = design.badge_h_mm * mm
    band_h = bh * design.band_ratio
    top_h = bh - band_h
    margin = design.margin_mm * mm

    # background
    c.setFillColor(HexColor(design.bg_black))
    c.rect(x, y + band_h, bw, top_h, stroke=0, fill=1)
    c.setFillColor(HexColor(design.band_color))
    c.rect(x, y, bw, band_h, stroke=0, fill=1)

    # date
    if cfg["show_date"]:
        draw_date_block(c, fonts, x, y, design, cfg)

    # circle image (fixed)
    circle_d = design.circle_diameter_mm * mm
    cx = x + bw / 2 - circle_d / 2
    cy = y + band_h + (top_h / 2) - (circle_d / 2) + (design.circle_y_offset_mm * mm)

    # prepare circle (small processing)
    dpx = int(design.circle_diameter_mm * 12)  # ~12 px/mm
    circ = make_circle_image(
        center_img,
        diameter_px=dpx,
        remove_black_bg=cfg["remove_black_bg"],
        black_threshold=cfg["black_threshold"],
    )
    c.drawImage(pil_to_reader(circ), cx, cy, width=circle_d, height=circle_d, mask="auto")

    # title overlay (on circle)
    if cfg["show_title"]:
        c.setFillColor(HexColor(WHITE))

        if cfg["show_edition"] and cfg["edition_text"].strip():
            c.setFont(fonts["regular"], design.edition_size)
            c.drawRightString(cx + circle_d - 2 * mm, cy + circle_d - 8 * mm, cfg["edition_text"].strip())

        # title
        title_lines = split_lines_soft(cfg["event_title"])
        c.setFont(fonts["bold"], design.title_size)
        line_h = (design.title_size + 3) * 0.9
        total_h = line_h * len(title_lines)
        ty = cy + circle_d / 2 + total_h / 2 - line_h

        for line in title_lines:
            c.drawCentredString(x + bw / 2, ty, line)
            ty -= line_h

        # tagline
        if cfg["show_tagline"] and cfg["event_tagline"].strip():
            c.setFillColor(HexColor(design.accent))
            c.setFont(fonts["bold"], design.tagline_size)
            c.drawCentredString(x + bw / 2, cy + 10 * mm, cfg["event_tagline"].strip())

    # sponsors row
    if cfg["show_sponsors"] and sponsor_imgs:
        # label
        if cfg["show_organised_by"] and cfg["organised_by_label"].strip():
            c.setFillColor(HexColor(WHITE))
            c.setFont(fonts["regular"], design.organised_by_size)
            c.drawString(x + margin, y + band_h + 20 * mm, cfg["organised_by_label"].strip())

        logos = sponsor_imgs[: cfg["max_sponsors"]]
        n = len(logos)
        gap = 2.5 * mm
        box_h = 12 * mm
        avail_w = bw - 2 * margin - gap * (n - 1)
        box_w = avail_w / n

        box_y = y + band_h + 6 * mm
        for i, lg in enumerate(logos):
            bx = x + margin + i * (box_w + gap)
            c.setFillColor(HexColor(WHITE))
            c.rect(bx, box_y, box_w, box_h, stroke=0, fill=1)
            draw_image_fit(c, lg, bx + 1.2 * mm, box_y + 1.2 * mm, box_w - 2.4 * mm, box_h - 2.4 * mm)

    # identity on front (OFF by default)
    if cfg["show_identity_front"]:
        first = str(person.get(cfg["first_col"], "")).strip()
        last = str(person.get(cfg["last_col"], "")).strip()
        full = (first + " " + last).strip()

        org = str(person.get(cfg["org_col"], "")).strip()
        title = str(person.get(cfg["title_col"], "")).strip()

        max_w = bw - 2 * margin

        c.setFillColor(HexColor(WHITE))
        fsz = fit_font_size(c, full, fonts["bold"], max_w, 26)
        c.setFont(fonts["bold"], fsz)
        c.drawCentredString(x + bw / 2, y + band_h + 34 * mm, full)

        if org:
            c.setFont(fonts["regular"], 11)
            c.drawCentredString(x + bw / 2, y + band_h + 28 * mm, org[:70])

        if cfg["show_title_front"] and title:
            c.setFont(fonts["regular"], 10)
            c.drawCentredString(x + bw / 2, y + band_h + 22 * mm, title[:80])

    # role in bottom band (like INTERVENANT)
    if cfg["show_role_band"]:
        role = str(person.get(cfg["role_col"], "")).strip().upper()
        if role:
            c.setFillColor(HexColor("#111111"))
            c.setFont(fonts["bold"], design.role_size)
            c.drawCentredString(x + bw / 2, y + band_h / 2 - 3 * mm, role)


def draw_back_badge(
    c: canvas.Canvas,
    fonts: Dict[str, str],
    design: BadgeDesign,
    x: float,
    y: float,
    cfg: dict,
):
    bw = design.badge_w_mm * mm
    bh = design.badge_h_mm * mm
    margin = design.margin_mm * mm

    # background full black
    c.setFillColor(HexColor(design.bg_black))
    c.rect(x, y, bw, bh, stroke=0, fill=1)

    # small header (optional)
    if cfg["back_show_event_name"]:
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], 12)
        c.drawCentredString(x + bw / 2, y + bh - 14 * mm, cfg["back_event_name"][:80])

    # sections
    cursor_y = y + bh - (28 * mm if cfg["back_show_event_name"] else 18 * mm)

    for sec in cfg["back_sections"]:
        if not sec["show"]:
            continue

        title = (sec["title"] or "").strip().upper()
        lines = split_lines_soft(sec["lines"])

        if not title and not lines:
            continue

        # title
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], 14)
        c.drawCentredString(x + bw / 2, cursor_y, title)
        cursor_y -= 8 * mm

        # lines
        c.setFont(fonts["regular"], 11)
        for ln in lines:
            c.drawCentredString(x + bw / 2, cursor_y, ln[:90])
            cursor_y -= 6 * mm

        cursor_y -= 6 * mm  # space between sections

        if cursor_y < y + 30 * mm:
            break

    # hashtag bottom
    if cfg["back_show_hashtag"] and cfg["back_hashtag"].strip():
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], 14)
        c.drawCentredString(x + bw / 2, y + 18 * mm, cfg["back_hashtag"].strip())


def draw_cut_marks(c: canvas.Canvas, x: float, y: float, w: float, h: float):
    c.setStrokeColor(HexColor("#9CA3AF"))
    c.setLineWidth(0.3)
    L = 5 * mm
    c.line(x, y, x + L, y); c.line(x, y, x, y + L)
    c.line(x + w, y, x + w - L, y); c.line(x + w, y, x + w, y + L)
    c.line(x, y + h, x + L, y + h); c.line(x, y + h, x, y + h - L)
    c.line(x + w, y + h, x + w - L, y + h); c.line(x + w, y + h, x + w, y + h - L)


# ============================================================
# PDF GENERATION
# ============================================================
def get_sheet_layout(sheet: str, design: BadgeDesign, cfg: dict) -> Tuple[Tuple[float, float], Tuple[float, float], List[Tuple[float, float]]]:
    """
    Retourne: (page_w,page_h), (badge_w,badge_h), positions 2x2
    """
    if sheet == "A3 (4×A5)":
        page_w, page_h = A3
        badge_w = design.badge_w_mm * mm
        badge_h = design.badge_h_mm * mm
    else:
        # option si besoin : A4 avec badges A6 (A5/2)
        page_w, page_h = A4
        badge_w = (design.badge_w_mm * mm) / 2
        badge_h = (design.badge_h_mm * mm) / 2

    gap = cfg["gap_mm"] * mm
    margin = cfg["page_margin_mm"] * mm

    positions = [
        (margin, page_h - margin - badge_h),
        (margin + badge_w + gap, page_h - margin - badge_h),
        (margin, page_h - margin - 2 * badge_h - gap),
        (margin + badge_w + gap, page_h - margin - 2 * badge_h - gap),
    ]
    return (page_w, page_h), (badge_w, badge_h), positions


def load_center_image_or_fail() -> Optional[Image.Image]:
    if CENTER_IMAGE_PATH.exists():
        return Image.open(CENTER_IMAGE_PATH)
    return None


def generate_pdf(
    df: pd.DataFrame,
    design: BadgeDesign,
    cfg: dict,
    sponsor_imgs: List[Image.Image],
) -> bytes:
    fonts = ensure_fonts()

    center_img = load_center_image_or_fail()
    if center_img is None:
        raise RuntimeError(f"Image centrale manquante: {CENTER_IMAGE_PATH.as_posix()}")

    sheet = cfg["sheet"]
    mode = cfg["mode"]

    (page_w, page_h), (badge_w, badge_h), positions = get_sheet_layout(sheet, design, cfg)

    # si mode A4 (A6), on adapte un design réduit
    local_design = design
    if sheet != "A3 (4×A5)":
        local_design = BadgeDesign(
            badge_w_mm=design.badge_w_mm / 2,
            badge_h_mm=design.badge_h_mm / 2,
            band_ratio=design.band_ratio,
            bg_black=design.bg_black,
            band_color=design.band_color,
            accent=design.accent,
            circle_diameter_mm=design.circle_diameter_mm * 0.7,
            circle_y_offset_mm=design.circle_y_offset_mm,
            margin_mm=design.margin_mm / 2,
            date_big=max(12, int(design.date_big * 0.7)),
            date_small=max(8, int(design.date_small * 0.7)),
            title_size=max(12, int(design.title_size * 0.7)),
            tagline_size=max(7, int(design.tagline_size * 0.7)),
            edition_size=max(7, int(design.edition_size * 0.8)),
            organised_by_size=max(6, int(design.organised_by_size * 0.9)),
            role_size=max(10, int(design.role_size * 0.75)),
        )

    rows = df.to_dict(orient="records")
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=(page_w, page_h))

    def draw_front_page(start_idx: int):
        idx = start_idx
        for p in range(4):
            if idx >= len(rows):
                break
            px, py = positions[p]
            draw_front_badge(
                c=c,
                fonts=fonts,
                design=local_design,
                x=px,
                y=py,
                cfg=cfg,
                person=rows[idx],
                center_img=center_img,
                sponsor_imgs=sponsor_imgs,
            )
            if cfg["cut_marks"]:
                draw_cut_marks(c, px, py, local_design.badge_w_mm * mm, local_design.badge_h_mm * mm)
            idx += 1

    def draw_back_page(start_idx: int):
        idx = start_idx
        for p in range(4):
            if idx >= len(rows):
                break
            px, py = positions[p]

            if cfg["mirror_backs"]:
                # miroir horizontal dans la zone badge (utile selon imprimeur/recto-verso)
                c.saveState()
                c.translate(px + (local_design.badge_w_mm * mm), py)
                c.scale(-1, 1)
                draw_back_badge(c, fonts, local_design, 0, 0, cfg)
                if cfg["cut_marks"]:
                    draw_cut_marks(c, 0, 0, local_design.badge_w_mm * mm, local_design.badge_h_mm * mm)
                c.restoreState()
            else:
                draw_back_badge(c, fonts, local_design, px, py, cfg)
                if cfg["cut_marks"]:
                    draw_cut_marks(c, px, py, local_design.badge_w_mm * mm, local_design.badge_h_mm * mm)
            idx += 1

    i = 0
    while i < len(rows):
        if mode == "Recto seul":
            draw_front_page(i)
            c.showPage()
        elif mode == "Verso seul":
            draw_back_page(i)
            c.showPage()
        elif mode == "Recto puis verso (pages séparées)":
            draw_front_page(i)
            c.showPage()
            draw_back_page(i)
            c.showPage()
        else:  # "Recto/Verso intercalés (1 page recto, 1 page verso)"
            draw_front_page(i)
            c.showPage()
            draw_back_page(i)
            c.showPage()

        i += 4

    c.save()
    return out.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# assets sanity
if ASSETS.exists() and not ASSETS.is_dir():
    st.error("⚠️ Dans ton repo, `assets` est un FICHIER. Il doit être un DOSSIER `assets/`.\n"
             "Supprime le fichier `assets` et crée `assets/.gitkeep`, puis upload `assets/center_image.jpg`.")
    st.stop()

if not ASSETS.exists():
    st.warning("Dossier `assets/` introuvable. Crée-le dans ton repo (assets/.gitkeep), puis ajoute `center_image.jpg`.")

center_ok = CENTER_IMAGE_PATH.exists()
if not center_ok:
    st.error(f"Image centrale fixe manquante : `{CENTER_IMAGE_PATH.as_posix()}`.\n"
             "Upload ton image microscope dans `assets/center_image.jpg` (GitHub → Add file → Upload).")
    st.stop()
else:
    with st.expander("✅ Image centrale fixe (lecture depuis assets/center_image.jpg)", expanded=False):
        try:
            st.image(Image.open(CENTER_IMAGE_PATH), caption="Image centrale utilisée pour TOUS les badges", use_container_width=True)
        except Exception:
            st.warning("Impossible d'afficher l'image, mais elle semble présente.")

# Data + settings columns
left, right = st.columns([1.1, 1])

with left:
    st.subheader("1) Données participants")
    file = st.file_uploader("CSV ou Excel", type=["csv", "xlsx"])

    df = None
    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.write("Aperçu :")
        st.dataframe(df.head(25), use_container_width=True)

with right:
    st.subheader("2) Logos sponsors")
    sponsors_up = st.file_uploader(
        "Logos sponsors (PNG conseillé, fond transparent) — multiple",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    sponsor_imgs: List[Image.Image] = []
    if sponsors_up:
        for f in sponsors_up:
            sponsor_imgs.append(Image.open(f))

    st.subheader("3) Couleurs (charte Imagine par défaut)")
    bg_black = st.color_picker("Fond haut", BLACK)
    band_color = st.color_picker("Bande basse", BAND_DEFAULT)
    accent = st.color_picker("Accent Imagine", IMAGINE_ROSE)

st.divider()

st.subheader("4) Recto — blocs & contenus")

c1, c2, c3 = st.columns(3)

with c1:
    show_date = st.checkbox("Afficher la date", True)
    date_l1 = st.text_input("Date ligne 1", "18 19")
    date_l2 = st.text_input("Date ligne 2", "février")
    date_l3 = st.text_input("Date ligne 3", "2026")

    st.markdown("**Image centrale (fixe)**")
    remove_black_bg = st.checkbox("Retirer le fond noir de l'image", True)
    black_threshold = st.slider("Seuil 'noir' (transparence)", 5, 60, 18)

with c2:
    show_title = st.checkbox("Afficher titre événement", True)
    show_edition = st.checkbox("Afficher édition", True)
    edition_text = st.text_input("Texte édition", "3e édition")
    event_title = st.text_area("Titre (1 ligne = 1 ligne)", "PARIS\nSACLAY\nSUMMIT", height=90)

    show_tagline = st.checkbox("Afficher tagline", True)
    event_tagline = st.text_input("Tagline", "CHOOSE SCIENCE")

with c3:
    show_sponsors = st.checkbox("Afficher la rangée de logos", True)
    show_organised_by = st.checkbox("Afficher “Organisé par”", True)
    organised_by_label = st.text_input("Libellé", "Organisé par")
    max_sponsors = st.slider("Nombre max de logos affichés", 1, 8, 4)

    cut_marks = st.checkbox("Traits de coupe", True)

st.divider()
st.subheader("5) Verso — sections (cases à cocher)")

back_show = st.checkbox("Activer le verso", True)
back_show_event_name = st.checkbox("Verso : afficher le nom de l'évènement en haut", False)
back_event_name = st.text_input("Verso : nom évènement", "PARIS SACLAY SUMMIT")

sections_cfg = []
if back_show:
    sec_cols = st.columns(2)
    default_sections = [
        ("ORGANISATION", "Anne-Sophie CLOAREC\n+33 7 65 71 53 99"),
        ("LOGISTIQUE", "Anne TISLER\n+33 7 81 18 40 94\nAurélie MESTELAN\n+33 6 38 22 26 36"),
        ("PARTENAIRES", "Justine COUTEILLE\n+33 6 75 58 75 32\nCharlotte M’NASRI\n+33 6 74 47 11 99"),
        ("ÉDITORIAL", "Romain GONZALEZ\n+33 7 69 26 95 89"),
    ]

    for i, (t, content) in enumerate(default_sections):
        col = sec_cols[i % 2]
        with col:
            show = st.checkbox(f"Afficher section {i+1}", True, key=f"sec_show_{i}")
            title = st.text_input(f"Titre {i+1}", t, key=f"sec_title_{i}")
            lines = st.text_area(f"Lignes {i+1} (1 info par ligne)", content, height=110, key=f"sec_lines_{i}")
            sections_cfg.append({"show": show, "title": title, "lines": lines})

    back_show_hashtag = st.checkbox("Verso : afficher hashtag", True)
    back_hashtag = st.text_input("Hashtag", "#ParisSaclaySummit")
else:
    back_show_event_name = False
    back_event_name = ""
    back_show_hashtag = False
    back_hashtag = ""
    sections_cfg = []

st.divider()

st.subheader("6) Données variables (CSV/Excel) — mapping + affichages")
if df is None:
    st.info("Charge un CSV/Excel pour activer la génération.")
    st.stop()

cols = list(df.columns)
m1, m2 = st.columns(2)

with m1:
    first_col = st.selectbox("Colonne prénom", cols, index=cols.index("prenom") if "prenom" in cols else 0)
    last_col = st.selectbox("Colonne nom", cols, index=cols.index("nom") if "nom" in cols else 0)
    org_col = st.selectbox("Colonne organisation", cols, index=cols.index("organisation") if "organisation" in cols else 0)

with m2:
    title_col = st.selectbox("Colonne fonction (optionnel)", cols, index=cols.index("fonction") if "fonction" in cols else 0)
    role_col = st.selectbox("Colonne catégorie / rôle (ex: INTERVENANT)", cols, index=cols.index("categorie") if "categorie" in cols else 0)

show_role_band = st.checkbox("Recto : afficher la catégorie dans la bande basse", True)

with st.expander("Option (désactivée par défaut) : identité sur le recto", expanded=False):
    show_identity_front = st.checkbox("Afficher NOM/ORG sur le recto", False)
    show_title_front = st.checkbox("Recto : afficher la fonction", False)

st.divider()
st.subheader("7) Mise en page & export PDF")

p1, p2, p3, p4 = st.columns(4)
with p1:
    sheet = st.selectbox("Format feuille", ["A3 (4×A5)", "A4 (4×A6)"], index=0)
with p2:
    page_margin_mm = st.number_input("Marge page (mm)", min_value=0, max_value=25, value=6)
with p3:
    gap_mm = st.number_input("Espace entre badges (mm)", min_value=0, max_value=20, value=6)
with p4:
    mode = st.selectbox(
        "Mode export",
        ["Recto seul", "Verso seul", "Recto puis verso (pages séparées)", "Recto/Verso intercalés (1 page recto, 1 page verso)"],
        index=2,
    )

mirror_backs = st.checkbox("Miroir horizontal des versos (utile selon imprimante recto/verso)", False)

circle_diameter_mm = st.slider("Diamètre du cercle (mm)", 45, 85, 62)
circle_y_offset_mm = st.slider("Décalage vertical du cercle (mm)", -30, 30, -8)

cfg = {
    "sheet": sheet,
    "page_margin_mm": float(page_margin_mm),
    "gap_mm": float(gap_mm),
    "mode": mode,
    "cut_marks": cut_marks,
    "mirror_backs": mirror_backs,

    # front blocks
    "show_date": show_date,
    "date_l1": date_l1,
    "date_l2": date_l2,
    "date_l3": date_l3,

    "remove_black_bg": remove_black_bg,
    "black_threshold": int(black_threshold),

    "show_title": show_title,
    "show_edition": show_edition,
    "edition_text": edition_text,
    "event_title": event_title,
    "show_tagline": show_tagline,
    "event_tagline": event_tagline,

    "show_sponsors": show_sponsors,
    "show_organised_by": show_organised_by,
    "organised_by_label": organised_by_label,
    "max_sponsors": int(max_sponsors),

    "show_role_band": show_role_band,

    "show_identity_front": show_identity_front,
    "show_title_front": show_title_front,

    # mapping
    "first_col": first_col,
    "last_col": last_col,
    "org_col": org_col,
    "title_col": title_col,
    "role_col": role_col,

    # back
    "back_sections": sections_cfg,
    "back_show_event_name": back_show_event_name if back_show else False,
    "back_event_name": back_event_name if back_show else "",
    "back_show_hashtag": back_show_hashtag if back_show else False,
    "back_hashtag": back_hashtag if back_show else "",
}

design = BadgeDesign(
    bg_black=bg_black,
    band_color=band_color,
    accent=accent,
    circle_diameter_mm=float(circle_diameter_mm),
    circle_y_offset_mm=float(circle_y_offset_mm),
)

if st.button("Générer le PDF"):
    try:
        pdf_bytes = generate_pdf(df=df, design=design, cfg=cfg, sponsor_imgs=sponsor_imgs)

        st.success("PDF généré.")

        # download (si autorisé)
        st.download_button(
            "Télécharger le PDF",
            data=pdf_bytes,
            file_name="badges_recto_verso.pdf",
            mime="application/pdf",
        )

        # affichage inline (utile si “download” bloqué)
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        components.html(
            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="900"></iframe>',
            height=920,
        )

    except Exception as e:
        st.error(f"Erreur : {e}")
