import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageDraw

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================
# CHARTE (defaults Imagine)
# =========================
IMAGINE_ROSE = "#C4007A"
BLACK = "#111827"
WHITE = "#FFFFFF"
BAND_DEFAULT = "#C9C4A6"  # proche de l'exemple, modifiable


def register_fonts():
    # Fallback Helvetica si pas de ttf
    fonts = {"regular": "Helvetica", "bold": "Helvetica-Bold"}
    try:
        pdfmetrics.registerFont(TTFont("Montserrat", "assets/Montserrat-Regular.ttf"))
        pdfmetrics.registerFont(TTFont("Montserrat-Bold", "assets/Montserrat-Bold.ttf"))
        fonts["regular"] = "Montserrat"
        fonts["bold"] = "Montserrat-Bold"
    except Exception:
        pass
    return fonts


def pil_to_reader(img: Image.Image) -> ImageReader:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return ImageReader(bio)


def make_circle_image(img: Image.Image, diameter_px: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")
    # crop carré centré
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((diameter_px, diameter_px), Image.LANCZOS)

    mask = Image.new("L", (diameter_px, diameter_px), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, diameter_px - 1, diameter_px - 1), fill=255)

    out = Image.new("RGBA", (diameter_px, diameter_px), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask=mask)
    return out


def draw_image_fit(c: canvas.Canvas, img: Image.Image, x: float, y: float, w: float, h: float):
    """Dessine l'image en conservant le ratio dans une boîte w×h (centrée)."""
    img = ImageOps.exif_transpose(img).convert("RGBA")
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return
    scale = min(w / iw, h / ih)
    nw, nh = iw * scale, ih * scale
    ox = x + (w - nw) / 2
    oy = y + (h - nh) / 2
    c.drawImage(pil_to_reader(img), ox, oy, width=nw, height=nh, mask="auto")


def fit_font_size(c: canvas.Canvas, text: str, font: str, max_w: float, start: int, min_size: int = 7) -> int:
    size = start
    while size > min_size and c.stringWidth(text, font, size) > max_w:
        size -= 1
    return max(size, min_size)


def split_two_lines(c: canvas.Canvas, text: str, font: str, size: int, max_w: float) -> List[str]:
    """Split simple en 2 lignes si ça déborde."""
    if c.stringWidth(text, font, size) <= max_w:
        return [text]
    parts = text.split()
    if len(parts) <= 1:
        return [text]
    best = None
    for i in range(1, len(parts)):
        a = " ".join(parts[:i])
        b = " ".join(parts[i:])
        wa = c.stringWidth(a, font, size)
        wb = c.stringWidth(b, font, size)
        if wa <= max_w and wb <= max_w:
            best = (a, b)
    return list(best) if best else [text]


@dataclass
class BadgeDesign:
    # badge format (A5)
    badge_w_mm: float = 148
    badge_h_mm: float = 210

    # proportions de l'exemple : bande basse ~28%
    band_ratio: float = 0.28

    # marges internes
    margin_mm: float = 10

    # couleurs
    bg_black: str = BLACK
    band_color: str = BAND_DEFAULT
    accent: str = IMAGINE_ROSE

    # textes (modifiables)
    edition_text: str = "3e édition"
    event_title: str = "PARIS\nSACLAY\nSUMMIT"
    event_tagline: str = "CHOOSE SCIENCE"
    organised_by_label: str = "Organisé par"
    hashtag: str = "#ParisSaclaySummit"

    # tailles de police
    date_size_big: int = 22
    date_size_small: int = 12
    title_size: int = 20
    tagline_size: int = 10
    edition_size: int = 9
    organised_by_size: int = 7
    role_size: int = 14

    # cercle central
    circle_diameter_mm: float = 70


def draw_front_badge(
    c: canvas.Canvas,
    x: float,
    y: float,
    design: BadgeDesign,
    fonts: dict,
    person: dict,
    cfg: dict,
    sponsor_logos: List[Image.Image],
    center_image: Optional[Image.Image],
):
    bw = design.badge_w_mm * mm
    bh = design.badge_h_mm * mm
    band_h = bh * design.band_ratio
    black_h = bh - band_h
    margin = design.margin_mm * mm

    # fonds
    c.setFillColor(HexColor(design.bg_black))
    c.rect(x, y + band_h, bw, black_h, stroke=0, fill=1)
    c.setFillColor(HexColor(design.band_color))
    c.rect(x, y, bw, band_h, stroke=0, fill=1)

    # ===== Date block (top-left)
    if cfg["show_date"]:
        date_line1 = cfg["date_line1"]
        date_line2 = cfg["date_line2"]
        date_line3 = cfg["date_line3"]

        pad = 6 * mm
        box_w = 34 * mm
        box_h = 22 * mm

        bx = x + pad
        by = y + bh - box_h - pad

        c.setFillColor(HexColor(design.bg_black))
        # pas de rectangle : comme l'exemple, juste texte blanc + accent
        c.setFillColor(HexColor(WHITE))
        c.setFont(fonts["bold"], design.date_size_big)
        c.drawString(bx, by + 9 * mm, date_line1)

        c.setFont(fonts["regular"], design.date_size_small)
        c.drawString(bx, by + 3.5 * mm, date_line2)

        c.setFont(fonts["regular"], design.date_size_small)
        c.drawString(bx, by - 2.0 * mm, date_line3)

        # petit trait accent (Imagine rose)
        c.setFillColor(HexColor(design.accent))
        c.rect(bx, by + 1.5 * mm, 10 * mm, 1.2 * mm, stroke=0, fill=1)

    # ===== Cercle central
    circle_d = design.circle_diameter_mm * mm
    cx = x + bw / 2 - circle_d / 2
    cy = y + band_h + (black_h / 2) - (circle_d / 2) + 5 * mm

    if cfg["show_center_image"] and center_image is not None:
        # diamètre en px pour PIL (approx)
        dpx = int(design.circle_diameter_mm * 12)  # ~12 px/mm
        circ = make_circle_image(center_image, dpx)
        c.drawImage(pil_to_reader(circ), cx, cy, width=circle_d, height=circle_d, mask="auto")
    else:
        # fallback : cercle discret
        c.setStrokeColor(HexColor("#374151"))
        c.setLineWidth(1)
        c.circle(x + bw / 2, cy + circle_d / 2, circle_d / 2, stroke=1, fill=0)

    # ===== Textes sur le cercle
    if cfg["show_title"]:
        c.setFillColor(HexColor(WHITE))

        # édition (petit, en haut à droite du cercle)
        if cfg["show_edition"]:
            c.setFont(fonts["regular"], design.edition_size)
            c.drawRightString(cx + circle_d - 2 * mm, cy + circle_d - 8 * mm, cfg["edition_text"])

        # titre multi-lignes centré
        title_lines = cfg["event_title"].split("\n")
        c.setFont(fonts["bold"], design.title_size)
        line_h = (design.title_size + 2) * 0.9
        total_h = line_h * len(title_lines)
        ty = cy + circle_d / 2 + total_h / 2 - line_h
        for line in title_lines:
            c.drawCentredString(x + bw / 2, ty, line.strip())
            ty -= line_h

        # tagline
        if cfg["show_tagline"] and cfg["event_tagline"].strip():
            c.setFont(fonts["bold"], design.tagline_size)
            c.setFillColor(HexColor(design.accent))  # accent Imagine
            c.drawCentredString(x + bw / 2, cy + 10 * mm, cfg["event_tagline"].strip())

    # ===== Sponsors
    if cfg["show_sponsors"] and sponsor_logos:
        # label "Organisé par"
        if cfg["show_organised_by"]:
            c.setFillColor(HexColor(WHITE))
            c.setFont(fonts["regular"], design.organised_by_size)
            c.drawString(x + margin, y + band_h + 20 * mm, cfg["organised_by_label"])

        # rangée de logos en boîtes blanches
        logos = sponsor_logos[: cfg["max_sponsors"]]
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
            # image fit
            draw_image_fit(c, lg, bx + 1.2 * mm, box_y + 1.2 * mm, box_w - 2.4 * mm, box_h - 2.4 * mm)

    # ===== Bande basse : rôle / catégorie
    if cfg["show_role"]:
        role = str(person.get(cfg["role_col"], "")).strip().upper()
        if role:
            c.setFillColor(HexColor(BLACK))
            c.setFont(fonts["bold"], design.role_size)
            c.drawCentredString(x + bw / 2, y + band_h / 2 - 3 * mm, role)

    # ===== Option : nom / org / fonction dans bande ou sur noir (au choix)
    # (Par défaut : non, car ton exemple ne l’a pas sur la face avant.
    #  Si tu veux, on l’ajoute en 15 secondes.)
    # => on le laisse prêt à activer :
    if cfg["show_identity"]:
        full_name = (str(person.get(cfg["first_col"], "")).strip() + " " + str(person.get(cfg["last_col"], "")).strip()).strip()
        org = str(person.get(cfg["org_col"], "")).strip()
        title = str(person.get(cfg["title_col"], "")).strip()

        c.setFillColor(HexColor(WHITE))
        max_w = bw - 2 * margin

        # Nom (au-dessus des logos)
        c.setFont(fonts["bold"], 20)
        fs = fit_font_size(c, full_name, fonts["bold"], max_w, 24)
        c.setFont(fonts["bold"], fs)
        c.drawCentredString(x + bw / 2, y + band_h + 32 * mm, full_name)

        # Organisation (une ou deux lignes)
        if org:
            c.setFont(fonts["regular"], 11)
            lines = split_two_lines(c, org, fonts["regular"], 11, max_w)
            oy = y + band_h + 26 * mm
            for ln in lines[:2]:
                c.drawCentredString(x + bw / 2, oy, ln)
                oy -= 4.2 * mm

        # Fonction
        if title:
            c.setFont(fonts["regular"], 10)
            c.drawCentredString(x + bw / 2, y + band_h + 18 * mm, title)


def draw_cut_marks(c: canvas.Canvas, x: float, y: float, w: float, h: float):
    # traits très fins, 4 coins
    c.setStrokeColor(HexColor("#9CA3AF"))
    c.setLineWidth(0.3)
    L = 5 * mm
    # bottom-left
    c.line(x, y, x + L, y)
    c.line(x, y, x, y + L)
    # bottom-right
    c.line(x + w, y, x + w - L, y)
    c.line(x + w, y, x + w, y + L)
    # top-left
    c.line(x, y + h, x + L, y + h)
    c.line(x, y + h, x, y + h - L)
    # top-right
    c.line(x + w, y + h, x + w - L, y + h)
    c.line(x + w, y + h, x + w, y + h - L)


def generate_pdf(
    df: pd.DataFrame,
    design: BadgeDesign,
    cfg: dict,
    sponsor_logos: List[Image.Image],
    center_image: Optional[Image.Image],
) -> bytes:
    fonts = register_fonts()

    # Page format : A3 si A5×4, sinon A4 si A6×4
    if cfg["sheet"] == "A3 (4×A5)":
        page_w, page_h = A3
        badge_w = design.badge_w_mm * mm
        badge_h = design.badge_h_mm * mm
    else:
        # mode alternatif : 4×A6 sur A4 (au cas où)
        page_w, page_h = A4
        badge_w = (design.badge_w_mm * mm) / 2
        badge_h = (design.badge_h_mm * mm) / 2

    # placement 2×2
    gap = cfg["gap_mm"] * mm
    margin = cfg["page_margin_mm"] * mm

    positions = [
        (margin, page_h - margin - badge_h),
        (margin + badge_w + gap, page_h - margin - badge_h),
        (margin, page_h - margin - 2 * badge_h - gap),
        (margin + badge_w + gap, page_h - margin - 2 * badge_h - gap),
    ]

    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=(page_w, page_h))

    # itération 4 par page
    idx = 0
    rows = df.to_dict(orient="records")
    while idx < len(rows):
        for pos_i in range(4):
            if idx >= len(rows):
                break
            px, py = positions[pos_i]
            person = rows[idx]

            # Ajuste design badge size si A4(4×A6)
            local_design = design
            if cfg["sheet"] != "A3 (4×A5)":
                local_design = BadgeDesign(
                    badge_w_mm=design.badge_w_mm / 2,
                    badge_h_mm=design.badge_h_mm / 2,
                    band_ratio=design.band_ratio,
                    margin_mm=design.margin_mm / 2,
                    bg_black=design.bg_black,
                    band_color=design.band_color,
                    accent=design.accent,
                    edition_text=design.edition_text,
                    event_title=design.event_title,
                    event_tagline=design.event_tagline,
                    organised_by_label=design.organised_by_label,
                    hashtag=design.hashtag,
                    date_size_big=max(12, int(design.date_size_big * 0.7)),
                    date_size_small=max(8, int(design.date_size_small * 0.7)),
                    title_size=max(12, int(design.title_size * 0.7)),
                    tagline_size=max(7, int(design.tagline_size * 0.7)),
                    edition_size=max(7, int(design.edition_size * 0.8)),
                    organised_by_size=max(6, int(design.organised_by_size * 0.9)),
                    role_size=max(10, int(design.role_size * 0.75)),
                    circle_diameter_mm=design.circle_diameter_mm * 0.7,
                )

            draw_front_badge(
                c=c,
                x=px,
                y=py,
                design=local_design,
                fonts=fonts,
                person=person,
                cfg=cfg,
                sponsor_logos=sponsor_logos,
                center_image=center_image,
            )

            if cfg["cut_marks"]:
                draw_cut_marks(c, px, py, local_design.badge_w_mm * mm, local_design.badge_h_mm * mm)

            idx += 1

        c.showPage()

    c.save()
    return bio.getvalue()


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Générateur de badges — Imagine", layout="wide")
st.title("Générateur de badges (A5, 4 par page) — design inspiré de ton modèle")

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("1) Données participants")
    file = st.file_uploader("CSV ou Excel", type=["csv", "xlsx"])

    df = None
    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.write("Aperçu :")
        st.dataframe(df.head(20), use_container_width=True)

with col_right:
    st.subheader("2) Visuels & personnalisation")
    sheet = st.selectbox("Format d'impression", ["A3 (4×A5)", "A4 (4×A6)"], index=0)

    center_up = st.file_uploader("Image centrale (cercle) — optionnel", type=["png", "jpg", "jpeg"])
    center_img = Image.open(center_up) if center_up else None

    sponsors_up = st.file_uploader(
        "Logos sponsors (PNG conseillé, fond transparent) — multiple",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    sponsor_imgs: List[Image.Image] = []
    if sponsors_up:
        for f in sponsors_up:
            sponsor_imgs.append(Image.open(f))

    bg_black = st.color_picker("Fond noir", BLACK)
    band_color = st.color_picker("Bande basse", BAND_DEFAULT)
    accent = st.color_picker("Accent (Imagine)", IMAGINE_ROSE)

st.divider()

st.subheader("3) Textes (modifiables) + cases à cocher")

c1, c2, c3 = st.columns(3)

with c1:
    show_date = st.checkbox("Afficher la date", True)
    date_line1 = st.text_input("Date ligne 1 (ex: 18 19)", "18 19")
    date_line2 = st.text_input("Date ligne 2 (ex: février)", "février")
    date_line3 = st.text_input("Date ligne 3 (ex: 2026)", "2026")

    show_center_image = st.checkbox("Afficher l'image centrale", True)

with c2:
    show_title = st.checkbox("Afficher le titre événement", True)
    show_edition = st.checkbox("Afficher l'édition", True)
    edition_text = st.text_input("Texte édition", "3e édition")
    event_title = st.text_area("Titre (une ligne par retour à la ligne)", "PARIS\nSACLAY\nSUMMIT", height=90)
    show_tagline = st.checkbox("Afficher le sous-titre / tagline", True)
    event_tagline = st.text_input("Tagline", "CHOOSE SCIENCE")

with c3:
    show_sponsors = st.checkbox("Afficher la rangée de logos", True)
    show_organised_by = st.checkbox("Afficher “Organisé par”", True)
    organised_by_label = st.text_input("Libellé", "Organisé par")
    max_sponsors = st.slider("Nombre max de logos affichés", 1, 8, 4)
    cut_marks = st.checkbox("Traits de coupe", True)

st.divider()

st.subheader("4) Champs variables (CSV) — tu choisis quoi afficher")
if df is not None:
    cols = list(df.columns)

    map_c1, map_c2 = st.columns(2)

    with map_c1:
        first_col = st.selectbox("Colonne prénom", cols, index=cols.index("prenom") if "prenom" in cols else 0)
        last_col = st.selectbox("Colonne nom", cols, index=cols.index("nom") if "nom" in cols else 0)
        org_col = st.selectbox("Colonne organisation", cols, index=cols.index("organisation") if "organisation" in cols else 0)

    with map_c2:
        title_col = st.selectbox("Colonne fonction (optionnel)", cols, index=cols.index("fonction") if "fonction" in cols else 0)
        role_col = st.selectbox("Colonne rôle/catégorie (ex: INTERVENANT)", cols, index=cols.index("categorie") if "categorie" in cols else 0)

    show_role = st.checkbox("Afficher le rôle dans la bande basse", True)
    show_identity = st.checkbox("Afficher Nom/Organisation/Fonction sur la face avant", False)

    st.caption("Astuce : si tu veux EXACTEMENT comme l'exemple (sans nom), laisse “Afficher Nom/Organisation/Fonction” décoché.")

    st.divider()

    st.subheader("5) Mise en page d'impression")
    p1, p2, p3 = st.columns(3)
    with p1:
        page_margin_mm = st.number_input("Marge page (mm)", min_value=0, max_value=25, value=6)
    with p2:
        gap_mm = st.number_input("Espace entre badges (mm)", min_value=0, max_value=20, value=6)
    with p3:
        st.write("4 badges par page : 2×2")

    if st.button("Générer le PDF"):
        design = BadgeDesign(
            bg_black=bg_black,
            band_color=band_color,
            accent=accent,
        )
        cfg = {
            "sheet": sheet,
            "gap_mm": float(gap_mm),
            "page_margin_mm": float(page_margin_mm),
            "cut_marks": cut_marks,

            "show_date": show_date,
            "date_line1": date_line1,
            "date_line2": date_line2,
            "date_line3": date_line3,

            "show_center_image": show_center_image,

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

            "show_role": show_role,
            "show_identity": show_identity,

            "first_col": first_col,
            "last_col": last_col,
            "org_col": org_col,
            "title_col": title_col,
            "role_col": role_col,
        }

        pdf_bytes = generate_pdf(
            df=df,
            design=design,
            cfg=cfg,
            sponsor_logos=sponsor_imgs,
            center_image=center_img if show_center_image else None,
        )

        st.success("PDF généré.")
        st.download_button(
            "Télécharger le PDF",
            data=pdf_bytes,
            file_name="badges.pdf",
            mime="application/pdf",
        )
else:
    st.info("Charge un CSV/Excel pour activer la génération.")
