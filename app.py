import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from skimage import measure, morphology, feature
from skimage.feature import graycomatrix, graycoprops

st.set_page_config(page_title="Microplast Counter (Lab)", layout="wide")

st.title("🔬 Microplast Counter (Lab) — contagem, tamanho (µm), cor, textura e forma")

st.markdown("""
**Fluxo:** upload da imagem → calibração (µm/pixel) → segmentação → separação → métricas → export (CSV + imagem anotada).
""")

# ----------------------------
# Helpers
# ----------------------------
def clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def compute_lbp(gray, P=8, R=1):
    # LBP uniform
    lbp = feature.local_binary_pattern(gray, P, R, method="uniform")
    return lbp

def glcm_features(gray_roi):
    # GLCM on ROI (downscale to reduce noise)
    g = cv2.resize(gray_roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    g = np.clip(g, 0, 255).astype(np.uint8)

    # distances and angles
    glcm = graycomatrix(g, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    feats = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        v = graycoprops(glcm, prop).mean()
        feats[f"glcm_{prop}"] = float(v)

    # entropy (custom)
    p = glcm.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    ent = -np.sum(p * np.log2(p + 1e-12))
    feats["glcm_entropy"] = float(ent)
    return feats

def circularity(area, perimeter):
    if perimeter <= 1e-9:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))

def safe_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ----------------------------
# UI
# ----------------------------
colA, colB = st.columns([1, 1])

with colA:
    uploaded = st.file_uploader("📷 Envie a imagem (PNG/JPG)", type=["png", "jpg", "jpeg"])

    st.subheader("📏 Calibração (µm por pixel)")
    st.caption("Se você tiver barra de escala: use um valor consistente por aumento **ou** digite o µm/pixel já conhecido.")
    um_per_px = st.number_input("µm/pixel (ex: 0.65)", min_value=0.000001, value=1.0, step=0.01, format="%.6f")

    st.subheader("⚙️ Segmentação")
    seg_mode = st.selectbox("Modo", ["Otsu (rápido)", "Adaptive (bom p/ iluminação irregular)"])
    invert = st.checkbox("Inverter (se microplástico ficar branco no fundo preto)", value=False)

    st.subheader("🧹 Limpeza / Separação")
    min_area_um2 = st.number_input("Área mínima (µm²) p/ aceitar objeto", min_value=0.0, value=50.0, step=10.0)
    max_area_um2 = st.number_input("Área máxima (µm²) p/ aceitar objeto", min_value=0.0, value=1e7, step=1000.0)
    do_watershed = st.checkbox("Separar objetos grudados (watershed)", value=True)

    st.subheader("🏷️ Regras iniciais de tipo")
    fiber_aspect = st.slider("Razão de aspecto mínima p/ 'fibra'", 2.0, 30.0, 8.0, 0.5)
    pellet_circ = st.slider("Circularidade mínima p/ 'pellet'", 0.1, 1.0, 0.75, 0.01)

with colB:
    st.subheader("🧾 Saída")
    st.caption("O app gera tabela por objeto e imagem anotada. Você pode baixar ambos ao final.")
    show_debug = st.checkbox("Mostrar imagens intermediárias (debug)", value=True)


# ----------------------------
# Main
# ----------------------------
if not uploaded:
    st.info("Envie uma imagem para começar.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Não consegui ler a imagem.")
    st.stop()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray = clahe_gray(gray)
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Segment
if seg_mode.startswith("Otsu"):
    _, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
else:
    bw = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 2)

if invert:
    bw = 255 - bw

# Morph cleanup
bw = morphology.remove_small_objects(bw.astype(bool), min_size=20)
bw = morphology.remove_small_holes(bw, area_threshold=50)
bw = bw.astype(np.uint8) * 255

# Optional watershed
labels = None
if do_watershed:
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # markers
    _, sure_fg = cv2.threshold((dist_norm * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ws_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    markers = cv2.watershed(cv2.cvtColor(ws_img, cv2.COLOR_RGB2BGR), markers)

    # watershed produces -1 boundaries
    labels = markers.copy()
    labels[labels <= 1] = 0
else:
    labels = measure.label(bw > 0, connectivity=2)

# Extract regions
props = measure.regionprops(labels, intensity_image=gray)
records = []

# conversions
px_area_to_um2 = (um_per_px ** 2)
px_to_um = um_per_px

annot = img_rgb.copy()
font = cv2.FONT_HERSHEY_SIMPLEX

for i, r in enumerate(props, start=1):
    area_px = float(r.area)
    area_um2 = area_px * px_area_to_um2

    if area_um2 < min_area_um2 or area_um2 > max_area_um2:
        continue

    # geometry
    perim = float(r.perimeter) if r.perimeter else 0.0
    circ = circularity(area_px, perim)

    major = float(r.major_axis_length) if r.major_axis_length else 0.0
    minor = float(r.minor_axis_length) if r.minor_axis_length else 0.0
    aspect = float(major / (minor + 1e-9))

    # bounding box
    minr, minc, maxr, maxc = r.bbox
    roi_rgb = img_rgb[minr:maxr, minc:maxc]
    roi_gray = gray[minr:maxr, minc:maxc]

    # mask for this region in bbox
    region_mask = (labels[minr:maxr, minc:maxc] == r.label).astype(np.uint8)

    # Color stats in LAB
    roi_lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    lab_vals = roi_lab[region_mask.astype(bool)]
    L_mean, a_mean, b_mean = lab_vals.mean(axis=0) if lab_vals.size else (np.nan, np.nan, np.nan)

    # Texture features
    # LBP mean
    lbp = compute_lbp(roi_gray)
    lbp_vals = lbp[region_mask.astype(bool)]
    lbp_mean = float(np.mean(lbp_vals)) if lbp_vals.size else np.nan

    glcm_feats = glcm_features(roi_gray)

    # classification (initial rules)
    tipo = "fragmento"
    if aspect >= fiber_aspect:
        tipo = "fibra"
    elif circ >= pellet_circ:
        tipo = "pellet"
    # "filme" heuristic (flat + low solidity)
    solidity = float(r.solidity) if r.solidity else 0.0
    if tipo == "fragmento" and solidity < 0.6 and area_um2 > (min_area_um2 * 10):
        tipo = "filme"

    # size in um
    length_um = major * px_to_um
    width_um = minor * px_to_um
    equiv_diam_um = float(np.sqrt(4 * area_px / np.pi) * px_to_um)

    # centroid for annotation
    cy, cx = r.centroid
    cx, cy = int(cx), int(cy)

    # draw contour
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # shift contour coords to image coords
        cnt = contours[0].copy()
        cnt[:, 0, 0] += minc
        cnt[:, 0, 1] += minr
        cv2.drawContours(annot, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(annot, str(i), (cx, cy), font, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    rec = {
        "id": i,
        "tipo": tipo,
        "area_um2": area_um2,
        "length_um": length_um,
        "width_um": width_um,
        "equiv_diam_um": equiv_diam_um,
        "perimeter_px": perim,
        "circularity": circ,
        "aspect_ratio": aspect,
        "solidity": solidity,
        "L_mean": float(L_mean),
        "a_mean": float(a_mean),
        "b_mean": float(b_mean),
        "lbp_mean": lbp_mean,
        **glcm_feats
    }
    records.append(rec)

df = pd.DataFrame(records)
count = len(df)

st.subheader("✅ Resultados")
k1, k2, k3 = st.columns(3)
k1.metric("Contagem", count)
k2.metric("µm/pixel", f"{um_per_px:.6f}")
k3.metric("Imagem", f"{img_rgb.shape[1]}×{img_rgb.shape[0]} px")

c1, c2 = st.columns([1, 1])

with c1:
    st.image(img_rgb, caption="Imagem original", use_container_width=True)

with c2:
    st.image(annot, caption="Imagem anotada (contornos + IDs)", use_container_width=True)

if show_debug:
    st.subheader("🧪 Debug")
    d1, d2 = st.columns(2)
    with d1:
        st.image(gray, caption="Gray (CLAHE)", use_container_width=True)
    with d2:
        st.image(bw, caption="Máscara binária (limpa)", use_container_width=True)

st.subheader("📊 Tabela por partícula")
st.dataframe(df, use_container_width=True)

# Downloads
st.subheader("⬇️ Baixar resultados")
csv_bytes = df.to_csv(index=False).encode("utf-8")

# encode annotated image
annot_bgr = cv2.cvtColor(annot, cv2.COLOR_RGB2BGR)
ok, png = cv2.imencode(".png", annot_bgr)
img_bytes = png.tobytes() if ok else b""

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button("Baixar CSV", data=csv_bytes, file_name="microplast_resultados.csv", mime="text/csv")
with dl2:
    st.download_button("Baixar imagem anotada (PNG)", data=img_bytes, file_name="microplast_anotada.png", mime="image/png")
