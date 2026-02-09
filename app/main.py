import os
import re
import uuid
from typing import Dict, Optional, List

import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ----------------------------
# App & paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Render'da yazılabilir alan: /tmp
CACHE_DIR = os.environ.get("AKAS_CACHE_DIR", "/tmp/akas_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI(title="Akıllı Katalog Analiz Sistemi (AKAS)")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------------------
# Helpers: file read + cache
# ----------------------------
def _safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name or "upload")
    return name[:80] if len(name) > 80 else name


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    content = file.file.read()

    if not content:
        raise ValueError("Dosya boş görünüyor.")

    # CSV
    if filename.endswith(".csv"):
        # UTF-8 / cp1254 vb. için basit fallback
        for enc in ("utf-8-sig", "utf-8", "cp1254", "latin1"):
            try:
                return pd.read_csv(pd.io.common.BytesIO(content), encoding=enc)
            except Exception:
                pass
        raise ValueError("CSV okunamadı (encoding sorunu olabilir).")

    # Excel
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(pd.io.common.BytesIO(content))

    raise ValueError("Sadece CSV veya Excel (.xlsx/.xls) desteklenir.")


def cache_df(token: str, df: pd.DataFrame) -> str:
    path = os.path.join(CACHE_DIR, f"{token}.parquet")
    df.to_parquet(path, index=False)
    return path


def load_cached_df(token: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"{token}.parquet")
    if not os.path.exists(path):
        raise ValueError("Geçici dosya bulunamadı. Lütfen dosyayı yeniden yükleyin.")
    return pd.read_parquet(path)


# ----------------------------
# Column mapping detection
# ----------------------------
EXPECTED = ["title", "image_url", "category", "sub_category"]

SYNONYMS = {
    "title": ["title", "urun_adi", "ürün adı", "urun adi", "product_name", "name", "başlık", "baslik", "product title"],
    "image_url": ["image_url", "image", "img", "gorsel", "görsel", "gorsel_url", "image link", "image_link", "resim", "photo", "foto", "url", "image path", "path"],
    "category": ["category", "kategori", "cat", "main_category", "ana kategori", "ana_kategori"],
    "sub_category": ["sub_category", "subcategory", "alt kategori", "alt_kategori", "sub cat", "subcat"],
}


def normalize_col(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ö", "o").replace("ü", "u").replace("ç", "c")
    c = re.sub(r"\s+", "_", c)
    return c


def detect_mapping(cols: List[str]) -> Dict[str, Optional[str]]:
    norm_map = {c: normalize_col(c) for c in cols}
    detected: Dict[str, Optional[str]] = {k: None for k in EXPECTED}

    for expected_key, syns in SYNONYMS.items():
        syn_norms = [normalize_col(s) for s in syns]
        for original, n in norm_map.items():
            if n in syn_norms:
                detected[expected_key] = original
                break
    return detected


# ----------------------------
# Analysis logic (MVP but stable)
# ----------------------------
def is_missing_image(val) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return True
    return False


def title_is_suspicious(title: str) -> bool:
    if title is None:
        return True
    t = str(title).strip()
    if len(t) < 8:
        return True
    # çok fazla tekrar / anlamsız karakterler
    if re.search(r"(.)\1\1\1", t):  # aaaa gibi
        return True
    # aşırı sembol
    sym_ratio = sum(1 for ch in t if not ch.isalnum() and ch != " ") / max(1, len(t))
    if sym_ratio > 0.25:
        return True
    return False


def suggest_title_fix(title: str) -> str:
    if title is None:
        return ""
    t = str(title).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[|/_\-]{2,}", " - ", t)
    t = t.strip(" -|_/")
    return t[:120]


def base_score(row: dict) -> int:
    score = 100
    t = row.get("title", None)
    img = row.get("image_url", None)

    if title_is_suspicious(t):
        score -= 25
    if is_missing_image(img):
        score -= 35

    # kategori doluysa küçük bonus (opsiyonel alan)
    if row.get("category"):
        score += 3
    if row.get("sub_category"):
        score += 2

    return max(0, min(100, score))


def analyze_df(df: pd.DataFrame) -> pd.DataFrame:
    # Beklenen kolonlar yoksa yine de çalışsın diye güvenli al
    if "title" not in df.columns:
        df["title"] = ""
    if "image_url" not in df.columns:
        df["image_url"] = ""

    out = df.copy()

    out["baslik_supheli"] = out["title"].apply(title_is_suspicious)
    out["gorsel_eksik"] = out["image_url"].apply(is_missing_image)

    # Öneriler (Yanlış/Doğru kolonları mantığı)
    out["yanlis_baslik"] = out["title"].fillna("").astype(str)
    out["dogru_baslik_onerisi"] = out["title"].apply(suggest_title_fix)

    out["yanlis_gorsel"] = out["image_url"].fillna("").astype(str)
    out["dogru_gorsel_onerisi"] = out["image_url"].apply(lambda x: "" if is_missing_image(x) else str(x).strip())

    # Skor
    out["kalite_skoru"] = out.apply(lambda r: base_score(r.to_dict()), axis=1)

    # Kısa açıklama/öneri
    def tip(row):
        tips = []
        if row["baslik_supheli"]:
            tips.append("Başlık zayıf/şüpheli: daha açıklayıcı hale getirin.")
        if row["gorsel_eksik"]:
            tips.append("Görsel eksik: ürün görseli URL/Path ekleyin.")
        if not tips:
            tips.append("Genel kalite iyi.")
        return " ".join(tips)

    out["oneriler"] = out.apply(tip, axis=1)
    return out


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "error": None},
    )


@app.get("/analyze")
def analyze_get():
    # Doğru akış: upload -> POST /analyze
    return RedirectResponse(url="/", status_code=302)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_upload(request: Request, file: UploadFile = File(...)):
    try:
        df = read_uploaded_file(file)
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Dosya okunamadı: {e}"},
            status_code=400,
        )

    # kolon listesi
    cols = [str(c) for c in df.columns.tolist()]
    detected = detect_mapping(cols)

    token = str(uuid.uuid4())
    cache_df(token, df)

    # title + image_url otomatik bulunduysa direkt analiz
    if detected.get("title") and detected.get("image_url"):
        # rename
        mapped_df = df.rename(
            columns={
                detected["title"]: "title",
                detected["image_url"]: "image_url",
                detected.get("category", ""): "category" if detected.get("category") else None,
                detected.get("sub_category", ""): "sub_category" if detected.get("sub_category") else None,
            }
        )
        # Yukarıdaki rename içinde None anahtarlar sorun çıkarabilir; temizleyelim:
        mapped_df = mapped_df.copy()
        # Kategori eşleşmemişse yok say
        if "category" not in mapped_df.columns:
            mapped_df["category"] = ""
        if "sub_category" not in mapped_df.columns:
            mapped_df["sub_category"] = ""

        analyzed = analyze_df(mapped_df)

        summary = {
            "suspect_titles": int(analyzed["baslik_supheli"].sum()),
            "missing_images": int(analyzed["gorsel_eksik"].sum()),
            "avg_score": float(analyzed["kalite_skoru"].mean()) if len(analyzed) else 0.0,
            "rows": int(len(analyzed)),
        }

        table_cols = [
            "title", "image_url", "category", "sub_category",
            "kalite_skoru", "baslik_supheli", "gorsel_eksik",
            "yanlis_baslik", "dogru_baslik_onerisi",
            "yanlis_gorsel", "dogru_gorsel_onerisi",
            "oneriler",
        ]
        # eksik kolon varsa ekleyelim
        for c in table_cols:
            if c not in analyzed.columns:
                analyzed[c] = ""

        records = analyzed[table_cols].fillna("").to_dict(orient="records")

        return templates.TemplateResponse(
            "results.html",
            {"request": request, "summary": summary, "records": records, "columns": table_cols},
        )

    # Aksi halde eşleştirme ekranı
    return templates.TemplateResponse(
        "map_columns.html",
        {
            "request": request,
            "token": token,
            "columns": cols,
            "detected": detected,
            "error": None,
        },
    )


@app.post("/analyze-mapped", response_class=HTMLResponse)
async def analyze_mapped(
    request: Request,
    token: str = Form(...),
    title_col: Optional[str] = Form(None),
    image_col: Optional[str] = Form(None),
    category_col: Optional[str] = Form(None),
    sub_category_col: Optional[str] = Form(None),
):
    # dropdown boş geliyorsa burada patlardı; ama biz columns'u dosyadan değil ekrandan seçtiğimiz için OK
    try:
        df = load_cached_df(token)
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Geçici dosya bulunamadı: {e}"},
            status_code=400,
        )

    cols = [str(c) for c in df.columns.tolist()]

    # Seçimler güvenli mi?
    if title_col and title_col not in cols:
        title_col = None
    if image_col and image_col not in cols:
        image_col = None
    if category_col and category_col not in cols:
        category_col = None
    if sub_category_col and sub_category_col not in cols:
        sub_category_col = None

    # title ve image seçilmediyse yine de çalışsın ama uyarı verelim
    mapped = df.copy()

    if title_col:
        mapped = mapped.rename(columns={title_col: "title"})
    else:
        mapped["title"] = ""

    if image_col:
        mapped = mapped.rename(columns={image_col: "image_url"})
    else:
        mapped["image_url"] = ""

    if category_col:
        mapped = mapped.rename(columns={category_col: "category"})
    else:
        mapped["category"] = ""

    if sub_category_col:
        mapped = mapped.rename(columns={sub_category_col: "sub_category"})
    else:
        mapped["sub_category"] = ""

    analyzed = analyze_df(mapped)

    summary = {
        "suspect_titles": int(analyzed["baslik_supheli"].sum()),
        "missing_images": int(analyzed["gorsel_eksik"].sum()),
        "avg_score": float(analyzed["kalite_skoru"].mean()) if len(analyzed) else 0.0,
        "rows": int(len(analyzed)),
    }

    table_cols = [
        "title", "image_url", "category", "sub_category",
        "kalite_skoru", "baslik_supheli", "gorsel_eksik",
        "yanlis_baslik", "dogru_baslik_onerisi",
        "yanlis_gorsel", "dogru_gorsel_onerisi",
        "oneriler",
    ]
    for c in table_cols:
        if c not in analyzed.columns:
            analyzed[c] = ""

    records = analyzed[table_cols].fillna("").to_dict(orient="records")

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "summary": summary, "records": records, "columns": table_cols},
    )
