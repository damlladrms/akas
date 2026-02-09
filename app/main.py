from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# -----------------------------
# Config
# -----------------------------
REQUIRED_FIELDS = {
    "title": ["title", "product_title", "urun_adi", "ürün adı", "urunadi", "name", "product_name", "baslik", "başlık"],
    "image_url": ["image_url", "img_url", "image", "imageurl", "gorsel", "görsel", "photo", "foto", "url"],
    "category": ["category", "kategori", "cat", "kategori_adi", "category_name"],
}

# output columns
OUT_COLS = [
    "quality_score",
    "issues",
    "correction_notes",
    "correct_title",
    "correct_category",
    "correct_image_url",
]

# In-memory store (demo için yeterli)
@dataclass
class AnalysisBundle:
    df_original: pd.DataFrame
    df_result: pd.DataFrame
    mapping: Dict[str, str]  # internal_field -> column_name used


STORE: Dict[str, AnalysisBundle] = {}


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="AKAS - Akıllı Katalog Analiz Sistemi")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# -----------------------------
# Helpers
# -----------------------------
def _normalize(s: str) -> str:
    return str(s).strip().lower()


def detect_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Try to match internal fields to existing columns using aliases.
    Returns internal_field -> found_column or None
    """
    col_map = {c: _normalize(c) for c in columns}
    found: Dict[str, Optional[str]] = {}

    for internal, aliases in REQUIRED_FIELDS.items():
        hit = None
        for c, cn in col_map.items():
            if cn == internal:
                hit = c
                break
        if hit is None:
            for alias in aliases:
                alias_n = _normalize(alias)
                for c, cn in col_map.items():
                    if cn == alias_n:
                        hit = c
                        break
                if hit is not None:
                    break
        found[internal] = hit
    return found


def read_uploaded_file(upload: UploadFile) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    content = upload.file.read()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        # Excel
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")

    if name.endswith(".csv"):
        # CSV (encoding fallback)
        for enc in ("utf-8-sig", "utf-8", "cp1254", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(content), encoding=enc)
            except Exception:
                continue
        # last resort
        return pd.read_csv(io.BytesIO(content), encoding="utf-8", errors="ignore")

    raise ValueError("Desteklenmeyen dosya türü. Lütfen CSV veya Excel (.xlsx) yükleyin.")


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def analyze_row(title: str, image_url: str, category: str) -> Tuple[int, List[str], Dict[str, str]]:
    """
    Basit MVP analiz kuralları (demo için yeterli):
    - Başlık zayıf: çok kısa / çok tekrar / anlamsız karakter
    - Görsel eksik: boş veya 'http' içermiyor
    - Kategori şüpheli: boş
    Skor: 0-100
    """
    issues = []
    corrections = {
        "correct_title": title,
        "correct_category": category,
        "correct_image_url": image_url,
    }

    score = 100

    t = title
    c = category
    img = image_url

    # Title checks
    if len(t) < 12:
        issues.append("Zayıf başlık (çok kısa)")
        score -= 18

    if "??" in t or "!!!" in t or "..." in t:
        issues.append("Zayıf başlık (şüpheli karakter)")
        score -= 10

    if t.count("  ") >= 1:
        issues.append("Zayıf başlık (fazla boşluk)")
        score -= 6
        corrections["correct_title"] = " ".join(t.split())

    # Image checks
    if not img or ("http" not in img.lower() and "www" not in img.lower()):
        issues.append("Görsel eksik / geçersiz URL")
        score -= 30

    # Category checks
    if not c:
        issues.append("Kategori boş / şüpheli")
        score -= 22

    score = max(0, min(100, score))
    return score, issues, corrections


def build_result_df(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    title_col = mapping["title"]
    image_col = mapping["image_url"]
    cat_col = mapping["category"]

    out = df.copy()

    quality_scores = []
    issues_list = []
    notes_list = []
    correct_title = []
    correct_category = []
    correct_image_url = []

    for _, row in out.iterrows():
        title = safe_str(row.get(title_col, ""))
        image_url = safe_str(row.get(image_col, "")) if image_col else ""
        category = safe_str(row.get(cat_col, "")) if cat_col else ""

        score, issues, corr = analyze_row(title, image_url, category)

        quality_scores.append(score)
        issues_list.append("; ".join(issues) if issues else "Sorun yok")
        # notes: only if correction applied
        notes = []
        if corr["correct_title"] != title:
            notes.append("Başlık düzeltildi (boşluk temizleme)")
        if ("Görsel eksik / geçersiz URL" in issues):
            notes.append("Görsel URL kontrol edin")
        if ("Kategori boş / şüpheli" in issues):
            notes.append("Kategori alanını doldurun")
        notes_list.append("; ".join(notes) if notes else "")

        correct_title.append(corr["correct_title"])
        correct_category.append(corr["correct_category"])
        correct_image_url.append(corr["correct_image_url"])

    out["quality_score"] = quality_scores
    out["issues"] = issues_list
    out["correction_notes"] = notes_list
    out["correct_title"] = correct_title
    out["correct_category"] = correct_category
    out["correct_image_url"] = correct_image_url

    # make sure these columns are at the end (nice view)
    # (keep original columns first)
    for col in OUT_COLS:
        if col not in out.columns:
            out[col] = ""
    return out


def summary_metrics(df_result: pd.DataFrame) -> Dict[str, int]:
    # suspicious title: contains "Zayıf başlık"
    suspicious_titles = int(df_result["issues"].str.contains("Zayıf başlık", na=False).sum())
    missing_images = int(df_result["issues"].str.contains("Görsel eksik", na=False).sum())
    avg_score = int(round(float(df_result["quality_score"].mean()))) if len(df_result) else 0
    total = int(len(df_result))
    return {
        "total": total,
        "suspicious_titles": suspicious_titles,
        "missing_images": missing_images,
        "avg_score": avg_score,
    }


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
from fastapi.responses import RedirectResponse

@app.get("/analyze")
def analyze_get():
    return RedirectResponse(url="/", status_code=302)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...)):
    try:
        df = read_uploaded_file(file)
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Dosya okunamadı: {e}"},
            status_code=400,
        )

    # Detect mapping
    detected = detect_mapping(list(df.columns))
    missing = [k for k, v in detected.items() if v is None]

    # If missing required columns -> go mapping screen
    if missing:
        # store original df temporarily with a token
        token = str(uuid.uuid4())
        # create placeholder mapping (None allowed here, but we’ll finalize after user selection)
        STORE[token] = AnalysisBundle(df_original=df, df_result=df, mapping={})
        return templates.TemplateResponse(
            "map_columns.html",
            {
                "request": request,
                "token": token,
                "columns": list(df.columns),
                "detected": detected,
                "missing": missing,
            },
        )

    # Build result
    token = str(uuid.uuid4())
    mapping = {"title": detected["title"], "image_url": detected["image_url"], "category": detected["category"]}
    df_result = build_result_df(df, mapping)
    STORE[token] = AnalysisBundle(df_original=df, df_result=df_result, mapping=mapping)

    metrics = summary_metrics(df_result)
    preview = df_result.head(50).to_dict(orient="records")

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "token": token,
            "metrics": metrics,
            "columns": list(df_result.columns),
            "rows": preview,
        },
    )


@app.post("/map-columns", response_class=HTMLResponse)
async def map_columns(
    request: Request,
    token: str = Form(...),
    map_title: str = Form(...),
    map_image_url: str = Form(""),
    map_category: str = Form(...),
):
    if token not in STORE:
        return RedirectResponse(url="/", status_code=303)

    bundle = STORE[token]
    df = bundle.df_original

    mapping = {
        "title": map_title,
        "image_url": map_image_url.strip() if map_image_url else "",
        "category": map_category,
    }

    df_result = build_result_df(df, mapping)
    new_token = str(uuid.uuid4())
    STORE[new_token] = AnalysisBundle(df_original=df, df_result=df_result, mapping=mapping)

    metrics = summary_metrics(df_result)
    preview = df_result.head(50).to_dict(orient="records")

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "token": new_token,
            "metrics": metrics,
            "columns": list(df_result.columns),
            "rows": preview,
        },
    )


@app.get("/download/{token}")
def download(token: str):
    if token not in STORE:
        return RedirectResponse(url="/", status_code=303)
    df = STORE[token].df_result
    data = df_to_csv_bytes(df)
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="akas_sonuclar_{token[:8]}.csv"'},
    )

