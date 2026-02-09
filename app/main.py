from __future__ import annotations

import io
import uuid
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AKAS - Akıllı Katalog Analiz Sistemi")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# token -> uploaded df (demo/mvp için yeterli)
DF_STORE: Dict[str, pd.DataFrame] = {}
RESULT_STORE: Dict[str, pd.DataFrame] = {}


# -------------------------
# Helpers
# -------------------------
def read_uploaded_file(upload: UploadFile) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    raw = upload.file.read()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")

    if name.endswith(".csv"):
        # encoding + delimiter esnek okuma
        for enc in ("utf-8-sig", "utf-8", "cp1254", "latin1"):
            try:
                text = raw.decode(enc)
                try:
                    return pd.read_csv(io.StringIO(text), sep=None, engine="python")
                except Exception:
                    return pd.read_csv(io.StringIO(text))
            except Exception:
                continue

        # son çare
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="ignore")

    raise ValueError("Desteklenmeyen dosya. Lütfen CSV veya Excel (.xlsx) yükleyin.")


def detect_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    def n(x: str) -> str:
        return str(x).strip().lower()

    cols_norm = {c: n(c) for c in columns}

    aliases = {
        "title": ["title", "product_title", "product_name", "name", "urun_adi", "ürün adı", "baslik", "başlık"],
        "image_url": ["image_url", "img_url", "image", "gorsel", "görsel", "resim", "photo", "foto", "image_link"],
        "category": ["category", "kategori", "cat", "kategori_adi", "category_name"],
    }

    found = {"title": None, "image_url": None, "category": None}

    for key, keys in aliases.items():
        hit = None
        for c, cn in cols_norm.items():
            if cn in [n(k) for k in keys]:
                hit = c
                break
        found[key] = hit
    return found


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def analyze_row(title: str, image_url: str, category: str) -> Tuple[int, str, str, str, str]:
    issues = []
    score = 100

    t = safe_str(title)
    i = safe_str(image_url)
    c = safe_str(category)

    # Title
    corrected_title = " ".join(t.split())
    if len(t) < 12:
        issues.append("Zayıf başlık (çok kısa)")
        score -= 18
    if corrected_title != t:
        issues.append("Başlıkta gereksiz boşluk")
        score -= 6

    # Image
    corrected_image = i
    if not i or ("http" not in i.lower() and "www" not in i.lower()):
        issues.append("Görsel eksik / geçersiz URL")
        score -= 30

    # Category
    corrected_category = c
    if not c:
        issues.append("Kategori boş / şüpheli")
        score -= 22

    score = max(0, min(100, score))
    issues_text = "; ".join(issues) if issues else "Sorun yok"

    # öneri notu
    notes = []
    if "Zayıf başlık (çok kısa)" in issues_text:
        notes.append("Başlığı daha açıklayıcı yapın (marka + model + özellik)")
    if "Görsel eksik / geçersiz URL" in issues_text:
        notes.append("Geçerli görsel URL ekleyin")
    if "Kategori boş / şüpheli" in issues_text:
        notes.append("Kategori alanını doldurun")
    note_text = "; ".join(notes)

    return score, issues_text, note_text, corrected_title, corrected_category


def run_analysis(df: pd.DataFrame, title_col: str, image_col: str, category_col: str) -> pd.DataFrame:
    out = df.copy()

    # source columns
    titles = out[title_col].astype(str).fillna("") if title_col in out.columns else ""
    images = out[image_col].astype(str).fillna("") if image_col and image_col in out.columns else ""
    cats = out[category_col].astype(str).fillna("") if category_col in out.columns else ""

    scores, issues, notes, ctitle, ccat = [], [], [], [], []

    for idx in range(len(out)):
        t = titles.iloc[idx] if hasattr(titles, "iloc") else ""
        i = images.iloc[idx] if hasattr(images, "iloc") else ""
        c = cats.iloc[idx] if hasattr(cats, "iloc") else ""

        sc, iss, nt, ct, cc = analyze_row(t, i, c)
        scores.append(sc)
        issues.append(iss)
        notes.append(nt)
        ctitle.append(ct)
        ccat.append(cc)

    out["quality_score"] = scores
    out["issues"] = issues
    out["correction_notes"] = notes

    # “doğru kolon” çıktıları
    out["correct_title"] = ctitle
    out["correct_category"] = ccat
    out["correct_image_url"] = out[image_col] if image_col in out.columns else ""

    return out


def metrics(df_result: pd.DataFrame) -> Dict[str, int]:
    suspicious_titles = int(df_result["issues"].str.contains("Zayıf başlık", na=False).sum())
    missing_images = int(df_result["issues"].str.contains("Görsel eksik", na=False).sum())
    avg_score = int(round(float(df_result["quality_score"].mean()))) if len(df_result) else 0
    return {
        "total": int(len(df_result)),
        "suspicious_titles": suspicious_titles,
        "missing_images": missing_images,
        "avg_score": avg_score,
    }


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analyze")
def analyze_get():
    # kullanıcı yanlışlıkla /analyze yazarsa ana sayfaya dönsün
    return RedirectResponse(url="/", status_code=302)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_post(request: Request, file: UploadFile = File(...)):
    try:
        df = read_uploaded_file(file)
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Dosya okunamadı: {e}"},
            status_code=400,
        )

    df.columns = [str(c).strip() for c in df.columns]
    columns = list(df.columns)

    detected = detect_mapping(columns)

    # Eğer title veya category yoksa eşleştirme ekranı
    if detected["title"] is None or detected["category"] is None:
        token = str(uuid.uuid4())
        DF_STORE[token] = df
        return templates.TemplateResponse(
            "map_columns.html",
            {
                "request": request,
                "token": token,
                "columns": columns,
                "detected": detected,
            },
        )

    # otomatik eşleştiyse direkt analiz
    mapping_token = str(uuid.uuid4())
    result = run_analysis(df, detected["title"], detected["image_url"] or "", detected["category"])
    RESULT_STORE[mapping_token] = result

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "token": mapping_token,
            "summary": metrics(result),
            "columns": list(result.columns),
            "rows": result.head(50).to_dict(orient="records"),
        },
    )


@app.post("/run", response_class=HTMLResponse)
async def run_mapped(
    request: Request,
    token: str = Form(...),
    title_column: str = Form(...),
    category_column: str = Form(...),
    image_column: str = Form(""),
):
    if token not in DF_STORE:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Oturum süresi doldu. Lütfen dosyayı yeniden yükleyin."},
            status_code=400,
        )

    df = DF_STORE.pop(token)
    result = run_analysis(df, title_column, image_column or "", category_column)

    out_token = str(uuid.uuid4())
    RESULT_STORE[out_token] = result

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "token": out_token,
            "summary": metrics(result),
            "columns": list(result.columns),
            "rows": result.head(50).to_dict(orient="records"),
        },
    )


@app.get("/download/{token}")
def download(token: str):
    if token not in RESULT_STORE:
        return RedirectResponse(url="/", status_code=302)

    data = df_to_csv_bytes(RESULT_STORE[token])
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="akas_sonuclar_{token[:8]}.csv"'},
    )
