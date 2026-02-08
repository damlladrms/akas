from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import io
import uuid


app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Geçici bellek (Render restart olursa sıfırlanır)
RAW_STORE: dict[str, pd.DataFrame] = {}      # kolon eşleştirme için ham veri
RESULT_STORE: dict[str, pd.DataFrame] = {}   # analiz sonuçları


def read_uploaded_file(file: UploadFile, content: bytes) -> pd.DataFrame:
    filename = (file.filename or "").lower().strip()

    # Uzantı varsa ona göre oku
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if filename.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))

    # Uzantı yoksa: önce CSV dene, olmazsa Excel dene
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        try:
            return pd.read_excel(io.BytesIO(content))
        except Exception:
            raise ValueError("Sadece CSV veya Excel (.xlsx) dosyaları desteklenmektedir.")


def run_analysis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- 1) Yazım / şüpheli tekrar (Monttt, Maxxx gibi) ---
    if "title" in out.columns:
        title_series = out["title"].fillna("").astype(str)

        out["spelling_flag"] = title_series.str.contains(r"(.)\1\1", regex=True)
        out["spelling_note"] = out["spelling_flag"].apply(lambda x: "Şüpheli tekrar" if x else "")

        # ÖNERİ: 3+ tekrarları 2'ye indir (Monttt->Montt, Maxxx->Maxx)
        out["title_suggestion"] = title_series.str.replace(r"(.)\1\1+", r"\1\1", regex=True)
        out.loc[out["spelling_flag"] == False, "title_suggestion"] = ""
    else:
        out["spelling_flag"] = False
        out["spelling_note"] = "title kolonu yok"
        out["title_suggestion"] = ""

    # --- 2) Görsel eksikliği ---
    if "image_url" in out.columns:
        s = out["image_url"].fillna("").astype(str).str.strip()
        out["image_missing"] = (s == "") | (s.str.lower().isin(["nan", "none"]))
        out["image_note"] = out["image_missing"].apply(lambda x: "Görsel yok/boş" if x else "")
    else:
        out["image_missing"] = True
        out["image_note"] = "image_url kolonu yok"

    out["image_suggestion"] = out["image_missing"].apply(
        lambda x: "Ürüne ait net bir görsel eklenmeli" if x else ""
    )

    # --- 3) Kalite skoru (MVP) ---
    out["quality_score"] = 90
    out.loc[out["spelling_flag"] == True, "quality_score"] -= 10
    out.loc[out["image_missing"] == True, "quality_score"] -= 20
    out["quality_score"] = out["quality_score"].clip(lower=0, upper=100)

    return out


def build_summary(out_df: pd.DataFrame) -> dict:
    # güvenli çekimler
    spelling_cnt = int(out_df["spelling_flag"].sum()) if "spelling_flag" in out_df.columns else 0
    image_missing_cnt = int(out_df["image_missing"].sum()) if "image_missing" in out_df.columns else 0
    avg_score = float(out_df["quality_score"].mean()) if "quality_score" in out_df.columns else 0.0

    return {
        "row_count": int(len(out_df)),
        "spelling_cnt": spelling_cnt,
        "image_missing_cnt": image_missing_cnt,
        "avg_score": round(avg_score, 1),
    }


def apply_mapping(df: pd.DataFrame, title_col: str, image_col: str, category_col: str | None, sub_category_col: str | None) -> pd.DataFrame:
    mapped = df.copy()

    rename_map = {}

    if title_col and title_col != "__none__":
        rename_map[title_col] = "title"
    if image_col and image_col != "__none__":
        rename_map[image_col] = "image_url"
    if category_col and category_col != "__none__":
        rename_map[category_col] = "category"
    if sub_category_col and sub_category_col != "__none__":
        rename_map[sub_category_col] = "sub_category"

    mapped = mapped.rename(columns=rename_map)

    # Eksik zorunlu kolonlar varsa bile analiz çalışsın diye garanti kolon açalım
    if "title" not in mapped.columns:
        mapped["title"] = ""
    if "image_url" not in mapped.columns:
        mapped["image_url"] = ""

    return mapped


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    content = await file.read()

    try:
        df = read_uploaded_file(file, content)
    except ValueError as ve:
        return HTMLResponse(str(ve), status_code=400)
    except Exception as e:
        return HTMLResponse(f"Dosya okunamadı: {e}", status_code=400)

    # Eğer standart kolonlar zaten varsa direkt analiz et
    cols = set([c.strip() for c in df.columns.astype(str)])
    has_title = "title" in cols
    has_image = "image_url" in cols

    if has_title and has_image:
        out_df = run_analysis(df)
        job_id = str(uuid.uuid4())
        RESULT_STORE[job_id] = out_df

        summary = build_summary(out_df)
        preview = out_df.head(200).to_dict(orient="records")
        columns = list(out_df.columns)

        return templates.TemplateResponse(
            "results.html",
            {"request": request, "job_id": job_id, "columns": columns, "rows": preview, "summary": summary},
        )

    # Yoksa kolon eşleştirme ekranına gönder
    upload_id = str(uuid.uuid4())
    RAW_STORE[upload_id] = df

    all_columns = [str(c) for c in df.columns.tolist()]
    # dropdown için ayrıca bir "seçmedim" opsiyonu
    options = ["__none__"] + all_columns

    return templates.TemplateResponse(
        "map_columns.html",
        {
            "request": request,
            "upload_id": upload_id,
            "options": options,
            "all_columns": all_columns,
        },
    )


@app.post("/analyze_mapped", response_class=HTMLResponse)
async def analyze_mapped(
    request: Request,
    upload_id: str = Form(...),
    title_col: str = Form("__none__"),
    image_col: str = Form("__none__"),
    category_col: str = Form("__none__"),
    sub_category_col: str = Form("__none__"),
):
    if upload_id not in RAW_STORE:
        return HTMLResponse("Yükleme bulunamadı. Lütfen yeniden dosya yükleyin.", status_code=404)

    df = RAW_STORE[upload_id]

    # eşleştir -> analiz
    mapped_df = apply_mapping(df, title_col, image_col, category_col, sub_category_col)
    out_df = run_analysis(mapped_df)

    job_id = str(uuid.uuid4())
    RESULT_STORE[job_id] = out_df

    summary = build_summary(out_df)
    preview = out_df.head(200).to_dict(orient="records")
    columns = list(out_df.columns)

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "job_id": job_id, "columns": columns, "rows": preview, "summary": summary},
    )


@app.get("/download/{job_id}")
def download(job_id: str):
    if job_id not in RESULT_STORE:
        return HTMLResponse("Bulunamadı", status_code=404)

    df = RESULT_STORE[job_id]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    output.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="akilli_katalog_sonuc.xlsx"'}
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )

