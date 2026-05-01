import base64
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from audio_search.database import init_db
from audio_search.search_engine import build_search_trace, index_folder

app = FastAPI(title="Audio Search UI")

# Resolve DATASET_FOLDER relative to app location
_audio_search_dir = Path(__file__).parent.parent  # /Users/tiendat/CSDLDPT
_default_dataset = _audio_search_dir / "male_dataset_500"
DATASET_FOLDER = Path(os.getenv("AUDIO_DATASET_FOLDER", str(_default_dataset))).resolve()


def render_page(title, message="", trace=None):
    trace = trace or {}
    query_summary = trace.get("query_summary") or {}
    query_audio_data_uri = trace.get("query_audio_data_uri") or ""
    vector_candidates = trace.get("vector_candidates") or []
    final_results = trace.get("final_results") or []

    def build_rows(items):
        if not items:
            return '<tr><td colspan="5" class="empty">Chưa có dữ liệu</td></tr>'

        rows = []
        for idx, item in enumerate(items, start=1):
            audio_src = item.get("audio_src", "")
            audio_cell = (
                f'<audio controls preload="none" src="{audio_src}" style="width: 180px;"></audio>'
                if audio_src
                else "-"
            )
            rows.append(
                "<tr>"
                f"<td>{idx}</td>"
                f"<td>{item['file_name']}</td>"
                f"<td>{item['similarity']:.6f}</td>"
                f"<td>{item['distance']:.6f}</td>"
                f"<td>{audio_cell}</td>"
                "</tr>"
            )
        return "".join(rows)

    summary_html = ""
    if query_summary:
        summary_html = f"""
        <section class="card">
          <h2>Thông tin file truy vấn</h2>
          <div class="query-player">{f'<audio controls src="{query_audio_data_uri}" style="width:100%;"></audio>' if query_audio_data_uri else ''}</div>
          <div class="summary-grid">
            <div><span>Tệp</span><strong>{query_summary['file_name']}</strong></div>
            <div><span>Sample rate</span><strong>{query_summary['sample_rate']} Hz</strong></div>
            <div><span>Độ dài chuẩn hóa</span><strong>{query_summary['duration_sec']} s</strong></div>
            <div><span>Silence ratio</span><strong>{query_summary['silence_ratio']:.6f}</strong></div>
            <div><span>Energy</span><strong>{query_summary['energy']:.6f}</strong></div>
            <div><span>ZCR</span><strong>{query_summary['zcr']:.6f}</strong></div>
            <div><span>Pitch mean</span><strong>{query_summary['pitch_mean']:.6f}</strong></div>
            <div><span>Spectral centroid</span><strong>{query_summary['spectral_centroid']:.6f}</strong></div>
            <div><span>Feature vector</span><strong>{query_summary['feature_vector_dim']} chiều</strong></div>
            <div><span>MFCC shape</span><strong>{query_summary['mfcc_matrix_shape']}</strong></div>
          </div>
        </section>
        """

    return HTMLResponse(
        f"""
        <!doctype html>
        <html lang="vi">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>{title}</title>
          <style>
            :root {{
              --bg: #f5f1ea;
              --panel: #fffaf3;
              --ink: #1e1a17;
              --muted: #6b6258;
              --accent: #7a4f2b;
              --accent-2: #a66a3f;
              --border: rgba(30, 26, 23, 0.12);
              --shadow: 0 20px 40px rgba(76, 48, 27, 0.12);
            }}
            * {{ box-sizing: border-box; }}
            body {{
              margin: 0;
              font-family: Georgia, "Times New Roman", serif;
              color: var(--ink);
              background:
                radial-gradient(circle at top left, rgba(166, 106, 63, 0.16), transparent 28%),
                radial-gradient(circle at right 15%, rgba(122, 79, 43, 0.12), transparent 24%),
                linear-gradient(180deg, #fbf7f1 0%, #f2eadf 100%);
            }}
            .shell {{ max-width: 1180px; margin: 0 auto; padding: 40px 20px 56px; }}
            .hero {{
              display: grid;
              grid-template-columns: 1.3fr 0.9fr;
              gap: 24px;
              align-items: stretch;
              margin-bottom: 24px;
            }}
            .hero-panel, .card {{
              background: rgba(255, 250, 243, 0.92);
              border: 1px solid var(--border);
              border-radius: 24px;
              box-shadow: var(--shadow);
              backdrop-filter: blur(8px);
            }}
            .hero-panel {{ padding: 30px; }}
            .hero h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 4vw, 3.6rem); line-height: 0.98; }}
            .hero p {{ margin: 0; color: var(--muted); font-size: 1.02rem; line-height: 1.6; max-width: 60ch; }}
            .pill {{
              display: inline-flex; align-items: center; gap: 10px;
              padding: 10px 14px; border-radius: 999px; background: rgba(122, 79, 43, 0.08);
              color: var(--accent); font-size: 0.95rem; margin-bottom: 18px;
            }}
            .stack {{ display: grid; gap: 16px; }}
            .card {{ padding: 22px; }}
            .card h2 {{ margin: 0 0 16px; font-size: 1.2rem; }}
            label {{ display: block; font-size: 0.92rem; color: var(--muted); margin-bottom: 8px; }}
            input[type="text"], input[type="number"], select, input[type="file"] {{
              width: 100%; padding: 12px 14px; border: 1px solid var(--border); border-radius: 14px;
              background: white; font: inherit; color: var(--ink);
            }}
            .grid-3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
            .actions {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }}
            button {{
              border: 0; border-radius: 14px; padding: 12px 18px; background: linear-gradient(135deg, var(--accent), var(--accent-2));
              color: white; font: inherit; font-weight: 700; cursor: pointer;
            }}
            .secondary {{ background: #e9ddd1; color: var(--ink); }}
            .message {{
              padding: 14px 16px; border-radius: 14px; background: rgba(122, 79, 43, 0.08);
              color: var(--accent); margin-bottom: 16px; border: 1px solid rgba(122, 79, 43, 0.16);
            }}
            .tables {{ display: grid; gap: 18px; margin-top: 18px; }}
            table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 14px; }}
            th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
            th {{ background: rgba(122, 79, 43, 0.08); color: var(--accent); }}
            tr:last-child td {{ border-bottom: 0; }}
            .empty {{ color: var(--muted); text-align: center; padding: 28px; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
            .summary-grid div {{ padding: 12px 14px; border: 1px solid var(--border); border-radius: 14px; background: white; }}
            .summary-grid span {{ display: block; color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }}
            .summary-grid strong {{ font-size: 1rem; }}
            .query-player {{ margin-bottom: 14px; }}
            .footer-note {{ color: var(--muted); font-size: 0.92rem; margin-top: 10px; }}
            @media (max-width: 920px) {{
              .hero {{ grid-template-columns: 1fr; }}
              .grid-3, .summary-grid {{ grid-template-columns: 1fr; }}
            }}
          </style>
        </head>
        <body>
          <div class="shell">
            <section class="hero">
              <div class="hero-panel">
                <div class="pill">Hệ CSDL lưu trữ và tìm kiếm giọng nói đàn ông</div>
                <h1>Giao diện cơ bản để nạp dữ liệu, tìm kiếm và xem kết quả trung gian.</h1>
                <p>
                  Tải lên một file âm thanh nam giới mới để tìm 5 file giống nhất, hoặc index lại bộ dữ liệu WAV có sẵn.
                  Giao diện này dùng trực tiếp pipeline trích xuất đặc trưng, pgvector và DTW re-ranking của hệ thống hiện tại.
                </p>
                <p class="footer-note">Chế độ mặc định: WAV, chuẩn hóa 5 giây, 16 kHz, mono.</p>
              </div>
              <div class="stack">
                <section class="card">
                  <h2>Index dữ liệu</h2>
                  <form method="post" action="/index">
                    <label for="folder_path">Thư mục dữ liệu WAV</label>
                    <input id="folder_path" name="folder_path" type="text" value="../male_dataset_500" />
                    <div class="actions">
                      <button type="submit">Index vào CSDL</button>
                    </div>
                  </form>
                </section>
                <section class="card">
                  <h2>Tìm kiếm file âm thanh</h2>
                  <form method="post" action="/search" enctype="multipart/form-data">
                    <label for="query_file">File truy vấn</label>
                    <input id="query_file" name="query_file" type="file" accept="audio/*" required />
                    <div class="grid-3" style="margin-top: 14px;">
                      <div>
                        <label for="metric">Metric</label>
                        <select id="metric" name="metric">
                          <option value="cosine">Cosine</option>
                          <option value="euclidean">Euclidean</option>
                          <option value="dtw">DTW</option>
                        </select>
                      </div>
                      <div>
                        <label for="top_k">Top-K</label>
                        <input id="top_k" name="top_k" type="number" min="1" max="20" value="5" />
                      </div>
                      <div>
                        <label for="dtw_candidate_pool">DTW pool</label>
                        <input id="dtw_candidate_pool" name="dtw_candidate_pool" type="number" min="5" max="100" value="30" />
                      </div>
                    </div>
                    <div class="actions">
                      <button type="submit">Tìm kiếm</button>
                      <button type="reset" class="secondary">Nhập lại</button>
                    </div>
                  </form>
                </section>
              </div>
            </section>

            {f'<div class="message">{message}</div>' if message else ''}
            {summary_html}

            <div class="tables">
              <section class="card">
                <h2>Các ứng viên trung gian</h2>
                <table>
                  <thead><tr><th>#</th><th>File</th><th>Similarity</th><th>Distance</th><th>Nghe</th></tr></thead>
                  <tbody>{build_rows(vector_candidates)}</tbody>
                </table>
              </section>
              <section class="card">
                <h2>Kết quả cuối cùng</h2>
                <table>
                  <thead><tr><th>#</th><th>File</th><th>Similarity</th><th>Distance</th><th>Nghe</th></tr></thead>
                  <tbody>{build_rows(final_results)}</tbody>
                </table>
              </section>
            </div>
          </div>
        </body>
        </html>
        """,
    )


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
def home():
    return render_page("Audio Search UI")


@app.get("/audio/{file_name:path}")
def serve_audio(file_name: str):
    try:
        # Construct the full path and resolve it
        audio_path = (DATASET_FOLDER / file_name).resolve()
        
        # Security check: ensure the resolved path is within DATASET_FOLDER
        audio_path.relative_to(DATASET_FOLDER)
        
        # Check if file exists
        if not audio_path.exists() or not audio_path.is_file():
            return HTMLResponse("File not found", status_code=404)
        
        return FileResponse(audio_path, media_type="audio/wav")
    except (ValueError, FileNotFoundError):
        # ValueError: audio_path is not relative to DATASET_FOLDER (security issue)
        # FileNotFoundError: path doesn't exist
        return HTMLResponse("File not found", status_code=404)


@app.post("/index", response_class=HTMLResponse)
def index_data(folder_path: str = Form(default="../male_dataset_500")):
    init_db()
    index_folder(folder_path)
    return render_page("Audio Search UI", message=f"Đã index dữ liệu từ {folder_path}.")


@app.post("/search", response_class=HTMLResponse)
async def search_audio(
    query_file: UploadFile = File(...),
    metric: str = Form(default="cosine"),
    top_k: int = Form(default=5),
    dtw_candidate_pool: int = Form(default=30),
):
    init_db()
    suffix = Path(query_file.filename or "query.wav").suffix or ".wav"
    file_bytes = await query_file.read()
    query_audio_data_uri = f"data:{query_file.content_type or 'audio/wav'};base64,{base64.b64encode(file_bytes).decode('ascii')}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        temp_path = tmp_file.name

    try:
        trace = build_search_trace(
            query_file_path=temp_path,
            metric=metric,
            top_k=top_k,
            dtw_candidate_pool=dtw_candidate_pool,
        )
        return render_page(
            "Audio Search UI",
            message=f"Đã tìm kiếm với metric={metric}, top_k={top_k}.",
            trace={**trace, "query_audio_data_uri": query_audio_data_uri},
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
