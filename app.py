import base64
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from src.database import init_db
from src.feature_extraction import preprocess_audio, extract_features, analyze_audio_frames
from src.retrieval import build_search_trace, index_folder

app = FastAPI(title="Audio Search UI")

# Resolve DATASET_FOLDER relative to the project root.
_project_root = Path(__file__).parent
_default_dataset = _project_root / "data" / "dataset" / "male_dataset_500"
DATASET_FOLDER = Path(os.getenv("AUDIO_DATASET_FOLDER", str(_default_dataset))).resolve()


def _svg_to_data_uri(svg_markup):
  """Convert SVG markup into a base64 data URI."""
  encoded = base64.b64encode(svg_markup.encode("utf-8")).decode("ascii")
  return f"data:image/svg+xml;base64,{encoded}"


def build_visualizations(file_path):
  """Build waveform and spectrogram visualizations for the query audio."""
  y, sr, _ = preprocess_audio(file_path)
  *_, spectral_matrix = extract_features(y, sr)

  width = 900
  height = 220
  mid_y = height / 2.0
  amplitude = float(np.max(np.abs(y))) or 1.0
  step = max(len(y) // 800, 1)
  sampled = y[::step]
  if sampled.size < 2:
    sampled = np.zeros(2, dtype=np.float32)
  x_points = np.linspace(0, width, num=sampled.size)
  y_points = mid_y - (sampled / amplitude) * (height * 0.42)
  points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_points, y_points))
  waveform_svg = f"""
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
    <rect width="100%" height="100%" fill="#fffaf3" rx="18" />
    <line x1="0" y1="{mid_y}" x2="{width}" y2="{mid_y}" stroke="#d6c4b2" stroke-width="1" />
    <polyline fill="none" stroke="#7a4f2b" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round" points="{points}" />
    <text x="18" y="28" fill="#7a4f2b" font-size="18" font-family="Georgia, serif">Waveform</text>
  </svg>
  """
  waveform_uri = _svg_to_data_uri(waveform_svg)

  if spectral_matrix.size == 0:
    spectrogram_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
      <rect width="100%" height="100%" fill="#fffaf3" rx="18" />
      <text x="18" y="28" fill="#7a4f2b" font-size="18" font-family="Georgia, serif">Spectrogram</text>
      <text x="18" y="62" fill="#6b6258" font-size="14" font-family="Georgia, serif">No spectral data</text>
    </svg>
    """
    spectrogram_uri = _svg_to_data_uri(spectrogram_svg)
  else:
    max_frames = 160
    if spectral_matrix.shape[0] > max_frames:
      indices = np.linspace(0, spectral_matrix.shape[0] - 1, num=max_frames, dtype=int)
      spectral_matrix = spectral_matrix[indices]

    matrix = spectral_matrix.T
    rows, cols = matrix.shape
    plot_x = 18
    plot_y = 40
    plot_width = 850
    plot_height = 150
    cell_w = plot_width / max(cols, 1)
    cell_h = plot_height / max(rows, 1)
    max_value = float(np.max(matrix)) or 1.0

    rects = []
    for row_idx in range(rows):
      for col_idx in range(cols):
        value = matrix[row_idx, col_idx] / max_value
        value = float(np.clip(value, 0.0, 1.0))
        red = int(28 + 198 * value)
        green = int(18 + 62 * value)
        blue = int(42 + 18 * value)
        rects.append(
          f'<rect x="{plot_x + col_idx * cell_w:.2f}" y="{plot_y + (rows - 1 - row_idx) * cell_h:.2f}" '
          f'width="{cell_w + 0.3:.2f}" height="{cell_h + 0.3:.2f}" fill="rgb({red},{green},{blue})" />'
        )

    spectrogram_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
      <rect width="100%" height="100%" fill="#fffaf3" rx="18" />
      <text x="18" y="28" fill="#7a4f2b" font-size="18" font-family="Georgia, serif">Spectrogram</text>
      {''.join(rects)}
      <rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d6c4b2" stroke-width="1" />
    </svg>
    """
    spectrogram_uri = _svg_to_data_uri(spectrogram_svg)

  return waveform_uri, spectrogram_uri


def render_page(title, message="", trace=None):
    trace = trace or {}
    query_summary = trace.get("query_summary") or {}
    query_audio_data_uri = trace.get("query_audio_data_uri") or ""
    waveform_image_uri = trace.get("waveform_image_uri") or ""
    spectrogram_image_uri = trace.get("spectrogram_image_uri") or ""
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
            <div><span>Spectral bandwidth</span><strong>{query_summary['spectral_bandwidth']:.6f}</strong></div>
            <div><span>Feature vector</span><strong>{query_summary['feature_vector_dim']} chiều</strong></div>
            <div><span>Spectral matrix shape</span><strong>{query_summary['spectral_matrix_shape']}</strong></div>
            <div><span>Frame count</span><strong>{query_summary['frame_count']}</strong></div>
            <div><span>Frame energy mean</span><strong>{query_summary['frame_energy_mean']:.6f}</strong></div>
            <div><span>Frame ZCR mean</span><strong>{query_summary['frame_zcr_mean']:.6f}</strong></div>
            <div><span>Frame silent ratio</span><strong>{query_summary['frame_silent_ratio']:.6f}</strong></div>
          </div>
        </section>
        """

    visual_html = ""
    if waveform_image_uri or spectrogram_image_uri:
        visual_html = f"""
        <section class="card">
          <h2>Kết quả trung gian trực quan</h2>
          <div class="viz-grid">
            <div>
              <h3>Waveform</h3>
              {f'<img src="{waveform_image_uri}" alt="Waveform" />' if waveform_image_uri else '<p class="empty">Chưa có waveform</p>'}
            </div>
            <div>
              <h3>Spectrogram</h3>
              {f'<img src="{spectrogram_image_uri}" alt="Spectrogram" />' if spectrogram_image_uri else '<p class="empty">Chưa có spectrogram</p>'}
            </div>
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
            .viz-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
            .viz-grid h3 {{ margin: 0 0 8px; font-size: 1rem; color: var(--accent); }}
            .viz-grid img {{ width: 100%; border-radius: 12px; border: 1px solid var(--border); background: #fff; }}
            .footer-note {{ color: var(--muted); font-size: 0.92rem; margin-top: 10px; }}
            @media (max-width: 920px) {{
              .hero {{ grid-template-columns: 1fr; }}
              .grid-3, .summary-grid, .viz-grid {{ grid-template-columns: 1fr; }}
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
                    <input id="folder_path" name="folder_path" type="text" value="data/dataset/male_dataset_500" />
                    <div class="actions">
                      <button type="submit">Index vào CSDL</button>
                    </div>
                  </form>
                </section>
                <section class="card">
                  <h2>Tìm kiếm file âm thanh</h2>
                  <form method="post" action="/search" enctype="multipart/form-data">
                    <label for="query_file">File truy vấn</label>
                    <input id="query_file" name="query_file" type="file" accept=".wav,audio/wav" required />
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
            {visual_html}

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
        audio_path = (DATASET_FOLDER / file_name).resolve()
        audio_path.relative_to(DATASET_FOLDER)
        if not audio_path.exists() or not audio_path.is_file():
            return HTMLResponse("File not found", status_code=404)
        return FileResponse(audio_path, media_type="audio/wav")
    except (ValueError, FileNotFoundError):
        return HTMLResponse("File not found", status_code=404)


@app.post("/index", response_class=HTMLResponse)
def index_data(folder_path: str = Form(default="data/dataset/male_dataset_500")):
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
      try:
        waveform_image_uri, spectrogram_image_uri = build_visualizations(temp_path)
        y, sr, _ = preprocess_audio(temp_path)
        frame_stats = analyze_audio_frames(y, sr)
        trace = build_search_trace(
          query_file_path=temp_path,
          metric=metric,
          top_k=top_k,
          dtw_candidate_pool=dtw_candidate_pool,
        )
      except sf.LibsndfileError:
        return render_page(
          "Audio Search UI",
          message="File tải lên chưa được hỗ trợ. Hãy dùng file WAV (16 kHz/mono nếu có thể).",
        )
      return render_page(
        "Audio Search UI",
        message=f"Đã tìm kiếm với metric={metric}, top_k={top_k}.",
        trace={
          **trace,
          "query_summary": {**trace.get("query_summary", {}), **frame_stats},
          "query_audio_data_uri": query_audio_data_uri,
          "waveform_image_uri": waveform_image_uri,
          "spectrogram_image_uri": spectrogram_image_uri,
        },
      )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
