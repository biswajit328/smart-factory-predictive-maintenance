from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .config import DASHBOARD_HOST, DASHBOARD_PORT, ROOT_DIR, V2_DASHBOARD_PATH
from .logging_utils import configure_logging, get_logger


logger = get_logger(__name__)
ASSETS_DIR = ROOT_DIR / "assets"


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f2;
      color: #17211a;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 20px;
    }}
    a {{ color: #176b4d; font-weight: 700; }}
    img {{ width: 100%; border: 1px solid #d7ddd2; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      margin-top: 28px;
    }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Factory Dashboard Service",
        version="0.1.0",
        description="Static dashboard service for predictive maintenance demo artifacts.",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "dashboard_exists": Path(V2_DASHBOARD_PATH).exists(),
            "assets_dir_exists": ASSETS_DIR.exists(),
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        body = """
        <h1>Smart Factory Predictive Maintenance Demo</h1>
        <p>This service hosts the generated dashboard and README preview assets from the latest local run.</p>
        <p><a href="/dashboard">Open the interactive dashboard</a></p>
        <div class="grid">
          <img src="/assets/project-preview.png" alt="Project preview" />
          <img src="/assets/api-contract.png" alt="API contract preview" />
        </div>
        """
        return HTMLResponse(_page("Smart Factory Demo", body))

    @app.get("/dashboard")
    def dashboard() -> FileResponse:
        dashboard_path = Path(V2_DASHBOARD_PATH)
        if not dashboard_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Dashboard artifact not found. Run `python -m src.v2_inference --live-demo` first.",
            )
        return FileResponse(dashboard_path)

    @app.get("/assets/{filename}")
    def asset(filename: str) -> FileResponse:
        path = ASSETS_DIR / filename
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found.")
        return FileResponse(path)

    return app


app = create_app()


def main() -> None:
    configure_logging()
    logger.info("starting_dashboard", extra={"host": DASHBOARD_HOST, "port": DASHBOARD_PORT})
    uvicorn.run("src.v2_dashboard_server:app", host=DASHBOARD_HOST, port=DASHBOARD_PORT, reload=False)


if __name__ == "__main__":
    main()
