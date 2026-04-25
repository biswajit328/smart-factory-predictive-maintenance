from __future__ import annotations

import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT_DIR / "site"
SITE_ASSETS_DIR = SITE_DIR / "assets"


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_site() -> Path:
    SITE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(ROOT_DIR / "assets" / "project-preview.png", SITE_ASSETS_DIR / "project-preview.png")
    _copy_if_exists(ROOT_DIR / "assets" / "api-contract.png", SITE_ASSETS_DIR / "api-contract.png")
    _copy_if_exists(ROOT_DIR / "assets" / "demo-walkthrough.gif", SITE_ASSETS_DIR / "demo-walkthrough.gif")
    _copy_if_exists(ROOT_DIR / "outputs" / "v2" / "smart_factory_dashboard.html", SITE_DIR / "dashboard.html")

    index_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Smart Factory Predictive Maintenance Demo</title>
  <style>
    body {
      margin: 0;
      background: #f6f7f2;
      color: #17211a;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
      padding: 42px 20px 64px;
    }
    h1 {
      font-size: clamp(2rem, 4vw, 4.2rem);
      line-height: 1.02;
      margin: 0 0 12px;
    }
    p {
      max-width: 820px;
      font-size: 1.05rem;
      line-height: 1.65;
      color: #425249;
    }
    a {
      color: #176b4d;
      font-weight: 800;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 24px 0 32px;
    }
    .actions a {
      border: 1px solid #176b4d;
      padding: 10px 14px;
      text-decoration: none;
    }
    img {
      width: 100%;
      display: block;
      border: 1px solid #d7ddd2;
      margin-top: 18px;
      background: white;
    }
  </style>
</head>
<body>
  <main>
    <h1>Smart Factory Predictive Maintenance Demo</h1>
    <p>
      A compact hosted view of the project artifacts: neural sensor-fusion evidence,
      API surface, and the generated dashboard from the simulated factory stream.
    </p>
    <div class="actions">
      <a href="dashboard.html">Open dashboard artifact</a>
      <a href="https://github.com/biswajit328/smart-factory-predictive-maintenance">View source</a>
    </div>
    <img src="assets/project-preview.png" alt="Model evidence preview" />
    <img src="assets/api-contract.png" alt="FastAPI contract preview" />
  </main>
</body>
</html>
"""
    (SITE_DIR / "index.html").write_text(index_html, encoding="utf-8")
    return SITE_DIR / "index.html"


def main() -> None:
    output = build_site()
    print(f"Built static demo site at {output}")


if __name__ == "__main__":
    main()
