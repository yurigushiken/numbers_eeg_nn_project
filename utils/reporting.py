import base64
import re
from pathlib import Path
from typing import Dict, List

def generate_html_report(
    run_dir: Path,
    report_title: str,
    txt_report_content: str,
    fold_plot_paths: List[Path],
    overall_plot_path: Path,
    banner_html: str = "",
) -> str:
    """Generates a self-contained HTML report from run artifacts."""

    def embed_image(image_path: Path) -> str:
        """Reads an image and returns a base64 encoded string for embedding."""
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
        except FileNotFoundError:
            return "Image not found"

    # --- HTML and CSS Template ---
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{report_title}</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            h1, h2 {{ text-align: center; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .report-text, .overall-plot {{ page-break-inside: avoid; }}
            .plots-section {{ page-break-before: always; }}
            .plots-grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }}
            .plot-pair {{ text-align: center; margin-bottom: 20px; page-break-inside: avoid; }}
            .plot-pair img {{ max-width: 300px; border: 1px solid #ccc; }}
            .overall-plot {{ text-align: center; margin-top: 40px; }}
            .overall-plot img {{ max-width: 700px; border: 1px solid #ccc; }}
            .report-text {{ white-space: pre-wrap; font-family: monospace; background-color: #f4f4f4;
                           padding: 1em; border: 1px solid #ddd; margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{report_title}</h1>
            
            {banner_html}
            <div class="report-text">
                <h2>Summary Report</h2>
                <pre>{txt_report_content}</pre>
            </div>

            <div class="overall-plot">
                <h2>Overall Confusion Matrix</h2>
                <img src="{embed_image(overall_plot_path)}" alt="Overall Confusion Matrix">
            </div>

            <div class="plots-section">
                <h2>Per-Fold Results</h2>
                <div class="plots-grid">
                    {"".join(f'''
                    <div class="plot-pair">
                        <img src="{embed_image(path)}" alt="{path.name}">
                        <p>{path.stem.replace("_", " ").title()}</p>
                    </div>
                    ''' for path in fold_plot_paths)}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template


def create_consolidated_reports(run_dir: Path, summary: Dict, task: str, engine: str):
    """
    Main function to generate an HTML report and then attempt to generate a 
    PDF report using the modern Playwright library.
    """
    # Presentation-only: show hyphens between digits in task name (e.g., 1_3 -> 1-3)
    task_title = re.sub(r"(?<=\d)_(?=\d)", "-", task)
    report_title = f"Run Report: {task_title} ({engine})"
    # NEW: Subtitle – if task module exposes CONDITIONS, render them as [x, y, ...]
    subtitle_html = ""
    try:
        import importlib
        task_module = importlib.import_module(f"tasks.{task}")
        conds = getattr(task_module, "CONDITIONS", None)
        if isinstance(conds, (list, tuple)) and len(conds) > 0:
            subtitle_html = f"<div class=\"subtitle\"><pre><strong>Conditions: {str(list(conds))}</strong></pre></div>"
    except Exception:
        pass
    
    # 1. Find all plot files
    fold_plots = sorted(run_dir.glob("fold*_*.png"))
    overall_plot = run_dir / "overall_confusion.png"

    # 2. Read the text report content
    txt_report_path = run_dir / f"report_{task}_{engine}.txt"
    try:
        txt_content = txt_report_path.read_text()
    except FileNotFoundError:
        txt_content = "Text report file not found."

    # 2b. If crop_ms / include_channels are present, prepend banner lines
    try:
        hyper = summary.get("hyper", {}) if isinstance(summary, dict) else {}
        crop = hyper.get("crop_ms")
        banners = []
        if isinstance(crop, (list, tuple)) and len(crop) == 2:
            banners.append(f"Time Window (crop_ms): {int(crop[0])}-{int(crop[1])} ms")
        incl = hyper.get("include_channels")
        if isinstance(incl, (list, tuple)) and len(incl) > 0:
            banners.append(f"Included Channels ({len(incl)}): {', '.join(map(str, incl))}")
        if banners:
            banner_text = "\n".join(banners) + "\n\n"
            txt_content = banner_text + txt_content
    except Exception:
        # Do not fail report generation due to optional banner
        pass

    # 3. Generate and save HTML content
    # Build visible HTML banners at the top (mirrors the TXT banners)
    banner_lines = []
    try:
        hyper = summary.get("hyper", {}) if isinstance(summary, dict) else {}
        crop = hyper.get("crop_ms")
        if isinstance(crop, (list, tuple)) and len(crop) == 2:
            banner_lines.append(f"<div class=\"crop-banner\"><pre><strong>Time Window (crop_ms): {int(crop[0])}-{int(crop[1])} ms</strong></pre></div>")
        incl = hyper.get("include_channels")
        if isinstance(incl, (list, tuple)) and len(incl) > 0:
            banner_lines.append(f"<div class=\"include-banner\"><pre><strong>Included Channels ({len(incl)}): {', '.join(map(str, incl))}</strong></pre></div>")
    except Exception:
        pass
    banner_html = "\n".join(banner_lines)

    banner_with_subtitle = "\n".join([s for s in [subtitle_html, banner_html] if s])
    html_content = generate_html_report(run_dir, report_title, txt_content, fold_plots, overall_plot, banner_html=banner_with_subtitle)
    html_output_path = run_dir / "consolidated_report.html"
    html_output_path.write_text(html_content, encoding='utf-8')
    print(f" -> Consolidated HTML report saved to {html_output_path}")

    # --- Best-effort PDF Generation via Playwright ---
    pdf_output_path = run_dir / "consolidated_report.pdf"
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            # Navigate to the local HTML file using a file:// URL
            page.goto(f"file://{html_output_path.resolve()}")
            # Generate the PDF
            page.pdf(path=str(pdf_output_path), format='A4', print_background=True)
            browser.close()
        
        print(f" -> Consolidated PDF report saved via Playwright to {pdf_output_path}")

    except ImportError:
        print(" -> Skipping PDF generation. To enable, run: pip install playwright && playwright install")
    except Exception as e:
        print(f" -> Playwright failed to generate PDF. Error: {e}")
        print(" -> To enable PDF generation, run: pip install playwright && playwright install")
