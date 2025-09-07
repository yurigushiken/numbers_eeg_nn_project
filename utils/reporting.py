import base64
from pathlib import Path
from typing import Dict, List

def generate_html_report(
    run_dir: Path,
    report_title: str,
    txt_report_content: str,
    fold_plot_paths: List[Path],
    overall_plot_path: Path,
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
    report_title = f"Run Report: {task} ({engine})"
    
    # 1. Find all plot files
    fold_plots = sorted(run_dir.glob("fold*_*.png"))
    overall_plot = run_dir / "overall_confusion.png"

    # 2. Read the text report content
    txt_report_path = run_dir / f"report_{task}_{engine}.txt"
    try:
        txt_content = txt_report_path.read_text()
    except FileNotFoundError:
        txt_content = "Text report file not found."

    # 3. Generate and save HTML content
    html_content = generate_html_report(run_dir, report_title, txt_content, fold_plots, overall_plot)
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
