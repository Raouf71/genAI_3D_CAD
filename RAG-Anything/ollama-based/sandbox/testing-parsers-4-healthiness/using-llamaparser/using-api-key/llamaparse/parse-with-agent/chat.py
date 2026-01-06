"""
PoC: Parse a PDF (text + tables + images) with LlamaParse (agentic mode).
Requirements:
  pip install -U llama-cloud-services
Env:
  export LLAMA_CLOUD_API_KEY="..."
Usage:
  python parse_pdf_poc.py /path/to/file.pdf
Outputs:
  - ./out/<stem>.md
  - ./out/<stem>_metadata.json
  - ./out/<stem>_images/ (if images are returned)
"""

import os
import sys
import json
import base64
from pathlib import Path

from llama_cloud_services import LlamaParse


def _write_images(images, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, img in enumerate(images or []):
        # Try common payload shapes
        b64 = img.get("base64") or img.get("b64") or img.get("data")
        mime = (img.get("mime_type") or img.get("mime") or "image/png").lower()
        ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/webp": "webp",
        }.get(mime, "bin")

        if not b64:
            continue

        raw = base64.b64decode(b64)
        p = out_dir / f"image_{i:03d}.{ext}"
        p.write_bytes(raw)
        saved.append(str(p))
    return saved

api_key="llx-WROpF69GBXDRmw9jP8oiN1lwU6iDTbJs0kRY0nj3ReVfXtuY"
os.environ["LLAMA_CLOUD_API_KEY"] = api_key

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python parse_pdf_poc.py /path/to/file.pdf")

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_PARSE_API_KEY")
    if not api_key:
        raise SystemExit("Set LLAMA_CLOUD_API_KEY (or LLAMA_PARSE_API_KEY).")

    out_root = Path("out")
    out_root.mkdir(parents=True, exist_ok=True)

    parser = LlamaParse(
        api_key=api_key,
        parse_mode="parse_page_with_agent",
        # model="openai-gpt-4-1-mini",
        model="anthropic-sonnet-4.0",
        result_type="markdown",
        language="de",
        high_res_ocr=True,
        adaptive_long_table=True,
        outlined_table_extraction=True,
        output_tables_as_HTML=True,
        extract_layout=True,
        take_screenshot=True,
        # Invalid args
        extract_images=True,
        include_image_data=True,
    )
    print("================================= Agentic Plus Mode Parser initialized")

    # Parse
    result = parser.parse(str(pdf_path))

    # Robustly extract markdown + metadata across possible result shapes
    markdown = None
    metadata = {}
    images = []

    if isinstance(result, str):
        markdown = result
    elif isinstance(result, dict):
        markdown = result.get("markdown") or result.get("text") or result.get("content")
        metadata = result.get("metadata") or {k: v for k, v in result.items() if k not in {"markdown", "text", "content"}}
        images = (
            result.get("images")
            or result.get("extracted_images")
            or (result.get("metadata", {}) or {}).get("images")
            or []
        )
    else:
        # object-like
        markdown = getattr(result, "markdown", None) or getattr(result, "text", None) or getattr(result, "content", None)
        metadata = getattr(result, "metadata", {}) or {}
        images = getattr(result, "images", None) or getattr(result, "extracted_images", None) or metadata.get("images", []) or []

    if markdown is None:
        markdown = str(result)

    stem = pdf_path.stem
    md_path = out_root / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    meta_path = out_root / f"{stem}_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    img_dir = out_root / f"{stem}_images"
    saved_imgs = _write_images(images, img_dir)

    if saved_imgs:
        with md_path.open("a", encoding="utf-8") as f:
            f.write("\n\n---\n\n## Extracted images\n")
            for p in saved_imgs:
                rel = Path(p).relative_to(out_root)
                f.write(f"\n- {rel.as_posix()}\n")


if __name__ == "__main__":
    main()
