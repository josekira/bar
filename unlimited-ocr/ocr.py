#!/usr/bin/env python3
"""Turn a PDF, an image, or a folder of page images into Markdown with Unlimited-OCR 2.

Needs a running server (bash start_server.sh). All pages go out in one request, so the
model can join paragraphs and tables that continue across pages.

Examples:
  python ocr.py report.pdf                    # Markdown to the screen
  python ocr.py report.pdf -o report.md       # ... or to a file
  python ocr.py scan.jpg -o scan.md           # a single image
  python ocr.py pages/ -o book.md             # a folder of page images, in name order
  python ocr.py report.pdf --raw report.txt   # also keep the model's raw layout output
  python ocr.py report.pdf --server http://gpu-box:10000
"""
import argparse
import base64
import json
import os
import re
import sys
import tempfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'Unlimited-OCR', 'utils'))
from process_unlimitedOCR2 import convert  # noqa: E402  (Baidu's raw-output -> Markdown converter)
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor  # noqa: E402

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def collect_pages(path, dpi):
    """Return page image paths in reading order."""
    if os.path.isdir(path):
        pages = sorted((f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS)), key=natural_key)
        if not pages:
            sys.exit(f'no images ({", ".join(IMAGE_EXTS)}) in {path}')
        return [os.path.join(path, f) for f in pages]
    if not os.path.isfile(path):
        sys.exit(f'not found: {path}')
    if path.lower().endswith('.pdf'):
        import fitz  # PyMuPDF
        out = tempfile.mkdtemp(prefix='uocr_')
        zoom = fitz.Matrix(dpi / 72, dpi / 72)
        pages = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, 1):
                p = os.path.join(out, f'page_{i:04d}.png')
                page.get_pixmap(matrix=zoom).save(p)
                pages.append(p)
        return pages
    if path.lower().endswith(IMAGE_EXTS):
        return [path]
    sys.exit(f'unsupported file type: {path} (use a PDF, an image, or a folder of images)')


def image_part(path):
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{data}'}}


def run_ocr(server, pages, tiled=True, tokens_per_page=8000, show_progress=True):
    """Send all pages in one streaming request; return the raw model output."""
    content = [image_part(p) for p in pages]
    content.append({'type': 'text', 'text': '<PAGE><image>' * len(pages) + 'Multi page parsing.'})
    payload = {
        'model': 'Unlimited-OCR-2',
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': tokens_per_page * len(pages),
        'temperature': 0,
        'skip_special_tokens': False,
        'images_config': {'crop_mode': tiled},  # True = tiled ("gundam"), False = base
        'custom_logit_processor': DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
        'custom_params': {
            'ngram_size': 35,
            'window_size': 1024 if len(pages) > 1 else 128,
            'page_token_id': 128827,
            'eos_token_id': 1,
            'max_page_tokens': 20000,
        },
        'stream': True,
    }
    session = requests.Session()
    session.trust_env = False  # talk to the server directly, not through a system proxy
    resp = session.post(f"{server.rstrip('/')}/v1/chat/completions", json=payload,
                        timeout=6000, stream=True)
    if resp.status_code != 200:
        sys.exit(f'server error {resp.status_code}: {resp.text[:500]}')
    chunks, done_pages = [], 0
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        choices = json.loads(data).get('choices') or []
        delta = (choices[0].get('delta') or {}).get('content') if choices else None
        if delta:
            chunks.append(delta)
            done_pages += delta.count('<PAGE>')
            if show_progress:
                print(f'\r  reading page {min(done_pages, len(pages))}/{len(pages)}',
                      end='', file=sys.stderr, flush=True)
    if show_progress:
        print(file=sys.stderr)
    return ''.join(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='PDF, image, or folder of page images')
    ap.add_argument('-o', '--output', help='write Markdown here (default: print it)')
    ap.add_argument('--raw', help="also save the model's raw layout output here")
    ap.add_argument('--server', default='http://127.0.0.1:10000', help='SGLang server URL')
    ap.add_argument('--dpi', type=int, default=300, help='PDF render resolution (default 300)')
    ap.add_argument('--base', action='store_true',
                    help='one global view per page instead of tiles: lower memory, for very long documents')
    args = ap.parse_args()

    pages = collect_pages(args.input, args.dpi)
    print(f'[ocr] {len(pages)} page(s) -> {args.server}', file=sys.stderr)
    raw = run_ocr(args.server, pages, tiled=not args.base)
    if args.raw:
        with open(args.raw, 'w', encoding='utf-8') as f:
            f.write(raw)
    markdown = convert(raw)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(markdown + '\n')
        print(f'[ocr] saved {args.output}', file=sys.stderr)
    else:
        print(markdown)


if __name__ == '__main__':
    main()
