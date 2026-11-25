# Unlimited-OCR 2 kit

Scripts for installing and using [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (Unlimited-OCR 2). The model reads PDFs and scanned pages and turns them into Markdown. It handles headings, tables, images, and text that continues from one page to the next. It can read documents with hundreds of pages in a single pass.

## What you need

- A Linux machine with an **NVIDIA GPU** and CUDA drivers. Baidu tested on an 80 GB A800. Smaller cards may work for short documents; lower `--mem-fraction-static` and the page windows in `start_server.sh` if you run out of memory.
- `git`, `git-lfs`, and [`uv`](https://docs.astral.sh/uv/).
- Internet access to GitHub and Hugging Face. The first server start downloads the model.

No GPU? Use Baidu's hosted demo at <https://huggingface.co/spaces/baidu/Unlimited-OCR-2>, or the [TextMind web app](https://console.bce.baidu.com/textmind/application/textParser).

## 1. Install (once)

```bash
cd unlimited-ocr
bash install.sh
```

This clones Baidu's repo into `Unlimited-OCR/`, creates a Python 3.10 environment in `.venv/`, and installs Baidu's customized SGLang server.

## 2. Start the server (terminal 1)

```bash
bash start_server.sh
```

Wait for `The server is fired up and ready to roll!`. Leave it running. To use a port other than 10000, run `bash start_server.sh 10001`.

## 3. Read documents (terminal 2)

```bash
.venv/bin/python ocr.py report.pdf -o report.md     # PDF -> Markdown file
.venv/bin/python ocr.py report.pdf                  # print Markdown to the screen
.venv/bin/python ocr.py scan.jpg -o scan.md         # one image
.venv/bin/python ocr.py pages/ -o book.md           # folder of page images (page_1, page_2, ... page_10 in order)
```

Options:

| Option | What it does |
|---|---|
| `-o FILE` | Write the Markdown to a file. |
| `--raw FILE` | Also save the model's raw layout output (blocks with coordinates). |
| `--server URL` | Use a server on another machine, e.g. `http://gpu-box:10000`. |
| `--dpi N` | Resolution for rendering PDF pages (default 300). |
| `--base` | One view per page instead of tiles. Uses less memory; good for very long documents. |

The server can run on a GPU machine while `ocr.py` runs on your laptop. Install the client there with `uv pip install <the sglang wheel> --no-deps`, plus `requests` and `pymupdf`.

### From Python

```python
import ocr  # this folder's ocr.py

pages = ocr.collect_pages("report.pdf", dpi=300)
raw = ocr.run_ocr("http://127.0.0.1:10000", pages)
markdown = ocr.convert(raw)
```

## What the output looks like

The model emits layout blocks such as:

```
<PAGE><BLOCK>title<LEVEL1>[50,40,950,90]<FORK>Annual Report
<BLOCK>cross_table[...]<FORK><table>...</table><PAGE><BLOCK>append_table[...]<FORK><table>...</table>
```

`ocr.py` converts that with Baidu's `process_unlimitedOCR2.convert`:

- `title<LEVELn>` becomes an `#`×n heading.
- Images and charts become `![](pageN_x1_y1_x2_y2.jpg)`. The coordinates are on a 0–1000 scale of the page.
- Tables stay HTML. A table split across pages is merged and its repeated header row is dropped.
- Paragraphs split across pages are joined.

## Troubleshooting

- **`install.sh` says the wheel is an LFS pointer.** Run `git lfs install`, then `git lfs pull` inside `Unlimited-OCR/`.
- **The server runs out of GPU memory.** Lower `--mem-fraction-static` or `--max-prefill-tokens` (and the last `--cuda-graph-bs-prefill` value to match), or lower the `*_PAGE_WINDOW` values.
- **`server error 400 ... maximum context length`.** The document is too long for one window. Use `--base`, or split the PDF.
