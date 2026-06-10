# DocLING extraction microservice

Layout-aware PDF text + table extraction for PDFTrusted translate pipeline.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status + engine |
| POST | `/extract` | Upload PDF → bbox blocks JSON |

## Run locally

```bash
cd repos/docling-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

## Railway / Docker

```bash
docker build -t pdftrusted-docling .
docker run -p 8080:8080 pdftrusted-docling
```

Set on AI worker:

```
DOCLING_URL=https://your-docling-service.railway.app
DOCLING_EXTRACT_ENABLED=true
```

## Response shape

Matches `TextBlock` in `server/translate/types.ts`:

```json
{
  "blocks": [
    {
      "id": "d0",
      "page_index": 0,
      "text": "Experience",
      "x": 72.0,
      "y": 650.0,
      "width": 120.0,
      "height": 14.0,
      "font_size": 12.0,
      "block_type": "text"
    }
  ],
  "page_count": 2,
  "engine": "docling",
  "tables_found": 1
}
```

## Engines

- **docling** — full layout + tables (install `docling` package)
- **pdfplumber** — fallback word-level bbox (always available)
- **auto** (default) — DocLING if installed, else pdfplumber
