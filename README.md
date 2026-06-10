# DocLING extraction microservice

Layout-aware PDF text + table extraction for [PDFTrusted](https://pdftrusted.com) translate pipeline.

Repo: https://github.com/shashankpandey2405-cpu/docling-service

## Health

```json
{
  "ok": true,
  "docling_installed": true,
  "engine": "auto",
  "active_engine": "docling"
}
```

If `docling_installed: false` → only pdfplumber fallback (word-level, no tables).

## Railway

1. Memory **≥ 4 GB** (DocLING + PyTorch models).
2. Public URL → set on AI worker:
   ```
   DOCLING_URL=https://docling-service-production-8769.up.railway.app
   DOCLING_EXTRACT_ENABLED=true
   ```
3. Or private network (same project):
   ```
   DOCLING_URL=http://docling-service.railway.internal:8080
   ```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status + whether DocLING is installed |
| POST | `/extract?max_pages=10&page_offset=0` | Upload PDF → bbox blocks JSON |

## Engines

| Engine | Quality |
|--------|---------|
| **docling** | Tables, layout, reading order (best) |
| **pdfplumber** | Word-level bbox fallback |
| **auto** | DocLING if installed, else pdfplumber |

Force engine: `DOCLING_ENGINE=docling` or `pdfplumber`
