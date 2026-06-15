"""enotropos — Language detection module.

Detects the language of extracted wine specification texts
using fasttext-langdetect (ftlangdetect).
"""
import json
import logging
from pathlib import Path
from typing import Any

from winegpt.config import EXTRACTED_DIR, LANG_DETECT_CHAR_LIMIT

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect language of a text snippet. Returns ISO 639-1 code."""
    snippet = text[:LANG_DETECT_CHAR_LIMIT]
    try:
        from ftlangdetect import detect

        result = detect(snippet)
        return result.get("lang", "unknown") if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.debug("Language detection failed: %s", e)
        return "unknown"


def enrich_metadata(json_path: Path) -> str | None:
    """Read an extraction JSON file, detect its language, and update the file.

    Returns the detected language code, or None on error.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Cannot read %s: %s", json_path, e)
        return None

    # Read text from companion .md file if available
    md_path = json_path.with_suffix(".md")
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
    else:
        text = ""

    if not text:
        return None

    lang = detect_language(text)
    data["language"] = lang
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return lang


def process_country(country: str) -> dict[str, int]:
    """Detect language for all extracted JSON files of a country.

    Returns stats dict.
    """
    country_extracted = EXTRACTED_DIR / country
    if not country_extracted.exists():
        logger.error("No extracted data for %s", country)
        return {}

    stats: dict[str, int] = {}

    for json_file in sorted(country_extracted.rglob("*.json")):
        lang = enrich_metadata(json_file)
        if lang:
            stats[lang] = stats.get(lang, 0) + 1

    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=== enotropos — Language Detection ===")

    stats = process_country("Espanya")
    logger.info("")
    logger.info("Language distribution for Espanya:")
    for lang, count in sorted(stats.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d PDFs", lang, count)


if __name__ == "__main__":
    main()
