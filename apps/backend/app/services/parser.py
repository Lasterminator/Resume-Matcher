"""Document parsing service using markitdown and LLM."""

import tempfile
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from app.llm import complete_json
from app.prompts import PARSE_RESUME_PROMPT
from app.prompts.templates import RESUME_SCHEMA_EXAMPLE
from app.schemas import ResumeData, normalize_resume_data


async def parse_document(content: bytes, filename: str) -> str:
    """Convert PDF/DOCX to Markdown using markitdown.

    Args:
        content: Raw file bytes
        filename: Original filename for extension detection

    Returns:
        Markdown text content
    """
    suffix = Path(filename).suffix.lower()

    # Write to temp file for markitdown
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        md = MarkItDown()
        result = md.convert(str(tmp_path))
        return result.text_content
    finally:
        tmp_path.unlink(missing_ok=True)


async def parse_resume_to_json(markdown_text: str) -> dict[str, Any]:
    """Parse resume markdown to structured JSON using LLM.

    Args:
        markdown_text: Resume content in markdown format

    Returns:
        Structured resume data matching ResumeData schema
    """
    prompt = PARSE_RESUME_PROMPT.format(
        schema=RESUME_SCHEMA_EXAMPLE,
        resume_text=markdown_text,
    )

    # Use more tokens and retries for local/small models (e.g. Ollama gemma3:4b)
    # so the full JSON is not truncated and we have retries for flaky output.
    result = await complete_json(
        prompt=prompt,
        system_prompt=(
            "You are a JSON extraction engine. Output ONLY a single valid JSON object. "
            "No markdown, no code fences, no explanation. Start with { and end with }."
        ),
        max_tokens=8192,
        retries=3,
    )

    # Normalize so missing sectionMeta/customSections get defaults (helps Ollama/small models)
    result = normalize_resume_data(result)

    # Validate against schema
    validated = ResumeData.model_validate(result)
    return validated.model_dump()
