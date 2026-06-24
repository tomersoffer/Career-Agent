# -*- coding: utf-8 -*-
"""
cv/extract.py
Extract plain text from an uploaded PDF or DOCX file.

Lazy-imports pypdf and docx so the module can be imported even if a dep
is temporarily missing (the import error will surface only at call time).
"""

import io
import os
import re


class ExtractError(Exception):
    """Raised when text extraction fails or produces an empty result."""
    pass


def _normalize(text: str) -> str:
    """Collapse whitespace runs and strip leading/trailing blanks."""
    text = re.sub(r"\r\n|\r", "\n", text)       # unify line endings
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal ws
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse blank lines
    return text.strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from a PDF or DOCX byte payload.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content (as received from an HTTP upload, for example).
    filename : str
        Original filename; used only to determine the format by extension.

    Returns
    -------
    str
        Non-empty, whitespace-normalised plain text.

    Raises
    ------
    ExtractError
        If the extension is unsupported, the file cannot be parsed, or the
        extracted text is empty/whitespace (e.g. a scanned image-only PDF).
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # lazy import
        except ImportError as exc:
            raise ExtractError("pypdf is not installed: " + str(exc)) from exc
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ExtractError("Could not open PDF: " + str(exc)) from exc

        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        text = "\n".join(parts)

    elif ext == ".docx":
        try:
            import docx  # python-docx exposes the 'docx' namespace
        except ImportError as exc:
            raise ExtractError("python-docx is not installed: " + str(exc)) from exc
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ExtractError("Could not open DOCX: " + str(exc)) from exc

        text = "\n".join(para.text for para in doc.paragraphs)

    else:
        raise ExtractError(
            f"Unsupported file type '{ext}'. Only .pdf and .docx are accepted."
        )

    text = _normalize(text)
    if not text:
        raise ExtractError(
            "could not extract text from the file "
            "(it may be an image-only / scanned document). "
            "Please paste your CV text manually."
        )
    return text
