"""
Corpus preprocessing pipeline for 20 Newsgroups dataset.
Strips metadata noise, normalises text, and applies length filtering.
"""

import re
import unicodedata
from typing import Optional


# --- Header patterns ---
_HEADER_KEYS = re.compile(
    r'^(From|Subject|Organization|Lines|NNTP-Posting-Host|'
    r'Reply-To|Distribution|Keywords|Summary|Expires|'
    r'References|Sender|Message-ID|Date|In-Reply-To|'
    r'Strstrng|Xref|X-\S+|Approved|Archive-name|'
    r'Last-modified|Version):\s*',
    re.IGNORECASE
)

_SUBJECT_RE = re.compile(r'^Subject:[ \t]*([^\n]+)', re.IGNORECASE | re.MULTILINE)

# --- Noise patterns ---
_QUOTED_LINE = re.compile(r'^\s*>+.*$', re.MULTILINE)
_PGP_BLOCK = re.compile(
    r'-----BEGIN PGP (?:SIGNED )?MESSAGE-----.*?'
    r'-----END PGP (?:SIGNATURE|MESSAGE)-----',
    re.DOTALL
)
_MIME_BOUNDARY = re.compile(r'^--[A-Za-z0-9_\-]+\s*$', re.MULTILINE)
_URL_RE = re.compile(r'https?://\S+|ftp://\S+|www\.\S+', re.IGNORECASE)
_EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
_WHITESPACE_COLLAPSE = re.compile(r'[ \t]+')
_MULTI_NEWLINE = re.compile(r'\n{3,}')


def extract_subject(raw_text: str) -> Optional[str]:
    """Extract Subject line from raw newsgroup post."""
    match = _SUBJECT_RE.search(raw_text)
    if match:
        subject = match.group(1).strip()
        # Remove Re: prefixes
        subject = re.sub(r'^(Re:\s*)+', '', subject, flags=re.IGNORECASE).strip()
        return subject if subject else None
    return None


def strip_headers(text: str) -> str:
    """Remove email-style headers from the beginning of a post."""
    lines = text.split('\n')
    body_start = 0
    in_header = True

    for i, line in enumerate(lines):
        if in_header:
            if _HEADER_KEYS.match(line) or (line.startswith(' ') and i > 0):
                body_start = i + 1
                continue
            elif line.strip() == '':
                body_start = i + 1
                in_header = False
                break
            else:
                # Non-header line found, body starts here
                body_start = i
                in_header = False
                break

    return '\n'.join(lines[body_start:])


def remove_quoted_replies(text: str) -> str:
    """Remove lines starting with > (quoted reply blocks)."""
    return _QUOTED_LINE.sub('', text)


def strip_pgp_signatures(text: str) -> str:
    """Remove PGP signed message blocks and signatures."""
    return _PGP_BLOCK.sub('', text)


def strip_mime_boundaries(text: str) -> str:
    """Remove MIME boundary markers."""
    return _MIME_BOUNDARY.sub('', text)


def remove_urls_and_emails(text: str) -> str:
    """Remove URLs and email addresses."""
    text = _URL_RE.sub('', text)
    text = _EMAIL_RE.sub('', text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to single space, limit newlines."""
    text = _WHITESPACE_COLLAPSE.sub(' ', text)
    text = _MULTI_NEWLINE.sub('\n\n', text)
    return text.strip()


def filter_short_lines(text: str, min_tokens: int = 3) -> str:
    """Remove lines with fewer than min_tokens tokens."""
    lines = text.split('\n')
    filtered = [
        line for line in lines
        if len(line.split()) >= min_tokens or line.strip() == ''
    ]
    return '\n'.join(filtered)


def normalize_unicode(text: str) -> str:
    """Apply NFKC Unicode normalisation."""
    return unicodedata.normalize('NFKC', text)


def preprocess_document(raw_text: str, min_tokens: int = 30, max_tokens: int = 512) -> Optional[str]:
    """
    Full preprocessing pipeline for a single newsgroup post.

    Returns cleaned text with subject prefix, or None if document
    is too short after cleaning.
    """
    # Extract subject before stripping headers
    subject = extract_subject(raw_text)

    # Apply cleaning pipeline in order
    text = strip_headers(raw_text)
    text = remove_quoted_replies(text)
    text = strip_pgp_signatures(text)
    text = strip_mime_boundaries(text)
    text = remove_urls_and_emails(text)
    text = collapse_whitespace(text)
    text = filter_short_lines(text)
    text = normalize_unicode(text)

    # Final cleanup
    text = text.strip()

    # Length filter
    tokens = text.split()
    if len(tokens) < min_tokens:
        return None

    # Truncate at max_tokens
    if len(tokens) > max_tokens:
        text = ' '.join(tokens[:max_tokens])

    # Prepend subject as high-signal prefix
    if subject:
        text = f"Subject: {subject}\n\n{text}"

    return text
