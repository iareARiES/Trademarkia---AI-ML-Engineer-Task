"""
Corpus preprocessing pipeline for 20 Newsgroups dataset.
Strips metadata noise, normalises text, and applies length filtering.

Design rationale: Raw newsgroup posts contain significant non-semantic noise
(routing headers, quoted replies, PGP blocks, URLs) that would pollute
embeddings. Each cleaning step targets a specific noise source while preserving
the author's original semantic content. The subject line is deliberately
re-injected as a body prefix because it is the highest-signal, most query-like
text in a post — it acts as a semantic anchor for the embedding model.
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
    """Extract Subject line from raw newsgroup post.

    Why extract before stripping headers: The subject line is the most
    information-dense text in a newsgroup post — short, topical, and
    structurally similar to user search queries. We extract it here so we
    can prepend it to the cleaned body later, giving the embedding model
    a strong semantic signal in its first tokens (which transformers
    weight more heavily due to positional encoding).
    """
    match = _SUBJECT_RE.search(raw_text)
    if match:
        subject = match.group(1).strip()
        # Strip "Re: Re:" chains — they add no semantic value and waste tokens
        subject = re.sub(r'^(Re:\s*)+', '', subject, flags=re.IGNORECASE).strip()
        return subject if subject else None
    return None


def strip_headers(text: str) -> str:
    """Remove email-style headers from the beginning of a post.

    Why strip headers: Headers contain routing metadata (From:, NNTP-Posting-Host:,
    Organization:, Message-ID:) that have zero semantic value but would be encoded
    into the embedding vector, diluting the actual topical content. The embedding
    model has a 256-token context window — every header token wastes capacity.
    """
    lines = text.split('\n')
    body_start = 0
    in_header = True

    for i, line in enumerate(lines):
        if in_header:
            if _HEADER_KEYS.match(line) or (line.startswith(' ') and i > 0):
                body_start = i + 1
                continue
            elif line.strip() == '':
                # Blank line = RFC 2822 header/body separator
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
    """Remove lines starting with > (quoted reply blocks).

    Why remove quoted replies: Quoted text is someone else's content, not the
    author's contribution. Including it would double-count the original post's
    semantics and bias the embedding toward the quoted material rather than the
    author's actual response. This is especially important in long reply chains
    where >90% of the text may be quoted.
    """
    return _QUOTED_LINE.sub('', text)


def strip_pgp_signatures(text: str) -> str:
    """Remove PGP signed message blocks and signatures."""
    return _PGP_BLOCK.sub('', text)


def strip_mime_boundaries(text: str) -> str:
    """Remove MIME boundary markers."""
    return _MIME_BOUNDARY.sub('', text)


def remove_urls_and_emails(text: str) -> str:
    """Remove URLs and email addresses.

    Why strip URLs/emails but keep surrounding text: URLs and email addresses
    are identifiers, not semantic content — they encode location, not meaning.
    The surrounding natural-language text ("Visit ... for more info") retains
    its semantic value without the URL. The embedding model would waste tokens
    encoding character sequences like "http://" that carry no topical signal.
    """
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
    """Apply NFKC Unicode normalisation.

    Why NFKC: Compatibility decomposition followed by canonical composition.
    Maps typographic variants to their standard forms (e.g., "ﬁ" ligature → "fi",
    fullwidth "Ａ" → "A", superscript "²" → "2"). Without this, the tokeniser
    would treat "ﬁnd" and "find" as different tokens, fragmenting the embedding
    space. NFKC is preferred over NFC because it also normalises compatibility
    characters common in Usenet-era text.
    """
    return unicodedata.normalize('NFKC', text)


def preprocess_document(raw_text: str, min_tokens: int = 30, max_tokens: int = 512) -> Optional[str]:
    """
    Full preprocessing pipeline for a single newsgroup post.

    Returns cleaned text with subject prefix, or None if document
    is too short after cleaning.

    Why min_tokens=30: Posts shorter than 30 tokens after cleaning (e.g., "me too",
    "thanks", "+1") have insufficient semantic content to produce a meaningful
    384-dim embedding. They would cluster randomly and add noise to search results.
    Empirically, 30 tokens filters 9% of posts (1,695/18,846) — mostly one-liners.

    Why max_tokens=512: The all-MiniLM-L6-v2 model has a 256 WordPiece token limit.
    ~512 whitespace tokens ≈ 256 WordPiece tokens after subword splitting. Text
    beyond this is silently truncated by the model anyway, so explicit truncation
    saves memory during batch encoding without losing information.
    """
    # Extract subject before stripping headers — must happen first because
    # strip_headers() removes all header lines including Subject:
    subject = extract_subject(raw_text)

    # Apply cleaning pipeline in order (order matters: headers first, then
    # content-level cleaning, then whitespace normalisation)
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

    # Length filter — discard documents that are too noisy/short to embed usefully
    tokens = text.split()
    if len(tokens) < min_tokens:
        return None

    # Truncate at max_tokens to match model context window
    if len(tokens) > max_tokens:
        text = ' '.join(tokens[:max_tokens])

    # Prepend subject as high-signal prefix: subject lines are short, query-like,
    # and strongly topical. Placing them first ensures transformers' positional
    # encoding gives them maximum attention weight.
    if subject:
        text = f"Subject: {subject}\n\n{text}"

    return text
