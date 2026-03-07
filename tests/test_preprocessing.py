"""
Unit tests for the preprocessing pipeline.
"""

import pytest

from src.ingestion.preprocessing import (
    collapse_whitespace,
    extract_subject,
    filter_short_lines,
    normalize_unicode,
    preprocess_document,
    remove_quoted_replies,
    remove_urls_and_emails,
    strip_headers,
    strip_mime_boundaries,
    strip_pgp_signatures,
)


class TestExtractSubject:
    def test_basic_subject(self):
        text = "From: user@example.com\nSubject: Hello World\n\nBody text"
        assert extract_subject(text) == "Hello World"

    def test_re_prefix_removal(self):
        text = "Subject: Re: Re: Hello\n\nBody"
        assert extract_subject(text) == "Hello"

    def test_no_subject(self):
        text = "From: user@example.com\n\nBody text"
        assert extract_subject(text) is None

    def test_empty_subject(self):
        text = "Subject: \n\nBody"
        assert extract_subject(text) is None


class TestStripHeaders:
    def test_removes_standard_headers(self):
        text = "From: user@test.com\nSubject: Test\nOrganization: ACME\n\nBody text here"
        result = strip_headers(text)
        assert "Body text here" in result
        assert "From:" not in result
        assert "Organization:" not in result

    def test_preserves_body(self):
        text = "From: user@test.com\n\nThis is the body.\nMore body text."
        result = strip_headers(text)
        assert "This is the body." in result
        assert "More body text." in result


class TestRemoveQuotedReplies:
    def test_removes_single_quoted(self):
        text = "My reply\n> Quoted text\n> More quoted\nBack to normal"
        result = remove_quoted_replies(text)
        assert "> Quoted text" not in result
        assert "My reply" in result
        assert "Back to normal" in result

    def test_removes_nested_quotes(self):
        text = "Reply\n>> Double quoted\n>>> Triple quoted\nNormal"
        result = remove_quoted_replies(text)
        assert ">>" not in result
        assert "Reply" in result


class TestRemoveUrlsAndEmails:
    def test_removes_urls(self):
        text = "Visit http://example.com for more info"
        result = remove_urls_and_emails(text)
        assert "http://example.com" not in result
        assert "Visit" in result

    def test_removes_emails(self):
        text = "Contact user@example.com for help"
        result = remove_urls_and_emails(text)
        assert "user@example.com" not in result
        assert "Contact" in result


class TestCollapseWhitespace:
    def test_collapses_spaces(self):
        text = "Hello    World"
        assert collapse_whitespace(text) == "Hello World"

    def test_limits_newlines(self):
        text = "Para 1\n\n\n\n\nPara 2"
        result = collapse_whitespace(text)
        assert "\n\n\n" not in result


class TestFilterShortLines:
    def test_removes_short_lines(self):
        text = "This is a good line\n--\nAnother good line here"
        result = filter_short_lines(text)
        assert "--" not in result
        assert "This is a good line" in result


class TestNormalizeUnicode:
    def test_nfkc_normalization(self):
        # ﬁ (fi ligature) → fi
        text = "ﬁnd"
        result = normalize_unicode(text)
        assert result == "find"


class TestPreprocessDocument:
    def test_full_pipeline_returns_content(self):
        raw = (
            "From: user@test.com\n"
            "Subject: Test Post About Science\n"
            "Organization: MIT\n"
            "\n"
            "This is a test document about science and technology. "
            "It contains enough words to pass the minimum token filter. "
            "We need at least thirty tokens in this document body "
            "for it to survive the preprocessing pipeline cleanup. "
            "Adding more words here to make sure we have enough content."
        )
        result = preprocess_document(raw)
        assert result is not None
        assert "Subject: Test Post About Science" in result

    def test_too_short_returns_none(self):
        raw = "From: user@test.com\nSubject: Hi\n\nHello"
        result = preprocess_document(raw)
        assert result is None

    def test_strips_quotes(self):
        raw = (
            "From: user@test.com\n"
            "Subject: Reply to Discussion\n"
            "\n"
            "> This is quoted material from the original post\n"
            "> More quoted material here\n"
            "This is my actual reply to the discussion thread with sufficient words. "
            "I am adding more content to make sure this passes the token filter. "
            "The preprocessing should strip the quoted lines above."
        )
        result = preprocess_document(raw)
        if result:
            assert "> This is quoted" not in result
