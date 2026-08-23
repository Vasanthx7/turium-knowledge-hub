"""Offline tests for HTML → readable-text extraction (no network)."""

from __future__ import annotations

from app.services.content_fetcher import HttpContentFetcher


def test_extract_pulls_title_and_strips_noise():
    html = """
    <html><head><title>  My Page  </title>
      <style>.x{color:red}</style></head>
      <body>
        <nav>home about contact</nav>
        <script>console.log('tracking')</script>
        <main><p>The core content lives here.</p>
              <p>Second paragraph of content.</p></main>
        <footer>copyright 2026</footer>
      </body></html>
    """
    title, text = HttpContentFetcher._extract(html)

    assert title == "My Page"
    assert "core content lives here" in text
    assert "Second paragraph of content" in text
    # Boilerplate tags are removed.
    assert "tracking" not in text
    assert "home about contact" not in text
    assert "copyright" not in text


def test_extract_prefers_main_over_body_chrome():
    html = """
    <html><head><title>Doc</title></head><body>
      <header>site header</header>
      <article>Only this article text should be kept.</article>
      <aside>related links</aside>
    </body></html>
    """
    _, text = HttpContentFetcher._extract(html)

    assert "Only this article text should be kept." in text
    assert "site header" not in text
    assert "related links" not in text


def test_extract_missing_title_returns_empty_title():
    _, text = HttpContentFetcher._extract("<html><body><p>Body only.</p></body></html>")
    title, _ = HttpContentFetcher._extract("<html><body><p>Body only.</p></body></html>")
    assert title == ""
    assert "Body only." in text
