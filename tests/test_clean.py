import pytest

from hst.clean import canonicalize_url, clean_title, normalize_title


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  World   leaders meet  ", "World leaders meet"),
        ("A &amp; B", "A & B"),
        ("cafe\u0301", "café"),
        ("LIVE: Debate begins", "Debate begins"),
        ("watch: Debate begins", "Debate begins"),
        ("UPDATE: Talks resume", "Talks resume"),
        ("Headline | BBC News", "Headline"),
        ("Headline - Reuters", "Headline"),
        ("Headline — CNN", "Headline"),
        ("Headline – Sky News", "Headline"),
        ("  Multiple&nbsp; spaces ", "Multiple spaces"),
        ("President, 5 ministers meet", "President, 5 ministers meet"),
        ("Weather:  HOT  and sunny", "Weather: HOT and sunny"),
        ("One\nTwo", "One Two"),
        ("A  /  B", "A / B"),
        ("The　city", "The city"),
        ("Interview: \tMinister", "Interview: Minister"),
        ("Breaking News! | Outlet", "Breaking News!"),
        ("WATCH: A live update", "A live update"),
        ("Live : A live update", "A live update"),
        ("UPDATE - Talks resume", "UPDATE - Talks resume"),
        ("Market | 2026", "Market"),
        ("  ", ""),
        ("<b>Bold</b>", "<b>Bold</b>"),
        ("Fish &amp;amp; Chips", "Fish &amp; Chips"),
        ("U.S. leaders meet", "U.S. leaders meet"),
        ("  LIVE:  City opens  ", "City opens"),
        ("New A–B accord", "New A–B accord"),
        ("Quote: “Hello”", "Quote: “Hello”"),
        ("Résumé", "Résumé"),
        ("Headline - The Daily Example", "Headline"),
    ],
)
def test_clean_title_cases(raw, clean):
    assert clean_title(raw) == clean


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("HTTPS://Example.COM/a?utm_source=x&b=2#top", "https://example.com/a?b=2"),
        ("https://EXAMPLE.com/a?fbclid=1", "https://example.com/a"),
        ("https://example.com/a#g", "https://example.com/a"),
        ("https://example.com", "https://example.com/"),
        ("http://Example.COM/a?gclid=x&x=1", "http://example.com/a?x=1"),
        ("https://example.com/a?utm_campaign=x&x=1", "https://example.com/a?x=1"),
        ("https://example.com/a?ref=123", "https://example.com/a"),
    ],
)
def test_canonicalize_url(raw, canonical):
    assert canonicalize_url(raw) == canonical


def test_normalize_title():
    assert normalize_title("  CITY, Opens!  ") == "city opens"
