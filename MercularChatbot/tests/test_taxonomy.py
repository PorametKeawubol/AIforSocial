from taxonomy import category_source_from_url, leaf_categories, parse_category_sitemap


def test_category_sitemap_retains_nested_public_categories_and_drops_placeholders():
    xml = """<?xml version=\"1.0\"?>
    <urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">
      <url><loc>https://www.mercular.com/audio/</loc></url>
      <url><loc>https://www.mercular.com/audio/dap-dac-amp/</loc></url>
      <url><loc>https://www.mercular.com/audio/dap-dac-amp/dac-amplifiers/</loc></url>
      <url><loc>https://www.mercular.com/audio/dap-dac-amp/turntable-cartridge/</loc></url>
      <url><loc>https://www.mercular.com/audio/-/</loc></url>
    </urlset>"""

    sources = parse_category_sitemap(xml)

    assert [source.path for source in sources] == [
        ("audio",),
        ("audio", "dap-dac-amp"),
        ("audio", "dap-dac-amp", "dac-amplifiers"),
        ("audio", "dap-dac-amp", "turntable-cartridge"),
    ]
    assert [source.path for source in leaf_categories(sources)] == [
        ("audio", "dap-dac-amp", "dac-amplifiers"),
        ("audio", "dap-dac-amp", "turntable-cartridge"),
    ]


def test_category_source_rejects_non_mercular_urls():
    assert category_source_from_url("https://example.com/audio") is None
    assert category_source_from_url("https://www.mercular.com/browse/all") is None
