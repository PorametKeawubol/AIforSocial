from advice_scraper import (
    AdviceBranchSearcher,
    Branch,
    extract_branches_from_html,
)


def test_extracts_current_branch_selector_and_deduplicates():
    html = """
    <div class="cu-accordion-item branch-detail-card">
      <a href="/branch/hatyai"><span class="t-branch-name">Advice หาดใหญ่</span></a>
    </div>
    <div class="cu-accordion-item branch-detail-card">
      <a href="/branch/hatyai"><span class="t-branch-name">Advice หาดใหญ่</span></a>
    </div>
    """

    assert extract_branches_from_html(html, "https://www.advice.co.th/wheretobuy") == [
        Branch("Advice หาดใหญ่", "https://www.advice.co.th/branch/hatyai")
    ]


def test_extracts_selector_from_exercise_handout():
    html = '<div class="list-items-branch"><h3><a href="/branch/a">Advice A</a></h3></div>'

    result = extract_branches_from_html(html, "https://www.advice.co.th/wheretobuy")

    assert result == [Branch("Advice A", "https://www.advice.co.th/branch/a")]


def test_extracts_only_the_exact_requested_province_group():
    html = """
    <button class="cu-btn-accordion" data-bs-target="#box-94">ปัตตานี : (2)</button>
    <div id="box-94">
      <span class="t-branch-name">Advice Pattani</span>
      <span class="t-branch-name">Advice PSU Pattani</span>
    </div>
    <button class="cu-btn-accordion" data-bs-target="#box-05">สงขลา : (2)</button>
    <div id="box-05">
      <span class="t-branch-name">Advice Ranot</span>
      <span class="t-branch-name">Advice Hatyai</span>
    </div>
    """

    result = extract_branches_from_html(
        html,
        "https://www.advice.co.th/wheretobuy",
        province="จังหวัดสงขลา",
    )

    assert result == [
        Branch("Advice Ranot", ""),
        Branch("Advice Hatyai", ""),
    ]


def test_search_always_delegates_to_selenium(monkeypatch):
    searcher = AdviceBranchSearcher()
    expected = [Branch("Advice หาดใหญ่", "https://example.test/branch")]
    received_keywords = []

    def fake_selenium_search(keyword):
        received_keywords.append(keyword)
        return expected

    monkeypatch.setattr(searcher, "_search_with_selenium", fake_selenium_search)

    assert searcher.search("  หาดใหญ่  ") == expected
    assert received_keywords == ["หาดใหญ่"]
