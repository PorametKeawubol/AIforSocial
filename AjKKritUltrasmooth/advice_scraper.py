"""Selenium-based Advice.co.th branch search.

The browser opens Advice's branch-search page and reads its rendered results;
this module doesn't call the site's internal API directly.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

try:
    import chromedriver_autoinstaller
except ImportError:  # pragma: no cover - useful when Selenium Manager is used
    chromedriver_autoinstaller = None


LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class Branch:
    """A branch result returned by the Advice website."""

    name: str
    link: str = ""


class AdviceScraperError(RuntimeError):
    """Raised when the Advice page cannot be opened or scraped."""


SEARCH_INPUT_LOCATORS: tuple[tuple[str, str], ...] = (
    # Selector from the exercise handout.
    (By.ID, "shop_find"),
    # Selector used by the current Advice website.
    (By.CSS_SELECTOR, "input.form-control-adv"),
    (By.XPATH, "//input[contains(@placeholder, 'ค้นหาชื่อจังหวัด') or contains(@placeholder, 'ค้นหาสาขา') ]"),
)

SEARCH_BUTTON_LOCATORS: tuple[tuple[str, str], ...] = (
    (By.CSS_SELECTOR, ".search-box button.btn-blue"),
    (By.CSS_SELECTOR, ".search-box + button.btn-blue"),
    (By.XPATH, "//input[@id='shop_find']/following::button[1]"),
    (
        By.XPATH,
        "//input[contains(@placeholder, 'ค้นหาชื่อจังหวัด') or contains(@placeholder, 'ค้นหาสาขา')]/"
        "ancestor::div[contains(@class, 'search-box')][1]/following-sibling::button[1]",
    ),
)

BRANCH_SELECTORS: tuple[str, ...] = (
    # Selector from the exercise handout.
    ".list-items-branch h3 > a",
    # Selector used by the current Advice website.
    ".t-branch-name",
)

PROVINCE_GROUP_SELECTOR = ".cu-btn-accordion"


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _element_text(element) -> str:
    """Read visible text and fall back to textContent for collapsed cards."""

    try:
        text = _clean_text(element.text or element.get_attribute("textContent"))
    except StaleElementReferenceException:
        return ""
    return re.sub(r"(?<=\w)\(", " (", text)


def _normalise_link(href: str | None, base_url: str) -> str:
    href = (href or "").strip()
    if not href or href.startswith("javascript:") or href == "#":
        return ""
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("www."):
        href = f"https://{href}"
    return urljoin(base_url, href)


def _normalise_province_name(value: str | None) -> str:
    """Normalize a province label from Advice's accordion or a user query."""

    value = _clean_text(value)
    value = re.sub(r"\s*:\s*\(\s*\d+\s*\)\s*$", "", value)
    value = re.sub(r"^(?:จังหวัด|จ\.)\s*", "", value)
    return value.casefold()


def _province_scope_from_soup(soup: BeautifulSoup, province: str):
    """Return the accordion panel for an exact province label, if present."""

    wanted = _normalise_province_name(province)
    if not wanted:
        return None

    for heading in soup.select(PROVINCE_GROUP_SELECTOR):
        if _normalise_province_name(heading.get_text(" ", strip=True)) != wanted:
            continue
        target = (heading.get("data-bs-target") or "").strip()
        if target.startswith("#"):
            return soup.find(id=target[1:])
    return None


def _branch_search_link(name: str, base_url: str) -> str:
    """Create a useful link when the current site renders a click-only card."""

    return f"{urljoin(base_url, '/wheretobuy/search')}?keyword={quote(name, safe='')}"


def _unique_branches(branches: Iterable[Branch]) -> list[Branch]:
    result: list[Branch] = []
    seen: dict[str, int] = {}
    for branch in branches:
        name = _clean_text(branch.name)
        link = branch.link.strip()
        if not name:
            continue
        # Selenium textContent can omit the whitespace around a <br>, while
        # BeautifulSoup preserves it.  Deduplicate by normalized name and
        # prefer a result that has a link.
        key = re.sub(r"\s+\(", "(", name).casefold()
        if key in seen:
            existing_index = seen[key]
            if not result[existing_index].link and link:
                result[existing_index] = Branch(name=name, link=link)
            continue
        seen[key] = len(result)
        result.append(Branch(name=name, link=link))
    return result


def extract_branches_from_html(
    html: str, base_url: str, province: str | None = None
) -> list[Branch]:
    """Extract branch names and links from a rendered Advice HTML page.

    This pure helper is also used as a last-resort parser when a browser
    element does not expose a link in the expected way.
    """

    soup = BeautifulSoup(html, "html.parser")
    scope = soup
    if province:
        scope = _province_scope_from_soup(soup, province)
        if scope is None:
            return []
    branches: list[Branch] = []

    for selector in BRANCH_SELECTORS:
        for element in scope.select(selector):
            name = _clean_text(element.get_text(" ", strip=True))
            href = element.get("href")
            if not href:
                parent_link = element.find_parent("a", href=True)
                href = parent_link.get("href") if parent_link else ""
            if not href:
                card = element.find_parent(
                    class_=lambda classes: classes
                    and any(
                        token in " ".join(classes) if isinstance(classes, list) else str(classes)
                        for token in ("branch-detail-card", "cu-accordion-item")
                    )
                )
                if card:
                    card_link = card.find("a", href=True)
                    href = card_link.get("href") if card_link else ""
            branches.append(Branch(name=name, link=_normalise_link(href, base_url)))

        if branches:
            # Avoid parsing the same visual result twice when both old and
            # new selectors happen to be present in a future page version.
            break

    return _unique_branches(branches)


class AdviceBranchSearcher:
    """Open Advice.co.th with Selenium and search for a branch."""

    def __init__(
        self,
        url: str = "https://www.advice.co.th/wheretobuy",
        timeout_seconds: int = 25,
        headless: bool = True,
        max_results: int = 10,
        page_settle_seconds: float = 5,
        save_html: bool = False,
        html_output_dir: str | Path = "artifacts",
        driver_factory: Callable[[webdriver.ChromeOptions], webdriver.Chrome] | None = None,
    ) -> None:
        self.url = url
        self.timeout_seconds = max(5, int(timeout_seconds))
        self.headless = headless
        self.max_results = max(1, int(max_results))
        self.page_settle_seconds = max(0, float(page_settle_seconds))
        self.save_html = save_html
        self.html_output_dir = Path(html_output_dir)
        self.driver_factory = driver_factory

    def _create_driver(self) -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-setuid-sandbox")
        options.add_argument("--no-zygote")
        options.add_argument("--remote-debugging-pipe")
        options.add_argument("--window-size=1440,1200")
        options.add_argument("--lang=th-TH")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
        chrome_binary = os.getenv("CHROME_BINARY", "").strip()
        if chrome_binary:
            options.binary_location = chrome_binary

        if self.driver_factory is not None:
            return self.driver_factory(options)

        # The handout uses chromedriver-autoinstaller.  Selenium Manager is
        # still the primary mechanism for current Selenium versions, while
        # this fallback keeps the handout's setup usable on older machines.
        auto_install = os.getenv("CHROMEDRIVER_AUTO_INSTALL", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if auto_install and chromedriver_autoinstaller is not None:
            try:
                chromedriver_autoinstaller.install()
            except Exception as exc:  # Selenium Manager may still succeed.
                LOGGER.warning("Could not auto-install ChromeDriver: %s", exc)

        try:
            return webdriver.Chrome(options=options)
        except WebDriverException as exc:
            raise AdviceScraperError(
                "เปิด Chrome/ChromeDriver ไม่สำเร็จ ตรวจสอบว่าติดตั้ง Chromium หรือ Chrome แล้ว"
            ) from exc

    @staticmethod
    def _first_visible(driver: webdriver.Chrome, locators: Iterable[tuple[str, str]]):
        for locator in locators:
            try:
                element = driver.find_element(*locator)
            except NoSuchElementException:
                continue
            if element.is_displayed():
                return element
        return False

    def _find_search_input(self, driver: webdriver.Chrome):
        try:
            return WebDriverWait(driver, self.timeout_seconds).until(
                lambda current: self._first_visible(current, SEARCH_INPUT_LOCATORS)
            )
        except TimeoutException as exc:
            raise AdviceScraperError("ไม่พบช่องค้นหาสาขาบนหน้า Advice") from exc

    @staticmethod
    def _click_search(driver: webdriver.Chrome) -> bool:
        for locator in SEARCH_BUTTON_LOCATORS:
            try:
                button = driver.find_element(*locator)
                if not button.is_displayed():
                    continue
                try:
                    button.click()
                except ElementClickInterceptedException:
                    driver.execute_script("arguments[0].click();", button)
                return True
            except NoSuchElementException:
                continue
        return False

    @staticmethod
    def _has_result_or_empty_state(driver: webdriver.Chrome) -> bool:
        for selector in BRANCH_SELECTORS:
            # Some current results are rendered inside a collapsed accordion;
            # presence plus text is more reliable than is_displayed().
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements and any(_element_text(element) for element in elements):
                return True
            if "/wheretobuy/search" in driver.current_url and elements:
                return True

        page_text = _clean_text(driver.find_element(By.TAG_NAME, "body").text)
        no_result_markers = (
            "ไม่พบสาขา",
            "ไม่พบสาขา Advice",
            "ไม่พบข้อมูล",
            "ขออภัย ไม่พบ",
        )
        return any(marker in page_text for marker in no_result_markers)

    def _wait_for_results(self, driver: webdriver.Chrome) -> None:
        try:
            WebDriverWait(driver, self.timeout_seconds).until(self._has_result_or_empty_state)
        except TimeoutException:
            # A route change can finish just after the page has rendered.  We
            # still parse the final source below and return an empty result if
            # the site genuinely has no matching branch.
            LOGGER.warning("Timed out waiting for Advice search results")

    @staticmethod
    def _province_container_for_keyword(driver: webdriver.Chrome, keyword: str):
        """Find the rendered accordion panel when ``keyword`` is a province."""

        wanted = _normalise_province_name(keyword)
        if not wanted:
            return None

        for heading in driver.find_elements(By.CSS_SELECTOR, PROVINCE_GROUP_SELECTOR):
            label = heading.get_attribute("textContent")
            if _normalise_province_name(label) != wanted:
                continue
            target = (heading.get_attribute("data-bs-target") or "").strip()
            if not target.startswith("#"):
                return None
            try:
                return driver.find_element(By.ID, target[1:])
            except NoSuchElementException:
                LOGGER.warning("Could not find Advice province panel %s", target)
                return None
        return None

    def _extract_from_driver(
        self, driver: webdriver.Chrome, province_keyword: str | None = None
    ) -> list[Branch]:
        province_container = (
            self._province_container_for_keyword(driver, province_keyword)
            if province_keyword
            else None
        )
        search_root = province_container or driver
        branches: list[Branch] = []

        for selector in BRANCH_SELECTORS:
            elements = search_root.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                name = _element_text(element)
                if not name:
                    continue

                href = element.get_attribute("href")
                if not href:
                    ancestor_links = element.find_elements(By.XPATH, "ancestor::a[@href][1]")
                    if ancestor_links:
                        href = ancestor_links[0].get_attribute("href")

                if not href:
                    for ancestor_locator in (
                        "ancestor::*[contains(@class, 'branch-detail-card')][1]",
                        "ancestor::*[contains(@class, 'cu-accordion-item')][1]",
                    ):
                        try:
                            card = element.find_element(By.XPATH, ancestor_locator)
                        except NoSuchElementException:
                            continue
                        links = card.find_elements(By.CSS_SELECTOR, "a[href]")
                        if links:
                            href = links[0].get_attribute("href")
                            break

                link = _normalise_link(href, self.url)
                if not link:
                    # The current Advice UI uses a click handler instead of
                    # an anchor.  A keyword URL still opens the matching
                    # branch result and is useful to a LINE user.
                    link = _branch_search_link(name, self.url)
                branches.append(Branch(name=name, link=link))

            if branches:
                break

        # The DOM may render the name separately from the anchor.  Parsing the
        # scoped panel fills that gap and ensures an exact province query does
        # not leak cards from another province with a matching branch name.
        scope_html = (
            province_container.get_attribute("outerHTML")
            if province_container is not None
            else driver.page_source
        )
        branches.extend(extract_branches_from_html(scope_html, self.url))
        return _unique_branches(branches)[: self.max_results]

    def _save_html(self, driver: webdriver.Chrome, suffix: str) -> None:
        if not self.save_html:
            return
        self.html_output_dir.mkdir(parents=True, exist_ok=True)
        path = self.html_output_dir / f"advice_{suffix}.html"
        path.write_text(driver.page_source, encoding="utf-8")
        LOGGER.info("Saved Advice page HTML to %s", path)

    def _search_with_selenium(self, keyword: str) -> list[Branch]:
        """Search the rendered Advice page with Selenium."""

        driver = self._create_driver()
        try:
            driver.get(self.url)
            WebDriverWait(driver, self.timeout_seconds).until(
                lambda current: current.execute_script("return document.readyState") == "complete"
            )
            # Advice renders the Vue input in SSR HTML before its event
            # handlers are hydrated.  Waiting here prevents send_keys from
            # being overwritten by the first client-side render.
            if self.page_settle_seconds:
                time.sleep(self.page_settle_seconds)
            search_input = self._find_search_input(driver)
            search_input.clear()
            search_input.send_keys(keyword)
            self._save_html(driver, "page01")

            if not self._click_search(driver):
                # The original handout submits the input when no dedicated
                # search button is available.
                try:
                    search_input.send_keys(Keys.ENTER)
                except WebDriverException:
                    search_input.submit()

            self._wait_for_results(driver)
            self._save_html(driver, "page02")
            return self._extract_from_driver(driver, province_keyword=keyword)
        except AdviceScraperError:
            raise
        except (NoSuchElementException, TimeoutException, WebDriverException) as exc:
            raise AdviceScraperError("ค้นหาสาขา Advice ไม่สำเร็จ กรุณาลองใหม่อีกครั้ง") from exc
        finally:
            try:
                driver.quit()
            except WebDriverException as exc:
                # A browser process can disappear before quit() during a
                # network/renderer failure; do not hide the useful result or
                # original scraper error with a cleanup exception.
                LOGGER.warning("Could not close browser cleanly: %s", exc)

    def search(self, keyword: str) -> list[Branch]:
        """Search Advice for ``keyword`` and return at most ``max_results``."""

        keyword = _clean_text(keyword)
        if not keyword:
            return []
        if len(keyword) > 100:
            raise ValueError("คำค้นหายาวเกินไป (สูงสุด 100 ตัวอักษร)")

        return self._search_with_selenium(keyword)


def build_searcher_from_env() -> AdviceBranchSearcher:
    """Build a scraper using the environment variables used by the app."""

    def as_bool(value: str, default: bool) -> bool:
        if value == "":
            return default
        return value.lower() in {"1", "true", "yes", "on"}

    return AdviceBranchSearcher(
        url=os.getenv("ADVICE_URL", "https://www.advice.co.th/wheretobuy"),
        timeout_seconds=int(os.getenv("SCRAPER_TIMEOUT_SECONDS", "25")),
        headless=as_bool(os.getenv("HEADLESS", "true"), True),
        max_results=int(os.getenv("MAX_RESULTS", "10")),
        page_settle_seconds=float(os.getenv("PAGE_SETTLE_SECONDS", "5")),
        save_html=as_bool(os.getenv("SAVE_HTML", "false"), False),
        html_output_dir=os.getenv("HTML_OUTPUT_DIR", "artifacts"),
    )
