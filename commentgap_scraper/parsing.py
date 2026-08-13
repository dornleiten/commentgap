from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse


STORY_RE = re.compile(r"^/story/(?P<story_id>\d{7,13})(?:/|$)")


@dataclass(frozen=True, slots=True)
class DiscoveredStory:
    story_id: str
    year: int
    month: int
    url: str
    sitemap_lastmod: str | None


def parse_sitemap(xml_text: str, year: int, month: int) -> list[DiscoveredStory]:
    root = ET.fromstring(xml_text)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    stories: dict[str, DiscoveredStory] = {}
    for entry in root.findall("sm:url", namespace):
        location = entry.findtext("sm:loc", default="", namespaces=namespace).strip()
        match = STORY_RE.match(urlparse(location).path)
        if not match:
            continue
        story_id = match.group("story_id")
        lastmod = entry.findtext("sm:lastmod", default="", namespaces=namespace).strip() or None
        stories[story_id] = DiscoveredStory(story_id, year, month, location, lastmod)
    return list(stories.values())


def extract_page_config(html: str) -> dict[str, Any]:
    marker = "window.DERSTANDARD.pageConfig.init("
    start = html.find(marker)
    if start < 0:
        return {}
    start += len(marker)
    try:
        value, _ = json.JSONDecoder().raw_decode(html[start:].lstrip())
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


class _ArticleParser(HTMLParser):
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str]]] = []
        self.capture: list[tuple[str, int, list[str]]] = []
        self.title = ""
        self.subtitle = ""
        self.body_paragraphs: list[str] = []
        self.breadcrumbs: list[str] = []
        self.times: list[str] = []
        self.taxonomy: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag in self.VOID_TAGS:
            if tag == "meta" and (values.get("name") or "").lower() == "cxenseparse:taxonomy":
                self.taxonomy = (values.get("content") or "").strip() or None
            return
        self.stack.append((tag, classes))
        depth = len(self.stack)
        if tag == "h1" and "article-title" in classes:
            self.capture.append(("title", depth, []))
        elif tag == "p" and "article-subtitle" in classes:
            self.capture.append(("subtitle", depth, []))
        elif tag == "p" and any("article-body" in parent_classes for _, parent_classes in self.stack[:-1]):
            self.capture.append(("body", depth, []))
        elif tag == "a" and any("breadcrumb" in cls for _, cs in self.stack[:-1] for cls in cs):
            if values.get("href"):
                self.breadcrumbs.append(values["href"].strip())
            else:
                self.capture.append(("breadcrumb", depth, []))
        elif tag == "time" and values.get("datetime"):
            self.times.append(values["datetime"].strip())

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "time":
            values = dict(attrs)
            if values.get("datetime"):
                self.times.append(values["datetime"].strip())

    def handle_data(self, data: str) -> None:
        for _, _, chunks in self.capture:
            chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        matching_index = next(
            (index for index in range(len(self.stack) - 1, -1, -1) if self.stack[index][0] == tag),
            None,
        )
        if matching_index is None:
            return
        depth = matching_index + 1
        finished = [item for item in self.capture if item[1] == depth]
        self.capture = [item for item in self.capture if item[1] != depth]
        for kind, _, chunks in finished:
            text = " ".join("".join(chunks).split())
            if not text:
                continue
            if kind == "title":
                self.title = text
            elif kind == "subtitle":
                self.subtitle = text
            elif kind == "body":
                self.body_paragraphs.append(text)
            elif kind == "breadcrumb":
                self.breadcrumbs.append(text)
        del self.stack[matching_index:]


def parse_article(
    html: str,
    requested_url: str,
    story_id: str,
    sitemap_lastmod: str | None = None,
) -> dict[str, Any]:
    parser = _ArticleParser()
    parser.feed(html)
    config = extract_page_config(html)
    canonical = config.get("canonicalUrl") or requested_url
    page_publishing_date = config.get("contentPublishingDate") or (parser.times[0] if parser.times else None)
    publishing_date = page_publishing_date or sitemap_lastmod
    modified_date = (
        config.get("contentModifiedOn")
        or config.get("contentModificationDate")
        or sitemap_lastmod
    )
    sections = [value for value in parser.breadcrumbs if value]
    if not sections and parser.taxonomy:
        parts = [part for part in parser.taxonomy.strip("/").split("/") if part]
        sections = ["/" + "/".join(parts[: index + 1]) for index in range(len(parts))]
    return {
        "story_id": story_id,
        "canonical_url": canonical,
        "published_at": publishing_date,
        "published_at_source": "page" if page_publishing_date else "sitemap_lastmod_fallback",
        "modified_at": modified_date,
        "sitemap_lastmod": sitemap_lastmod,
        "title": config.get("contentTitle") or parser.title or None,
        "subtitle": config.get("contentSummary") or parser.subtitle or None,
        "body": "\n\n".join(parser.body_paragraphs) or None,
        "section_1": sections[0] if len(sections) > 0 else None,
        "section_2": sections[1] if len(sections) > 1 else None,
        "section_3": sections[2] if len(sections) > 2 else None,
    }
