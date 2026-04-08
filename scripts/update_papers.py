import argparse
import csv
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "AwesomeGeoFMPapers/0.1 (https://github.com/your-name/awesome-geofm-papers)"
ROOT = Path(__file__).resolve().parents[1]
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class Paper:
    id: str
    source: str
    title: str
    authors: List[str]
    published: str
    updated: str
    venue: str
    url: str
    pdf_url: str
    doi: str
    abstract: str
    topics: List[str]
    query: str

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "updated": self.updated,
            "venue": self.venue,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "abstract": self.abstract,
            "topics": self.topics,
            "query": self.query,
        }


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_key(value: str) -> str:
    value = normalize_space(value).lower()
    value = value.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return re.sub(r"[^a-z0-9]+", "", value)


def short_authors(authors: List[str], limit: int = 4) -> str:
    if not authors:
        return ""
    if len(authors) <= limit:
        return ", ".join(authors)
    return ", ".join(authors[:limit]) + " et al."


def get_text(url: str, params: Optional[Dict] = None, pause: float = 0.5) -> str:
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while fetching {url}: {exc}") from exc
    time.sleep(pause)
    return payload


def get_json(url: str, params: Optional[Dict] = None, pause: float = 0.5) -> Dict:
    return json.loads(get_text(url, params=params, pause=pause))


def parse_date(value: str) -> str:
    if not value:
        return ""
    return value[:10]


def is_recent(published: str, days_back: int, today: date) -> bool:
    if not published:
        return True
    try:
        parsed = date.fromisoformat(published[:10])
    except ValueError:
        return True
    return parsed >= today - timedelta(days=days_back)


def arxiv_search_query(query: str) -> str:
    cleaned = query.replace('"', "")
    terms = [part for part in re.split(r"\s+", cleaned) if part]
    return " AND ".join(f'all:"{term}"' for term in terms)


def fetch_arxiv(config: Dict, today: date) -> List[Paper]:
    papers: List[Paper] = []
    max_results = int(config["search"]["max_results_per_query"])
    days_back = int(config["search"]["days_back"])
    for query in config["queries"]:
        payload = get_text(
            "https://export.arxiv.org/api/query",
            {
                "search_query": arxiv_search_query(query),
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        root = ET.fromstring(payload)
        for entry in root.findall("atom:entry", ARXIV_NS):
            published = parse_date(entry.findtext("atom:published", default="", namespaces=ARXIV_NS))
            if not is_recent(published, days_back, today):
                continue
            raw_id = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
            pdf_url = ""
            for link in entry.findall("atom:link", ARXIV_NS):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            papers.append(
                Paper(
                    id=raw_id.rsplit("/", 1)[-1],
                    source="arXiv",
                    title=normalize_space(entry.findtext("atom:title", default="", namespaces=ARXIV_NS)),
                    authors=[
                        normalize_space(author.findtext("atom:name", default="", namespaces=ARXIV_NS))
                        for author in entry.findall("atom:author", ARXIV_NS)
                    ],
                    published=published,
                    updated=parse_date(entry.findtext("atom:updated", default="", namespaces=ARXIV_NS)),
                    venue="arXiv",
                    url=raw_id,
                    pdf_url=pdf_url,
                    doi=normalize_space(entry.findtext("arxiv:doi", default="", namespaces=ARXIV_NS)),
                    abstract=normalize_space(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS)),
                    topics=[],
                    query=query,
                )
            )
    return papers


def fetch_openalex(config: Dict, today: date) -> List[Paper]:
    papers: List[Paper] = []
    max_results = int(config["search"]["max_results_per_query"])
    since = today - timedelta(days=int(config["search"]["days_back"]))
    for query in config["queries"]:
        payload = get_json(
            "https://api.openalex.org/works",
            {
                "search": query.replace('"', ""),
                "filter": f"from_publication_date:{since.isoformat()},to_publication_date:{today.isoformat()}",
                "sort": "publication_date:desc",
                "per-page": max_results,
            },
        )
        for item in payload.get("results", []):
            doi = normalize_space(item.get("doi") or "").replace("https://doi.org/", "")
            authors = [
                auth.get("author", {}).get("display_name", "")
                for auth in item.get("authorships", [])
                if auth.get("author", {}).get("display_name")
            ]
            source = item.get("primary_location", {}).get("source") or {}
            papers.append(
                Paper(
                    id=item.get("id", ""),
                    source="OpenAlex",
                    title=normalize_space(item.get("display_name", "")),
                    authors=authors,
                    published=parse_date(item.get("publication_date", "")),
                    updated=parse_date(item.get("updated_date", "")),
                    venue=normalize_space(source.get("display_name", "")),
                    url=item.get("doi") or item.get("id", ""),
                    pdf_url=((item.get("primary_location") or {}).get("pdf_url") or ""),
                    doi=doi,
                    abstract=rebuild_openalex_abstract(item.get("abstract_inverted_index") or {}),
                    topics=[],
                    query=query,
                )
            )
    return papers


def rebuild_openalex_abstract(inverted: Dict[str, List[int]]) -> str:
    tokens = []
    for token, positions in inverted.items():
        for pos in positions:
            tokens.append((pos, token))
    tokens.sort(key=lambda item: item[0])
    return normalize_space(" ".join(token for _, token in tokens))


def matches_scope(paper: Paper, config: Dict) -> bool:
    if config["search"].get("include_without_abstract_match"):
        return True
    haystack = f"{paper.title} {paper.abstract}".lower()
    has_model_signal = any(term.lower() in haystack for term in config.get("must_have_terms", []))
    has_geo_context = any(term.lower() in haystack for term in config.get("context_terms", []))
    return has_model_signal and has_geo_context


def assign_topics(paper: Paper, topic_rules: Dict[str, List[str]]) -> List[str]:
    haystack = f"{paper.title} {paper.abstract}".lower()
    topics = [topic for topic, terms in topic_rules.items() if any(term.lower() in haystack for term in terms)]
    return topics or ["Uncategorized"]


def paper_keys(paper: Paper) -> List[str]:
    keys = []
    for value in [paper.doi, paper.id, paper.url, paper.title]:
        key = normalize_key(value)
        if key:
            keys.append(key)
    arxiv_match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", " ".join([paper.id, paper.url, paper.doi]))
    if arxiv_match:
        keys.append(normalize_key(arxiv_match.group(1)))
    return list(dict.fromkeys(keys))


def dedupe(papers: Iterable[Paper]) -> List[Paper]:
    best: Dict[str, Paper] = {}
    aliases: Dict[str, str] = {}
    for paper in papers:
        keys = paper_keys(paper)
        if not keys:
            continue
        canonical = next((aliases[key] for key in keys if key in aliases), keys[0])
        old = best.get(canonical)
        if old is None or completeness_score(paper) > completeness_score(old):
            best[canonical] = paper
        for key in keys:
            aliases[key] = canonical
    return list(best.values())


def completeness_score(paper: Paper) -> int:
    fields = [paper.title, paper.authors, paper.published, paper.venue, paper.url, paper.pdf_url, paper.doi, paper.abstract]
    return sum(1 for field in fields if field)


def prune_old(papers: Iterable[Paper], keep_recent_days: int, today: date) -> List[Paper]:
    cutoff = today - timedelta(days=keep_recent_days)
    kept = []
    for paper in papers:
        if not paper.published:
            kept.append(paper)
            continue
        try:
            if date.fromisoformat(paper.published[:10]) >= cutoff:
                kept.append(paper)
        except ValueError:
            kept.append(paper)
    return kept


def load_existing(path: Path) -> List[Paper]:
    rows = load_json(path, [])
    return [
        Paper(
            id=row.get("id", ""),
            source=row.get("source", ""),
            title=row.get("title", ""),
            authors=row.get("authors", []),
            published=row.get("published", ""),
            updated=row.get("updated", ""),
            venue=row.get("venue", ""),
            url=row.get("url", ""),
            pdf_url=row.get("pdf_url", ""),
            doi=row.get("doi", ""),
            abstract=row.get("abstract", ""),
            topics=row.get("topics", []),
            query=row.get("query", ""),
        )
        for row in rows
    ]


def render_readme(config: Dict, papers: List[Paper], generated_at: datetime) -> str:
    title = config["project"]["name"]
    description = config["project"]["description"]
    lines = [
        f"# {title}",
        "",
        description,
        "",
        f"Last updated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "This repository is updated automatically by GitHub Actions. It tracks GeoFM-related papers from public scholarly APIs, currently arXiv and OpenAlex.",
        "",
        "## Recent Papers",
        "",
    ]
    if not papers:
        lines.extend(["No papers found yet.", ""])
        return "\n".join(lines)

    sorted_papers = sorted(papers, key=lambda item: (item.published or "", item.updated or "", item.title), reverse=True)
    by_topic: Dict[str, List[Paper]] = {}
    for paper in sorted_papers:
        primary = paper.topics[0] if paper.topics else "Uncategorized"
        by_topic.setdefault(primary, []).append(paper)

    for topic in sorted(by_topic):
        lines.extend([f"### {topic}", "", "| Date | Paper | Authors | Source | Links |", "| --- | --- | --- | --- | --- |"])
        for paper in by_topic[topic]:
            links = []
            if paper.url:
                links.append(f"[Page]({paper.url})")
            if paper.pdf_url:
                links.append(f"[PDF]({paper.pdf_url})")
            if paper.doi:
                links.append(f"[DOI](https://doi.org/{paper.doi})")
            title_cell = html.escape(paper.title)
            venue = html.escape(paper.venue or paper.source)
            lines.append(
                f"| {paper.published or ''} | {title_cell} | {html.escape(short_authors(paper.authors))} | {venue} | {' / '.join(links)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Search Scope",
            "",
            "Keywords live in `config.json`. Edit the query list and topic rules to make the tracker broader or stricter.",
            "",
            "## Local Update",
            "",
            "```bash",
            "python scripts/update_papers.py",
            "```",
            "",
            "Generated files:",
            "",
            "- `README.md`",
            "- `data/papers.json`",
            "- `data/papers.csv`",
            "- `data/run_summary.json`",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(papers: List[Paper], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["published", "title", "authors", "source", "venue", "url", "pdf_url", "doi", "topics", "query", "abstract"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            writer.writerow(
                {
                    "published": paper.published,
                    "title": paper.title,
                    "authors": "; ".join(paper.authors),
                    "source": paper.source,
                    "venue": paper.venue,
                    "url": paper.url,
                    "pdf_url": paper.pdf_url,
                    "doi": paper.doi,
                    "topics": "; ".join(paper.topics),
                    "query": paper.query,
                    "abstract": paper.abstract,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update GeoFM paper tracker.")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--data", default=str(ROOT / "data" / "papers.json"))
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    parser.add_argument("--csv", default=str(ROOT / "data" / "papers.csv"))
    parser.add_argument("--summary", default=str(ROOT / "data" / "run_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(Path(args.config), {})
    today = datetime.now(timezone.utc).date()
    fetched: List[Paper] = []

    if config["sources"].get("arxiv", True):
        fetched.extend(fetch_arxiv(config, today))
    if config["sources"].get("openalex", True):
        fetched.extend(fetch_openalex(config, today))

    fetched = [paper for paper in fetched if matches_scope(paper, config)]
    existing = load_existing(Path(args.data))
    candidates = [paper for paper in [*existing, *fetched] if matches_scope(paper, config)]
    for paper in candidates:
        paper.topics = assign_topics(paper, config.get("topic_rules", {}))

    merged = dedupe(candidates)
    merged = prune_old(merged, int(config["search"]["keep_recent_days"]), today)
    merged = sorted(merged, key=lambda item: (item.published or "", item.updated or "", item.title), reverse=True)

    write_json(Path(args.data), [paper.to_dict() for paper in merged])
    write_csv(merged, Path(args.csv))
    generated_at = datetime.now(timezone.utc)
    Path(args.readme).write_text(render_readme(config, merged, generated_at), encoding="utf-8")
    write_json(
        Path(args.summary),
        {
            "generated_at": generated_at.isoformat(timespec="seconds"),
            "fetched_this_run": len(fetched),
            "records_total": len(merged),
            "sources": [name for name, enabled in config["sources"].items() if enabled],
        },
    )
    print(json.dumps(load_json(Path(args.summary), {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




