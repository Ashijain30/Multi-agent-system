import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv

from models import SourceItem

load_dotenv()

logger = logging.getLogger(__name__)

# Request timeout settings: (connect timeout, read timeout) in seconds
DEFAULT_TIMEOUT = (5, 15)
DEFAULT_MAX_RESULTS = 6
DEFAULT_CONTENT_LIMIT = 3000


def get_credential(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve credential from environment variables or Streamlit secrets.
    Compatible with local .env development and Streamlit Community Cloud deployments.
    """
    # 1. Environment variable (from .env or OS)
    val = os.getenv(key)
    if val and val.strip():
        return val.strip()

    # 2. Streamlit secrets (Streamlit Community Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            secret_val = st.secrets[key]
            if secret_val:
                return str(secret_val).strip()
    except Exception:
        pass

    return default


def sanitize_error_message(message: str) -> str:
    """Sanitize error messages to prevent exposing API keys or sensitive tokens."""
    if not isinstance(message, str):
        return ""
    # Redact common key formats
    sanitized = re.sub(r"(gsk_[a-zA-Z0-9_-]{10,}|tvly-[a-zA-Z0-9_-]{10,})", "[REDACTED_API_KEY]", message)
    # Redact active credential values if present
    for env_key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
        val = get_credential(env_key)
        if val and len(val) > 4:
            sanitized = sanitized.replace(val, "[REDACTED_API_KEY]")
    return sanitized


def normalize_url(url: str) -> str:
    """
    Normalize a URL for reliable deduplication.
    Lowercases scheme and domain, strips fragments and tracking query params,
    and removes trailing slashes.
    """
    if not url or not isinstance(url, str):
        return ""
    clean = url.strip().strip("'\"<>")
    try:
        parsed = urlparse(clean)
        if not parsed.scheme or not parsed.netloc:
            return clean
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        # Filter out common analytics and tracking parameters
        query_pairs = [
            (k, v) for k, v in parse_qsl(parsed.query)
            if not k.lower().startswith(("utm_", "ref", "fbclid", "gclid"))
        ]
        new_query = urlencode(query_pairs)
        return urlunparse((scheme, netloc, path, "", new_query, ""))
    except Exception:
        return clean


def get_tavily_client() -> TavilyClient:
    """Retrieve an initialized TavilyClient or raise ValueError if key is missing."""
    api_key = get_credential("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not configured. Please set TAVILY_API_KEY in your .env file or Streamlit secrets.")
    return TavilyClient(api_key=api_key)


def search_sources(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, str]]:
    """
    Query Tavily and return deduplicated raw search items.
    Each item contains 'title', 'url', and 'snippet'.
    """
    if not query or not query.strip():
        return []

    client = get_tavily_client()
    clean_query = query.strip()
    logger.info("Executing Tavily search for: '%s' (max_results=%d)", clean_query, max_results)

    raw_results = client.search(query=clean_query, max_results=max_results, timeout=15)
    items = raw_results.get("results", []) if isinstance(raw_results, dict) else []

    deduped = []
    seen_urls = set()

    for item in items:
        url = item.get("url", "").strip()
        norm = normalize_url(url)
        if not norm or norm in seen_urls:
            continue
        seen_urls.add(norm)
        deduped.append({
            "title": item.get("title", "Untitled").strip(),
            "url": url,
            "snippet": item.get("content", "").strip(),
        })

    logger.info("Retrieved %d unique search sources for '%s'", len(deduped), clean_query)
    return deduped


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    if not query or not query.strip():
        return "Search query cannot be empty."

    try:
        items = search_sources(query.strip(), max_results=DEFAULT_MAX_RESULTS)
        if not items:
            return f"No search results found for query: '{query.strip()}'"

        out = []
        for r in items:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("snippet", "")[:300].strip() or "No snippet available."
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")

        return "\n----\n".join(out)
    except ValueError as ve:
        logger.error("Configuration error in web_search: %s", ve)
        return sanitize_error_message(str(ve))
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return sanitize_error_message(f"Web search failed ({type(e).__name__}): {str(e)}")


def fetch_url_content(clean_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch and clean text content from a validated URL.
    Returns (cleaned_text, error_message). Exactly one will be non-None.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        logger.info("Scraping URL: %s", clean_url)
        resp = requests.get(clean_url, timeout=DEFAULT_TIMEOUT, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if not text:
            return None, "No readable text content found on webpage."

        return text[:DEFAULT_CONTENT_LIMIT], None

    except requests.exceptions.Timeout:
        logger.warning("Scraping timed out for URL: %s", clean_url)
        return None, "Request timed out."
    except requests.exceptions.HTTPError as he:
        status = resp.status_code if "resp" in locals() else "unknown"
        logger.warning("HTTP %s error scraping %s", status, clean_url)
        return None, f"HTTP {status} error."
    except requests.exceptions.RequestException as re_err:
        logger.warning("Network connection error scraping %s: %s", clean_url, re_err)
        return None, f"Network error ({type(re_err).__name__})."
    except Exception as e:
        logger.exception("Unexpected error scraping %s: %s", clean_url, e)
        return None, f"Unexpected error: {type(e).__name__}."


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    if not url or not isinstance(url, str) or not url.strip():
        return "Could not scrape URL: URL must be a non-empty string."

    clean_url = url.strip().strip("'\"<>")
    parsed = urlparse(clean_url)
    if not parsed.scheme or parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return f"Could not scrape URL: Invalid URL format '{clean_url}'. URL must start with http:// or https://"

    content, err = fetch_url_content(clean_url)
    if err:
        return sanitize_error_message(f"Could not scrape URL: {err} when accessing '{clean_url}'.")
    return content or ""


def scrape_multiple_sources(sources: List[Dict[str, str]], max_urls: int = 5) -> List[SourceItem]:
    """
    Extract content from approximately 3-5 unique sources.
    Deduplicates URLs, handles inaccessible websites gracefully without failing the pipeline,
    and falls back to search snippets for any unreachable URLs.
    """
    selected_items: List[Dict[str, str]] = []
    seen_urls = set()

    for item in sources:
        url = item.get("url", "").strip()
        norm = normalize_url(url)
        if not norm or norm in seen_urls:
            continue
        seen_urls.add(norm)
        selected_items.append(item)
        if len(selected_items) >= max_urls:
            break

    logger.info("Beginning multi-source extraction for %d URLs (max_urls=%d)", len(selected_items), max_urls)
    results: List[SourceItem] = []

    for item in selected_items:
        url = item.get("url", "").strip()
        title = item.get("title", "Untitled").strip()
        snippet = item.get("snippet", "").strip()

        # Validate URL format before attempting scrape
        parsed = urlparse(url)
        if not parsed.scheme or parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            logger.warning("Skipping invalid URL: %s", url)
            results.append(SourceItem(
                title=title,
                url=url,
                snippet=snippet,
                content=snippet,
                is_scraped=False,
                error="Invalid URL format"
            ))
            continue

        content, err = fetch_url_content(url)
        if err or not content:
            logger.warning("Inaccessible website %s (%s). Falling back to search snippet.", url, err)
            results.append(SourceItem(
                title=title,
                url=url,
                snippet=snippet,
                content=snippet,
                is_scraped=False,
                error=err or "No content found"
            ))
        else:
            logger.info("Successfully scraped %s (%d chars)", url, len(content))
            results.append(SourceItem(
                title=title,
                url=url,
                snippet=snippet,
                content=content,
                is_scraped=True,
                error=None
            ))

    return results
