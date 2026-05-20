"""
QuantPilot AI — Financial News Fetcher

Async news-collection layer that aggregates headlines from:

* **RSS feeds** — Reuters, MarketWatch, CNBC, Bloomberg, Seeking Alpha, etc.
* **Yahoo Finance** — ticker-specific news via ``yfinance``.
* **Keyword search** — filters aggregated headlines by symbol / keyword.

All HTTP requests use ``aiohttp`` for non-blocking I/O.

Usage::

    import asyncio
    from backend.sentiment.news_fetcher import NewsFetcher

    fetcher = NewsFetcher()

    # Fetch RSS headlines
    headlines = asyncio.run(fetcher.fetch_rss_feeds())

    # Ticker-specific news
    yahoo_news = asyncio.run(fetcher.fetch_yahoo_news("AAPL"))

    # Search across everything
    results = asyncio.run(fetcher.search_news("Apple"))
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Sequence

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from loguru import logger


# ── Default Financial RSS Feeds ──────────────────────────────────────────

DEFAULT_RSS_FEEDS: List[Dict[str, str]] = [
    {
        "name": "Reuters — Business",
        "url": "https://feeds.reuters.com/reuters/businessNews",
    },
    {
        "name": "Reuters — Markets",
        "url": "https://feeds.reuters.com/reuters/marketsNews",
    },
    {
        "name": "MarketWatch — Top Stories",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
    },
    {
        "name": "MarketWatch — Market Pulse",
        "url": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    },
    {
        "name": "CNBC — Top News",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    },
    {
        "name": "CNBC — Finance",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    },
    {
        "name": "Yahoo Finance — Top Stories",
        "url": "https://finance.yahoo.com/news/rssindex",
    },
    {
        "name": "Seeking Alpha — Market News",
        "url": "https://seekingalpha.com/market_currents.xml",
    },
    {
        "name": "Investing.com — News",
        "url": "https://www.investing.com/rss/news.rss",
    },
    {
        "name": "Bloomberg — Markets",
        "url": "https://feeds.bloomberg.com/markets/news.rss",
    },
]


# ── Data Containers ──────────────────────────────────────────────────────


@dataclass
class NewsArticle:
    """A single news article / headline.

    Attributes:
        title: Headline text.
        url: Link to the full article.
        source: Name of the feed or provider.
        published_at: Publication timestamp (UTC).
        summary: Short summary / description (may be empty).
        symbols: Ticker symbols mentioned in the article (auto-detected).
    """

    title: str
    url: str = ""
    source: str = ""
    published_at: Optional[datetime] = None
    summary: str = ""
    symbols: List[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        """SHA-256 hash of normalised title for deduplication."""
        normalised = self.title.strip().lower()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": (
                self.published_at.isoformat() if self.published_at else None
            ),
            "summary": self.summary,
            "symbols": self.symbols,
        }


# ── News Fetcher ─────────────────────────────────────────────────────────


class NewsFetcher:
    """Async financial news aggregator.

    Parameters:
        feeds: List of RSS feed definitions (dicts with ``name`` and ``url``
            keys). Defaults to :data:`DEFAULT_RSS_FEEDS`.
        request_timeout: Per-request timeout in seconds.
        max_concurrent: Maximum number of concurrent HTTP requests.
        user_agent: HTTP User-Agent header.
    """

    def __init__(
        self,
        feeds: Optional[List[Dict[str, str]]] = None,
        request_timeout: int = 15,
        max_concurrent: int = 10,
        user_agent: str = "QuantPilotAI/1.0 (+https://github.com/quantpilot)",
    ) -> None:
        self._feeds = feeds or DEFAULT_RSS_FEEDS
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._user_agent = user_agent

        # In-memory dedup set (fingerprints)
        self._seen: Set[str] = set()

        logger.info(
            "NewsFetcher initialised | feeds={n} | timeout={t}s | "
            "max_concurrent={mc}",
            n=len(self._feeds),
            t=request_timeout,
            mc=max_concurrent,
        )

    # ────────────────────────────────────────────────────────────
    # RSS Feeds
    # ────────────────────────────────────────────────────────────

    async def fetch_rss_feeds(
        self,
        feeds: Optional[List[Dict[str, str]]] = None,
    ) -> List[NewsArticle]:
        """Fetch and parse multiple RSS feeds concurrently.

        Args:
            feeds: Optional override list of feed definitions. Uses the
                instance default if not provided.

        Returns:
            Deduplicated list of ``NewsArticle`` objects, newest first.
        """
        target_feeds = feeds or self._feeds
        logger.info("Fetching {n} RSS feeds …", n=len(target_feeds))

        async with aiohttp.ClientSession(
            timeout=self._timeout,
            headers={"User-Agent": self._user_agent},
        ) as session:
            tasks = [
                self._fetch_single_feed(session, f["url"], f.get("name", f["url"]))
                for f in target_feeds
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: List[NewsArticle] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("RSS feed error: {e}", e=result)
                continue
            articles.extend(result)

        deduped = self.deduplicate(articles)
        # Sort by published time (newest first), unknowns at the end
        deduped.sort(
            key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        logger.info(
            "RSS fetch complete | raw={raw} | deduped={ded}",
            raw=len(articles),
            ded=len(deduped),
        )
        return deduped

    async def _fetch_single_feed(
        self,
        session: aiohttp.ClientSession,
        url: str,
        source_name: str,
    ) -> List[NewsArticle]:
        """Fetch and parse a single RSS feed.

        Args:
            session: Active aiohttp session.
            url: Feed URL.
            source_name: Human-readable source name.

        Returns:
            List of parsed ``NewsArticle`` objects.
        """
        async with self._semaphore:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(
                            "RSS {src} returned HTTP {s}",
                            src=source_name,
                            s=response.status,
                        )
                        return []
                    content = await response.text()
            except asyncio.TimeoutError:
                logger.warning("Timeout fetching RSS: {src}", src=source_name)
                return []
            except aiohttp.ClientError as exc:
                logger.warning(
                    "HTTP error fetching {src}: {e}", src=source_name, e=exc
                )
                return []

        feed = feedparser.parse(content)
        articles: List[NewsArticle] = []

        for entry in feed.entries:
            title = self._clean_html(entry.get("title", "")).strip()
            if not title:
                continue

            # Parse publication date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(
                        *entry.published_parsed[:6], tzinfo=timezone.utc
                    )
                except (ValueError, TypeError):
                    pass
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                try:
                    published = datetime(
                        *entry.updated_parsed[:6], tzinfo=timezone.utc
                    )
                except (ValueError, TypeError):
                    pass

            summary = self._clean_html(entry.get("summary", ""))[:500]

            articles.append(
                NewsArticle(
                    title=title,
                    url=entry.get("link", ""),
                    source=source_name,
                    published_at=published,
                    summary=summary,
                )
            )

        logger.debug(
            "Parsed {n} articles from {src}", n=len(articles), src=source_name
        )
        return articles

    # ────────────────────────────────────────────────────────────
    # Yahoo Finance News
    # ────────────────────────────────────────────────────────────

    async def fetch_yahoo_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch recent news for a specific ticker from Yahoo Finance.

        Uses the ``yfinance`` library (sync under the hood), wrapped in an
        executor so the calling event loop is not blocked.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).

        Returns:
            List of ``NewsArticle`` objects with the symbol pre-tagged.
        """
        logger.info("Fetching Yahoo Finance news for {sym}", sym=symbol)
        loop = asyncio.get_event_loop()

        try:
            articles = await loop.run_in_executor(
                None, self._sync_fetch_yahoo, symbol
            )
        except Exception as exc:
            logger.error(
                "Yahoo Finance fetch failed for {sym}: {e}", sym=symbol, e=exc
            )
            return []

        deduped = self.deduplicate(articles)
        logger.info(
            "Yahoo Finance | {sym} | articles={n}", sym=symbol, n=len(deduped)
        )
        return deduped

    @staticmethod
    def _sync_fetch_yahoo(symbol: str) -> List[NewsArticle]:
        """Synchronous helper to call yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning(
                "yfinance not installed — Yahoo news unavailable. "
                "Install with: pip install yfinance"
            )
            return []

        ticker = yf.Ticker(symbol)
        news = ticker.news or []

        articles: List[NewsArticle] = []
        for item in news:
            title = item.get("title", "").strip()
            if not title:
                continue

            published = None
            pub_epoch = item.get("providerPublishTime")
            if pub_epoch:
                try:
                    published = datetime.fromtimestamp(pub_epoch, tz=timezone.utc)
                except (ValueError, OSError):
                    pass

            articles.append(
                NewsArticle(
                    title=title,
                    url=item.get("link", ""),
                    source=item.get("publisher", "Yahoo Finance"),
                    published_at=published,
                    symbols=[symbol],
                )
            )
        return articles

    # ────────────────────────────────────────────────────────────
    # Keyword / Symbol Search
    # ────────────────────────────────────────────────────────────

    async def search_news(
        self,
        keyword: str,
        include_rss: bool = True,
        include_yahoo: bool = True,
    ) -> List[NewsArticle]:
        """Search for news matching a keyword or ticker symbol.

        Aggregates from RSS feeds and Yahoo Finance, then filters articles
        whose title or summary contains the keyword (case-insensitive).

        Args:
            keyword: Search term (e.g. ``"AAPL"`` or ``"inflation"``).
            include_rss: Whether to include RSS feed results.
            include_yahoo: Whether to query Yahoo Finance (treats *keyword*
                as a ticker symbol).

        Returns:
            Filtered, deduplicated list of articles.
        """
        logger.info("Searching news for keyword={kw!r}", kw=keyword)
        tasks: List[Any] = []

        if include_rss:
            tasks.append(self.fetch_rss_feeds())
        if include_yahoo:
            tasks.append(self.fetch_yahoo_news(keyword))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: List[NewsArticle] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Search sub-task error: {e}", e=result)
                continue
            all_articles.extend(result)

        # Filter by keyword
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        filtered = [
            a
            for a in all_articles
            if pattern.search(a.title) or pattern.search(a.summary)
        ]

        # Tag matching articles with the symbol
        upper_kw = keyword.upper()
        for article in filtered:
            if upper_kw not in article.symbols:
                article.symbols.append(upper_kw)

        deduped = self.deduplicate(filtered)
        logger.info(
            "Search results for {kw!r} | matched={n}",
            kw=keyword,
            n=len(deduped),
        )
        return deduped

    # ────────────────────────────────────────────────────────────
    # Deduplication
    # ────────────────────────────────────────────────────────────

    def deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on normalised title hashing.

        Args:
            articles: List of articles to deduplicate.

        Returns:
            New list with duplicates removed (preserves order).
        """
        unique: List[NewsArticle] = []
        seen_local: Set[str] = set()

        for article in articles:
            fp = article.fingerprint
            if fp not in self._seen and fp not in seen_local:
                seen_local.add(fp)
                unique.append(article)

        # Merge local set into persistent set
        self._seen.update(seen_local)
        return unique

    def clear_seen(self) -> None:
        """Reset the deduplication set."""
        self._seen.clear()
        logger.info("Deduplication set cleared")

    # ────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_html(text: str) -> str:
        """Strip HTML tags from text using BeautifulSoup.

        Args:
            text: Raw HTML or plain text.

        Returns:
            Cleaned plain text.
        """
        if not text:
            return ""
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        return text

    @property
    def feed_count(self) -> int:
        """Number of configured RSS feeds."""
        return len(self._feeds)

    @property
    def seen_count(self) -> int:
        """Number of article fingerprints in the dedup set."""
        return len(self._seen)

    def add_feed(self, name: str, url: str) -> None:
        """Add a new RSS feed to the fetcher at runtime.

        Args:
            name: Human-readable feed name.
            url: Feed URL.
        """
        self._feeds.append({"name": name, "url": url})
        logger.info("Added RSS feed: {name} ({url})", name=name, url=url)

    def remove_feed(self, name: str) -> bool:
        """Remove a feed by name.

        Args:
            name: Name of the feed to remove.

        Returns:
            ``True`` if a feed was removed, ``False`` if not found.
        """
        before = len(self._feeds)
        self._feeds = [f for f in self._feeds if f.get("name") != name]
        removed = len(self._feeds) < before
        if removed:
            logger.info("Removed RSS feed: {name}", name=name)
        return removed

    def list_feeds(self) -> List[Dict[str, str]]:
        """Return a list of all configured feeds."""
        return list(self._feeds)

    def __repr__(self) -> str:
        return (
            f"<NewsFetcher feeds={self.feed_count} seen={self.seen_count}>"
        )
