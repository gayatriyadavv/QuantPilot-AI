"""
QuantPilot AI — Sentiment Analysis Module

Provides financial news sentiment analysis using FinBERT and multi-source
news aggregation.

Components:
    - SentimentAnalyzer: FinBERT-based sentiment scoring with caching and
      GPU acceleration.
    - NewsFetcher: Async news collector from RSS feeds, Yahoo Finance, and
      keyword search.
"""

from backend.sentiment.analyzer import SentimentAnalyzer
from backend.sentiment.news_fetcher import NewsFetcher

__all__ = [
    "SentimentAnalyzer",
    "NewsFetcher",
]
