"""
QuantPilot AI — Sentiment Analysis API Routes
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from backend.api.schemas import SentimentAnalyzeRequest, SentimentResponse, SentimentResult

router = APIRouter()

# Lazy-loaded analyzer
_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from backend.sentiment.analyzer import SentimentAnalyzer
        _analyzer = SentimentAnalyzer()
    return _analyzer


@router.get("/{symbol}", response_model=SentimentResponse)
async def get_symbol_sentiment(
    symbol: str,
    limit: int = Query(10, description="Max headlines to analyze"),
):
    """Get sentiment analysis for a symbol's latest news."""
    try:
        from backend.sentiment.news_fetcher import NewsFetcher

        fetcher = NewsFetcher()
        headlines = await fetcher.fetch_yahoo_news(symbol, limit=limit)

        if not headlines:
            return SentimentResponse(
                symbol=symbol,
                overall_sentiment="neutral",
                sentiment_score=0.0,
                results=[],
                analyzed_at=datetime.now().isoformat(),
            )

        analyzer = get_analyzer()
        texts = [h["title"] for h in headlines]
        results = analyzer.analyze_batch(texts)

        sentiment_results = []
        for headline, result in zip(headlines, results):
            sentiment_results.append(SentimentResult(
                headline=headline["title"],
                label=result["label"],
                positive_score=round(result["positive"], 4),
                negative_score=round(result["negative"], 4),
                neutral_score=round(result["neutral"], 4),
                confidence=round(result["confidence"], 4),
                source=headline.get("source"),
                published_at=headline.get("published_at"),
            ))

        # Calculate overall sentiment
        avg_positive = sum(r.positive_score for r in sentiment_results) / len(sentiment_results)
        avg_negative = sum(r.negative_score for r in sentiment_results) / len(sentiment_results)
        score = avg_positive - avg_negative

        overall = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"

        return SentimentResponse(
            symbol=symbol,
            overall_sentiment=overall,
            sentiment_score=round(score, 4),
            results=sentiment_results,
            analyzed_at=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Sentiment error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=SentimentResult)
async def analyze_text(request: SentimentAnalyzeRequest):
    """Analyze custom text for financial sentiment."""
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_text(request.text)

        return SentimentResult(
            headline=request.text,
            label=result["label"],
            positive_score=round(result["positive"], 4),
            negative_score=round(result["negative"], 4),
            neutral_score=round(result["neutral"], 4),
            confidence=round(result["confidence"], 4),
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_sentiment(limit: int = 20):
    """Get latest sentiment scores across all tracked assets."""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    results = {}

    for symbol in symbols:
        try:
            from backend.sentiment.news_fetcher import NewsFetcher
            fetcher = NewsFetcher()
            headlines = await fetcher.fetch_yahoo_news(symbol, limit=3)

            if headlines:
                analyzer = get_analyzer()
                texts = [h["title"] for h in headlines]
                batch_results = analyzer.analyze_batch(texts)

                avg_score = sum(
                    r["positive"] - r["negative"] for r in batch_results
                ) / len(batch_results)

                results[symbol] = {
                    "score": round(avg_score, 4),
                    "label": "bullish" if avg_score > 0.1 else "bearish" if avg_score < -0.1 else "neutral",
                    "headlines_analyzed": len(batch_results),
                }
        except Exception as e:
            results[symbol] = {"score": 0.0, "label": "neutral", "error": str(e)}

    return {"timestamp": datetime.now().isoformat(), "sentiments": results}
