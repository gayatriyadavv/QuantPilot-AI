"""
QuantPilot AI — Financial Sentiment Analyzer

Uses `ProsusAI/finbert <https://huggingface.co/ProsusAI/finbert>`_ to score
financial headlines as *positive*, *negative*, or *neutral*. Supports single
and batch analysis, aggregate sentiment over a time window, results caching,
and automatic GPU detection.

Usage::

    from backend.sentiment.analyzer import SentimentAnalyzer

    analyzer = SentimentAnalyzer()

    # Single headline
    result = analyzer.analyze_text("Apple reports record quarterly revenue")
    # {'positive': 0.93, 'negative': 0.02, 'neutral': 0.05,
    #  'label': 'positive', 'confidence': 0.93}

    # Batch
    results = analyzer.analyze_batch([
        "Markets crash on inflation fears",
        "Federal Reserve holds rates steady",
    ])
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import torch
from loguru import logger

from backend.config import get_settings

# Label mapping used by ProsusAI/finbert
_LABEL_MAP: Dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}


class SentimentAnalyzer:
    """FinBERT-based financial sentiment analyzer.

    Key design choices:

    * **Lazy loading** — the model + tokenizer are downloaded / loaded on
      first use so import time stays near zero.
    * **GPU-aware** — automatically uses CUDA when available.
    * **Caching** — recent results are kept in an in-memory LRU dict with
      configurable TTL to avoid redundant inference.

    Parameters:
        model_name: HuggingFace model ID. Defaults to ``Settings.finbert_model``.
        cache_ttl: Seconds to keep cached results. Defaults to
            ``Settings.sentiment_cache_ttl``.
        max_cache_size: Maximum number of entries in the cache before eviction.
        max_length: Maximum token length passed to the tokenizer.
        batch_size: Default batch size for ``analyze_batch``.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        max_cache_size: int = 10_000,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        settings = get_settings()

        self._model_name: str = model_name or settings.finbert_model
        self._cache_ttl: int = cache_ttl if cache_ttl is not None else settings.sentiment_cache_ttl
        self._max_cache_size: int = max_cache_size
        self._max_length: int = max_length
        self._batch_size: int = batch_size

        # Lazy-loaded model components
        self._model: Any = None
        self._tokenizer: Any = None

        # Device selection
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # In-memory cache: key → (result_dict, timestamp)
        self._cache: Dict[str, tuple[Dict[str, Any], float]] = {}

        logger.info(
            "SentimentAnalyzer configured | model={m} | device={d} | "
            "cache_ttl={ttl}s | max_cache={mc}",
            m=self._model_name,
            d=self._device,
            ttl=self._cache_ttl,
            mc=self._max_cache_size,
        )

    # ────────────────────────────────────────────────────────────
    # Model Loading
    # ────────────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Load the model and tokenizer on first use (lazy init)."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading FinBERT model: {m} …", m=self._model_name)
        start = time.monotonic()

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )
        self._model.to(self._device)
        self._model.eval()

        elapsed = time.monotonic() - start
        logger.info(
            "FinBERT loaded in {t:.1f}s on {d}",
            t=elapsed,
            d=self._device,
        )

    # ────────────────────────────────────────────────────────────
    # Single-Text Analysis
    # ────────────────────────────────────────────────────────────

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze a single headline or sentence.

        Args:
            text: The financial headline to analyze.

        Returns:
            Dictionary with keys:
                - ``positive`` (float): Positive class probability.
                - ``negative`` (float): Negative class probability.
                - ``neutral`` (float): Neutral class probability.
                - ``label`` (str): Winning class name.
                - ``confidence`` (float): Probability of the winning class.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to analyze_text")
            return self._empty_result()

        # Check cache
        cache_key = self._cache_key(text)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug("Cache hit for: {t!r:.60}", t=text)
            return cached

        # Run inference
        self._ensure_model_loaded()
        result = self._infer([text])[0]

        self._put_cache(cache_key, result)
        return result

    # ────────────────────────────────────────────────────────────
    # Batch Analysis
    # ────────────────────────────────────────────────────────────

    def analyze_batch(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze multiple headlines efficiently.

        Automatically batches the texts to keep GPU memory usage bounded and
        checks the cache for each text individually.

        Args:
            texts: Iterable of headline strings.
            batch_size: Override the default batch size.

        Returns:
            List of result dictionaries, one per input text (same order).
        """
        if not texts:
            return []

        bs = batch_size or self._batch_size
        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Resolve cache hits
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = self._empty_result()
                continue
            cache_key = self._cache_key(text)
            cached = self._get_cached(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Run inference for uncached texts in batches
        if uncached_texts:
            self._ensure_model_loaded()
            logger.info(
                "Batch inference | total={n} | uncached={u} | batch_size={bs}",
                n=len(texts),
                u=len(uncached_texts),
                bs=bs,
            )

            inferred: List[Dict[str, Any]] = []
            for start in range(0, len(uncached_texts), bs):
                batch = uncached_texts[start : start + bs]
                batch_results = self._infer(batch)
                inferred.extend(batch_results)

            for idx, result in zip(uncached_indices, inferred):
                results[idx] = result
                self._put_cache(self._cache_key(texts[idx]), result)

        return results  # type: ignore[return-value]

    # ────────────────────────────────────────────────────────────
    # Aggregate Sentiment
    # ────────────────────────────────────────────────────────────

    def get_aggregate_sentiment(
        self,
        headlines: Sequence[str],
        symbol: Optional[str] = None,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Compute average sentiment across a collection of headlines.

        This is typically used to get the overall market sentiment for a
        symbol over a recent time window. The caller is responsible for
        filtering headlines by time before passing them in; the *hours*
        parameter is recorded in the output for context.

        Args:
            headlines: Sequence of headline strings.
            symbol: Optional ticker symbol for labelling.
            hours: Time window in hours (informational).

        Returns:
            Dictionary with:
                - ``symbol``: The symbol (or ``None``).
                - ``hours``: The time window.
                - ``count``: Number of headlines analysed.
                - ``avg_positive``: Mean positive score.
                - ``avg_negative``: Mean negative score.
                - ``avg_neutral``: Mean neutral score.
                - ``overall_label``: Label with highest average score.
                - ``overall_confidence``: The highest average score.
                - ``sentiment_score``: A single scalar in [-1, 1] computed as
                  ``avg_positive − avg_negative``.
                - ``individual_results``: Full per-headline results.
        """
        if not headlines:
            return {
                "symbol": symbol,
                "hours": hours,
                "count": 0,
                "avg_positive": 0.0,
                "avg_negative": 0.0,
                "avg_neutral": 0.0,
                "overall_label": "neutral",
                "overall_confidence": 0.0,
                "sentiment_score": 0.0,
                "individual_results": [],
            }

        results = self.analyze_batch(list(headlines))

        avg_pos = sum(r["positive"] for r in results) / len(results)
        avg_neg = sum(r["negative"] for r in results) / len(results)
        avg_neu = sum(r["neutral"] for r in results) / len(results)

        scores = {"positive": avg_pos, "negative": avg_neg, "neutral": avg_neu}
        overall_label = max(scores, key=scores.get)  # type: ignore[arg-type]
        overall_confidence = scores[overall_label]
        sentiment_score = avg_pos - avg_neg  # [-1, 1]

        logger.info(
            "Aggregate sentiment | sym={sym} | count={n} | label={lbl} "
            "({conf:.2f}) | score={sc:+.3f}",
            sym=symbol,
            n=len(results),
            lbl=overall_label,
            conf=overall_confidence,
            sc=sentiment_score,
        )

        return {
            "symbol": symbol,
            "hours": hours,
            "count": len(results),
            "avg_positive": round(avg_pos, 4),
            "avg_negative": round(avg_neg, 4),
            "avg_neutral": round(avg_neu, 4),
            "overall_label": overall_label,
            "overall_confidence": round(overall_confidence, 4),
            "sentiment_score": round(sentiment_score, 4),
            "individual_results": results,
        }

    # ────────────────────────────────────────────────────────────
    # Internal Inference
    # ────────────────────────────────────────────────────────────

    def _infer(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run model inference on a list of texts.

        Args:
            texts: List of headline strings (already filtered for blanks).

        Returns:
            List of result dictionaries.
        """
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        results: List[Dict[str, Any]] = []
        for probs in probabilities:
            probs_list = probs.cpu().tolist()
            label_idx = int(probs.argmax())
            label = _LABEL_MAP.get(label_idx, "neutral")
            confidence = probs_list[label_idx]
            results.append(
                {
                    "positive": round(probs_list[0], 4),
                    "negative": round(probs_list[1], 4),
                    "neutral": round(probs_list[2], 4),
                    "label": label,
                    "confidence": round(confidence, 4),
                }
            )
        return results

    # ────────────────────────────────────────────────────────────
    # Caching
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(text: str) -> str:
        """Create a deterministic cache key from text."""
        normalised = text.strip().lower()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a result from cache if it exists and hasn't expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        result, ts = entry
        if time.time() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        return result

    def _put_cache(self, key: str, result: Dict[str, Any]) -> None:
        """Store a result in the cache, evicting oldest entries if full."""
        if len(self._cache) >= self._max_cache_size:
            # Evict oldest 10%
            evict_count = max(1, self._max_cache_size // 10)
            sorted_keys = sorted(self._cache, key=lambda k: self._cache[k][1])
            for k in sorted_keys[:evict_count]:
                del self._cache[k]

        self._cache[key] = (result, time.time())

    def clear_cache(self) -> None:
        """Clear the entire sentiment cache."""
        self._cache.clear()
        logger.info("Sentiment cache cleared")

    @property
    def cache_size(self) -> int:
        """Current number of cached results."""
        return len(self._cache)

    # ────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return a default result for empty/invalid input."""
        return {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "label": "neutral",
            "confidence": 1.0,
        }

    @property
    def device(self) -> str:
        """Device the model is running on."""
        return str(self._device)

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return self._model is not None

    def __repr__(self) -> str:
        return (
            f"<SentimentAnalyzer model={self._model_name!r} "
            f"device={self._device} loaded={self.is_loaded} "
            f"cache={self.cache_size}>"
        )
