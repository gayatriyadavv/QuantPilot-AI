"""Dataset Service — synthetic data generation for fine-tuning."""
import random
import uuid
from collections import Counter

from app.data.mock_responses import MOCK_RESPONSES, get_random_mock, get_mock_by_cuisine
from app.models.schemas import DatasetEntry, AnalysisResponse, CookingStageEnum


# Prompt templates used to create varied synthetic prompts
_PROMPT_TEMPLATES = [
    "Analyze this image of {dish} being {stage_verb} in a kitchen.",
    "What dish is this? It looks like {dish}. Tell me the ingredients and cooking stage.",
    "I'm cooking {dish}. How far along am I and what should I do next?",
    "Identify the ingredients and cooking progress in this image of {dish}.",
    "Help me with this {cuisine} dish — it appears to be {dish}.",
]

_STAGE_VERBS = {
    CookingStageEnum.RAW: "prepared",
    CookingStageEnum.PREP: "prepped",
    CookingStageEnum.COOKING: "cooked",
    CookingStageEnum.ALMOST_DONE: "nearly finished",
    CookingStageEnum.DONE: "served",
    CookingStageEnum.OVERCOOKED: "overcooked",
}

_DIFFICULTIES = ["easy", "medium", "hard"]


class DatasetService:
    """Generates synthetic dataset entries from mock responses."""

    def __init__(self):
        self._entries: list[DatasetEntry] = []

    def generate_synthetic_prompts(
        self,
        count: int = 10,
        cuisine: str | None = None,
    ) -> list[DatasetEntry]:
        """Create *count* synthetic dataset entries, optionally filtered by cuisine."""
        entries: list[DatasetEntry] = []

        for i in range(count):
            if cuisine:
                mock = get_mock_by_cuisine(cuisine)
            else:
                mock = get_random_mock()

            stage_verb = _STAGE_VERBS.get(mock.stage, "cooked")
            template = random.choice(_PROMPT_TEMPLATES)
            prompt = template.format(
                dish=mock.dish_prediction.name,
                stage_verb=stage_verb,
                cuisine=mock.dish_prediction.cuisine,
            )

            entry = DatasetEntry(
                image_path=f"datasets/synthetic/image_{i + 1:03d}.jpg",
                prompt=prompt,
                response=mock,
                metadata={
                    "source": "synthetic",
                    "cuisine": mock.dish_prediction.cuisine,
                    "difficulty": random.choice(_DIFFICULTIES),
                    "stage": mock.stage.value,
                },
            )
            entries.append(entry)

        self._entries.extend(entries)
        return entries

    def export_dataset(self, entries: list[DatasetEntry] | None = None) -> dict:
        """Export entries (or all stored entries) as a JSON-serializable dict."""
        data = entries if entries is not None else self._entries
        return {
            "count": len(data),
            "entries": [e.model_dump() for e in data],
        }

    def get_stats(self) -> dict:
        """Return aggregate statistics over all stored entries."""
        if not self._entries:
            return {"total": 0, "by_cuisine": {}, "by_stage": {}, "by_difficulty": {}}

        cuisines = Counter(e.metadata.get("cuisine", "Unknown") for e in self._entries)
        stages = Counter(e.metadata.get("stage", "unknown") for e in self._entries)
        difficulties = Counter(e.metadata.get("difficulty", "unknown") for e in self._entries)

        return {
            "total": len(self._entries),
            "by_cuisine": dict(cuisines),
            "by_stage": dict(stages),
            "by_difficulty": dict(difficulties),
        }


# Module-level singleton
dataset_service = DatasetService()
