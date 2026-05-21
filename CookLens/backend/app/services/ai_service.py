"""AI Service — wraps OpenAI (or compatible) vision API with mock fallback."""
import base64
import json
import logging
import uuid

from openai import AsyncOpenAI

from app.config import settings
from app.data.mock_responses import get_random_mock, get_mock_by_cuisine
from app.models.schemas import (
    AnalysisResponse,
    Ingredient,
    CookingStep,
    DishPrediction,
    CopilotSuggestion,
    NutritionEstimate,
    SpiceRecommendation,
    CookingStageEnum,
)

logger = logging.getLogger(__name__)


ANALYSIS_SYSTEM_PROMPT = """You are CookLens, an expert culinary AI assistant.
Your goal is to accurately detect food, ingredients, and cooking stages, even in blurry, low-light, or messy kitchen photos.

# CHAIN OF THOUGHT REASONING
You MUST analyze the image in this exact order before predicting the dish:
1. Assess image quality (blurry, oily, low-light). If poor, lower your confidence scores.
2. Identify core ingredients based on texture and shape (e.g., differentiate rice from pasta, or egg from avocado).
3. Identify cuisine-specific markers, especially for Indian cuisine (e.g., gravies, separated oil, turmeric stains, paneer cubes, roti textures).
4. Determine the cooking stage.
5. Finally, predict the dish based on the above evidence.

Return ONLY a valid JSON object matching this structure (no markdown, no text outside JSON):
{
  "dish_prediction": {
    "reasoning": "Step 1: Image is slightly blurry but shows a pan. Step 2: Red gravy with white cubes (paneer). Step 3: Oil is separating at the edges, typical of Indian curries. Step 4: Cooking is midway. Step 5: Likely Paneer Tikka Masala.",
    "name": "...", 
    "cuisine": "...", 
    "confidence": 0.0-1.0, 
    "description": "...",
    "alternatives": ["Alternative Dish 1", "Alternative Dish 2"]
  },
  "stage": "raw|chopped|preparation|sautéing|simmering|frying|boiling|gravy_thickening|oil_separating|cooking|almost_done|done|plated|overcooked",
  "ingredients": [{"name": "...", "confidence": 0.0-1.0, "emoji": "...", "quantity": "..."}],
  "instructions": [{"step_number": 1, "instruction": "...", "duration": "...", "temperature": "...", "tip": "..."}],
  "remaining_time": "...",
  "tips": ["..."],
  "copilot": {"next_step": "...", "mistakes_detected": ["..."], "fixes": ["..."], "readiness_percent": 0-100, "readiness_label": "..."},
  "nutrition": {"calories": 0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0, "tags": ["..."]},
  "spice_recommendations": [{"name": "...", "emoji": "...", "reason": "..."}],
  "cuisine_style": "..."
}

CRITICAL RULES:
- If confidence is < 0.7, provide 2-3 realistic `alternatives`.
- If the image is extremely ambiguous, do NOT hallucinate a specific complex dish. Output a generic description (e.g., "Sautéed Vegetables") and low confidence.
- Be highly accurate distinguishing Indian dishes (e.g., Biryani vs Pulao, Dosa vs Crepe)."""


RECIPE_SYSTEM_PROMPT = """You are CookLens, an expert culinary AI assistant.
Given the following food analysis, generate a more detailed recipe adjusted for {servings} servings.
Keep the same JSON structure but enhance the instructions with more detail and adjust quantities.
Return ONLY valid JSON — no markdown fences, no explanation."""


class AIService:
    """Handles AI-powered image analysis and recipe generation."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY or "mock-key",
            base_url=settings.OPENAI_BASE_URL,
        )
        self.model = settings.AI_MODEL

    async def analyze_image(
        self,
        image_bytes: bytes,
        language: str = "en",
        cuisine_hint: str | None = None,
    ) -> AnalysisResponse:
        """Analyze a food image and return structured cooking information."""

        # Fast-path: use mock data
        if settings.USE_MOCK_DATA:
            logger.info("USE_MOCK_DATA is True — returning mock response")
            if cuisine_hint:
                return get_mock_by_cuisine(cuisine_hint)
            return get_random_mock()

        # Real API path
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            user_content: list[dict] = [
                {
                    "type": "text",
                    "text": self._build_analysis_prompt(language, cuisine_hint),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=2048,
                temperature=0.4,
            )

            raw_text = response.choices[0].message.content or ""
            return self._parse_analysis(raw_text, language)

        except Exception as exc:
            logger.warning("AI API call failed (%s) — falling back to mock", exc)
            if cuisine_hint:
                return get_mock_by_cuisine(cuisine_hint)
            return get_random_mock()

    async def generate_recipe(
        self,
        analysis: AnalysisResponse,
        servings: int = 2,
    ) -> AnalysisResponse:
        """Enhance an existing analysis with a more detailed recipe for *servings*."""

        if settings.USE_MOCK_DATA:
            # Scale quantities naively and return
            return self._scale_mock_recipe(analysis, servings)

        try:
            prompt = RECIPE_SYSTEM_PROMPT.format(servings=servings)
            analysis_json = analysis.model_dump_json()

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": analysis_json},
                ],
                max_tokens=2048,
                temperature=0.4,
            )

            raw_text = response.choices[0].message.content or ""
            return self._parse_analysis(raw_text, analysis.language)

        except Exception as exc:
            logger.warning("Recipe generation failed (%s) — returning scaled mock", exc)
            return self._scale_mock_recipe(analysis, servings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_analysis_prompt(language: str, cuisine_hint: str | None) -> str:
        parts = ["Please analyze this food image."]
        if cuisine_hint:
            parts.append(f"The cuisine is likely: {cuisine_hint}.")
        if language != "en":
            parts.append(f"Respond in {language} where possible.")
        return " ".join(parts)

    @staticmethod
    def _parse_analysis(raw_text: str, language: str = "en") -> AnalysisResponse:
        """Parse raw JSON text from the AI into an AnalysisResponse."""
        # Strip markdown fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)

        return AnalysisResponse(
            id=str(uuid.uuid4()),
            ingredients=[Ingredient(**i) for i in data.get("ingredients", [])],
            stage=CookingStageEnum(data.get("stage", "cooking")),
            dish_prediction=DishPrediction(**data.get("dish_prediction", {
                "name": "Unknown Dish",
                "cuisine": "Unknown",
                "confidence": 0.5,
            })),
            instructions=[CookingStep(**s) for s in data.get("instructions", [])],
            remaining_time=data.get("remaining_time", "Unknown"),
            tips=data.get("tips", []),
            copilot=CopilotSuggestion(**data.get("copilot", {
                "next_step": "Continue cooking",
                "readiness_percent": 50.0,
                "readiness_label": "In Progress",
            })),
            nutrition=NutritionEstimate(**data.get("nutrition", {
                "calories": 0,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
                "fiber_g": 0,
            })),
            spice_recommendations=[
                SpiceRecommendation(**s) for s in data.get("spice_recommendations", [])
            ],
            cuisine_style=data.get("cuisine_style", "International"),
            language=language,
        )

    @staticmethod
    def _scale_mock_recipe(analysis: AnalysisResponse, servings: int) -> AnalysisResponse:
        """Return a copy of the analysis adjusted for serving count."""
        scaled = analysis.model_copy(deep=True)
        scaled.id = str(uuid.uuid4())
        # Add a note about scaling
        scaled.tips = [
            f"Recipe scaled for {servings} serving(s).",
            *scaled.tips,
        ]
        return scaled


# Module-level singleton
ai_service = AIService()
