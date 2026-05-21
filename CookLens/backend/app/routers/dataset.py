"""Dataset Router — synthetic dataset generation and management."""
import logging

from fastapi import APIRouter

from app.models.schemas import DatasetGenerateRequest, DatasetEntry
from app.services.dataset_service import dataset_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.post("/generate", response_model=list[DatasetEntry])
async def generate_dataset(request: DatasetGenerateRequest):
    """Generate synthetic dataset entries for fine-tuning."""
    entries = dataset_service.generate_synthetic_prompts(
        count=request.count,
        cuisine=request.cuisine_filter,
    )
    return entries


@router.get("/stats")
async def get_stats():
    """Return aggregate statistics over all generated dataset entries."""
    return dataset_service.get_stats()


@router.post("/export")
async def export_dataset():
    """Export all generated dataset entries as JSON."""
    return dataset_service.export_dataset()
