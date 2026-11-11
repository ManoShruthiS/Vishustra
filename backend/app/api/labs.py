"""HTTP endpoints for the first interactive VISHUSTRA laboratories.

The endpoints deliberately expose the intermediate values from scaled
dot-product attention.  A client can therefore render the calculation,
not merely display its final result.
"""

from __future__ import annotations

import math

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.core.config import paper_configs
from app.transformer import ScaledDotProductAttention

router = APIRouter(prefix="/api", tags=["laboratories"])


class PaperConfigResponse(BaseModel):
    """A reference Transformer configuration from the original paper."""

    name: str
    num_encoder_layers: int
    num_decoder_layers: int
    d_model: int
    d_ff: int
    num_heads: int
    d_k: int
    d_v: int
    dropout: float
    label_smoothing: float
    train_steps: int
    params_millions: int
    notes: list[str]


class AttentionRequest(BaseModel):
    """Inputs for a reproducible scaled dot-product attention calculation."""

    tokens: list[str] = Field(min_length=1, max_length=128)
    d_k: int = Field(default=8, ge=1, le=512)
    seed: int = Field(default=7, ge=0, le=2**63 - 1)

    @field_validator("tokens")
    @classmethod
    def tokens_must_have_text(cls, tokens: list[str]) -> list[str]:
        if any(not token.strip() for token in tokens):
            raise ValueError("tokens must not be empty")
        return tokens


class AttentionResponse(BaseModel):
    """The complete calculation a visualization needs to explain attention."""

    tokens: list[str]
    d_k: int
    scaler: float
    scores: list[list[float]]
    scaled_scores: list[list[float]]
    weights: list[list[float]]
    output: list[list[float]]


def _random_qkv(
    token_count: int, d_k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic Q, K and V tensors for the interactive lab."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, token_count, d_k)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    )


@router.get("/configs", response_model=list[PaperConfigResponse])
def list_paper_configs() -> list[PaperConfigResponse]:
    """Return the paper-locked base and big Transformer configurations."""
    return [
        PaperConfigResponse(
            name=config.name,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_heads=config.num_heads,
            d_k=config.d_k,
            d_v=config.d_v,
            dropout=config.dropout,
            label_smoothing=config.label_smoothing,
            train_steps=config.train_steps,
            params_millions=config.params_millions,
            notes=list(config.notes),
        )
        for config in paper_configs()
    ]


@router.post("/attention/analyze", response_model=AttentionResponse)
def analyze_attention(request: AttentionRequest) -> AttentionResponse:
    """Run Equation 1 and return every displayable matrix in the calculation."""
    query, key, value = _random_qkv(len(request.tokens), request.d_k, request.seed)
    attention = ScaledDotProductAttention(d_k=request.d_k)

    with torch.no_grad():
        output, trace = attention(query, key, value, token_ids=list(range(len(request.tokens))))

    return AttentionResponse(
        tokens=request.tokens,
        d_k=request.d_k,
        scaler=1 / math.sqrt(request.d_k),
        scores=trace.scores[0].tolist(),
        scaled_scores=trace.scaled[0].tolist(),
        weights=trace.weights[0].tolist(),
        output=output[0].tolist(),
    )
