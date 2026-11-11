"""Tests for VISHUSTRA's public API routes and endpoint behaviour."""

import pytest
from app.api.labs import AttentionRequest, analyze_attention, list_paper_configs
from app.main import app
from pydantic import ValidationError


def test_health_check_reports_service_status():
    routes = {route.path: route for route in app.routes}
    response = routes["/health"].endpoint()

    assert response.status == "ok"
    assert response.service == "vishustra"
    assert response.version == "0.2.0"


def test_paper_configs_expose_the_base_and_big_reference_models():
    configs = list_paper_configs()

    assert [config.name for config in configs] == ["base", "big"]
    assert configs[0].d_model == 512
    assert configs[1].num_heads == 16


def test_attention_analysis_returns_normalized_weights_and_all_calculation_steps():
    payload = {"tokens": ["the", "scientist", "studied"], "d_k": 4, "seed": 11}
    request = AttentionRequest(**payload)
    body = analyze_attention(request)

    assert body.tokens == payload["tokens"]
    assert body.d_k == 4
    assert len(body.scores) == len(payload["tokens"])
    assert len(body.output[0]) == 4
    for row in body.weights:
        assert abs(sum(row) - 1) < 1e-6


def test_attention_analysis_rejects_empty_tokens():
    with pytest.raises(ValidationError, match="tokens must not be empty"):
        AttentionRequest(tokens=["valid", " "])
