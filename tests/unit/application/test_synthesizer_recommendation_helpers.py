from unittest.mock import MagicMock

from investigator.application.synthesizer_recommendation import (
    calculate_consistency_bonus,
    calculate_price_target,
    calculate_stop_loss,
    determine_final_recommendation,
    extract_catalysts,
    extract_position_size,
)


def test_determine_final_recommendation_downgrades_strong_for_low_quality():
    result = determine_final_recommendation(
        overall_score=7.2,
        ai_recommendation={"recommendation": "STRONG BUY", "confidence": "HIGH"},
        data_quality=0.3,
    )
    assert result["recommendation"] == "BUY"
    assert result["confidence"] == "LOW"


def test_calculate_price_target_uses_score_mapping_and_logs_when_price_missing():
    logger = MagicMock()
    recommendation = {"composite_scores": {"overall_score": 8.1}}
    price_target = calculate_price_target("AAPL", recommendation, 0.0, logger)
    assert price_target == 115.0
    logger.warning.assert_called_once()
    logger.info.assert_called_once()


def test_stop_loss_and_position_catalyst_extractors():
    stop = calculate_stop_loss(100.0, {"recommendation": "BUY"}, 3.0)
    assert stop == 95.0

    size = extract_position_size({"investment_recommendation": {"position_sizing": {"recommended_weight": 0.051}}})
    assert size == "LARGE"

    catalysts = extract_catalysts(
        {
            "key_catalysts": [
                {"catalyst": "Margin expansion"},
                "Product cycle",
                {"catalyst": "Buybacks"},
                {"catalyst": "Ignored due to cap"},
            ]
        }
    )
    assert catalysts == ["Margin expansion", "Product cycle", "Buybacks"]


def test_calculate_consistency_bonus_bounds():
    assert calculate_consistency_bonus([7.0]) == 0.0
    assert 0.0 <= calculate_consistency_bonus([7.0, 7.1, 6.9, 7.05]) <= 1.0
