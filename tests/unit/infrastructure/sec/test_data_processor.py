import math

from investigator.infrastructure.sec.data_processor import SECDataProcessor


def _processor():
    # Bypass __init__ to avoid eager DB connections during unit tests
    return SECDataProcessor.__new__(SECDataProcessor)


def test_enrich_debt_fields_derives_short_and_net_debt():
    processor = _processor()
    filing = {
        "data": {
            "long_term_debt": 7_000_000_000.0,
            "total_debt": 7_800_000_000.0,
            "short_term_debt": None,
            "cash_and_equivalents": 950_000_000.0,
        }
    }

    processor._enrich_debt_fields(filing)

    derived_short = filing["data"]["short_term_debt"]
    net_debt = filing["data"]["net_debt"]

    assert math.isclose(derived_short, 800_000_000.0)
    assert math.isclose(net_debt, 6_850_000_000.0)


def test_enrich_debt_fields_backfills_total_from_components():
    processor = _processor()
    filing = {
        "data": {
            "long_term_debt": 5_100_000_000.0,
            "short_term_debt": 600_000_000.0,
            "total_debt": None,
        }
    }

    processor._enrich_debt_fields(filing)

    assert math.isclose(filing["data"]["total_debt"], 5_700_000_000.0)


def test_enrich_share_counts_uses_diluted_weighted_average():
    processor = _processor()
    filing = {
        "data": {
            "shares_outstanding": None,
            "weighted_average_diluted_shares_outstanding": 1_234_000_000.0,
        }
    }

    processor._enrich_share_counts(filing)

    assert filing["data"]["shares_outstanding"] == 1_234_000_000.0


def test_enrich_book_value_per_share_uses_equity_and_shares():
    processor = _processor()
    filing = {
        "data": {
            "stockholders_equity": 212_000_000.0,
            "shares_outstanding": 822_500_000.0,
            "book_value_per_share": None,
        }
    }

    processor._enrich_book_value_per_share(filing)

    equity = filing["data"]["stockholders_equity"]
    shares = filing["data"]["shares_outstanding"]
    expected = equity / shares

    assert math.isclose(filing["data"]["book_value_per_share"], expected, rel_tol=1e-9)


def test_enrich_debt_fields_uses_financial_deposit_heuristics():
    processor = _processor()
    filing = {
        "data": {
            "financial_total_deposits": 900_000_000.0,
            "financial_fhlb_borrowings": 125_000_000.0,
            "long_term_debt": 275_000_000.0,
            "total_debt": None,
            "short_term_debt": None,
        }
    }

    processor._enrich_debt_fields(filing)

    assert math.isclose(filing["data"]["total_debt"], 1_300_000_000.0)


def test_enrich_debt_fields_sets_short_term_from_repo_components():
    processor = _processor()
    filing = {
        "data": {
            "financial_repo_borrowings": 200_000_000.0,
            "financial_other_short_term_borrowings": 50_000_000.0,
            "total_debt": None,
            "long_term_debt": None,
            "short_term_debt": None,
        }
    }

    processor._enrich_debt_fields(filing)

    assert math.isclose(filing["data"]["short_term_debt"], 250_000_000.0)
    assert math.isclose(filing["data"]["total_debt"], 250_000_000.0)
