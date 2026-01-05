"""
Macro economic data commands for InvestiGator CLI
"""

import json
import sys
from datetime import date, datetime
from typing import Optional

import click


@click.group()
@click.pass_context
def macro(ctx):
    """Macroeconomic data and indicators

    Access Federal Reserve data, treasury yields, and economic indicators.

    Examples:
        investigator macro summary
        investigator macro indicators --category inflation
        investigator macro fed --district atlanta
        investigator macro treasury curve
    """
    pass


@macro.command("summary")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def summary(ctx, json_output):
    """Show macroeconomic summary dashboard

    Displays key economic indicators from all sources.
    """
    from investigator.domain.services.data_sources.facade import get_data_source_facade

    facade = get_data_source_facade()
    analysis_data = facade.get_historical_data_sync(
        symbol="_MACRO",
        as_of_date=date.today(),
    )

    regional_fed = analysis_data.regional_fed_indicators or {}
    cboe = analysis_data.cboe_data or {}

    if json_output:
        output = {
            "regional_fed": regional_fed,
            "cboe": cboe,
            "as_of_date": date.today().isoformat(),
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    click.echo("\n" + "=" * 60)
    click.echo("MACROECONOMIC SUMMARY")
    click.echo("=" * 60)
    click.echo(f"As of: {date.today().isoformat()}")

    # CBOE Volatility
    if cboe:
        click.echo("\nVOLATILITY (CBOE)")
        click.echo("-" * 40)
        vix = cboe.get("vix")
        skew = cboe.get("skew")
        vix3m = cboe.get("vix3m")
        regime = cboe.get("volatility_regime", "unknown")

        if vix:
            status = "HIGH" if vix > 25 else "ELEVATED" if vix > 20 else "LOW"
            click.echo(f"  VIX:              {vix:.2f} ({status})")
        if vix3m:
            click.echo(f"  VIX3M:            {vix3m:.2f}")
        if skew:
            skew_status = "ELEVATED" if skew > 130 else "Normal"
            click.echo(f"  SKEW:             {skew:.2f} ({skew_status})")
        click.echo(f"  Volatility Regime: {regime.upper()}")
        if cboe.get("is_backwardation"):
            click.echo("  VIX in BACKWARDATION (fear signal)")

    # Federal Reserve data
    fed_summary = regional_fed.get("summary", {}) if isinstance(regional_fed, dict) else {}
    if fed_summary:
        click.echo("\nECONOMIC ACTIVITY")
        click.echo("-" * 40)

        if fed_summary.get("gdpnow") is not None:
            gdp = fed_summary["gdpnow"]
            status = "NEGATIVE" if gdp < 0 else "STRONG" if gdp > 2 else "MODERATE"
            click.echo(f"  GDPNow (Atlanta Fed):     {gdp:.2f}% ({status})")

        if fed_summary.get("cfnai") is not None:
            cfnai = fed_summary["cfnai"]
            status = "CONTRACTION" if cfnai < -0.7 else "EXPANSION" if cfnai > 0 else "NEUTRAL"
            click.echo(f"  CFNAI (Chicago Fed):      {cfnai:.3f} ({status})")

        if fed_summary.get("empire_state_mfg") is not None:
            emp = fed_summary["empire_state_mfg"]
            status = "CONTRACTING" if emp < -10 else "EXPANDING" if emp > 10 else "NEUTRAL"
            click.echo(f"  Empire State Mfg:         {emp:.1f} ({status})")

        click.echo("\nFINANCIAL CONDITIONS")
        click.echo("-" * 40)

        if fed_summary.get("nfci") is not None:
            nfci = fed_summary["nfci"]
            status = "TIGHT" if nfci > 0.5 else "LOOSE" if nfci < 0 else "NEUTRAL"
            click.echo(f"  NFCI (Chicago Fed):       {nfci:.3f} ({status})")

        if fed_summary.get("kcfsi") is not None:
            kcfsi = fed_summary["kcfsi"]
            status = "STRESS" if kcfsi > 0.5 else "NORMAL" if kcfsi < 0 else "ELEVATED"
            click.echo(f"  KCFSI (Kansas City):      {kcfsi:.3f} ({status})")

        click.echo("\nINFLATION & RECESSION")
        click.echo("-" * 40)

        if fed_summary.get("inflation_expectations") is not None:
            infl = fed_summary["inflation_expectations"]
            status = "HIGH" if infl > 4 else "TARGET" if infl < 2.5 else "ABOVE TARGET"
            click.echo(f"  Inflation Expectations:   {infl:.2f}% ({status})")

        if fed_summary.get("recession_probability") is not None:
            rec = fed_summary["recession_probability"]
            rec_pct = rec * 100 if rec < 1 else rec
            status = "HIGH RISK" if rec_pct > 30 else "LOW RISK" if rec_pct < 15 else "MODERATE"
            click.echo(f"  Recession Probability:    {rec_pct:.1f}% ({status})")

    click.echo("\n" + "=" * 60)


@macro.command("fed")
@click.option(
    "--district",
    "-d",
    type=click.Choice(
        [
            "atlanta",
            "chicago",
            "cleveland",
            "dallas",
            "kansas_city",
            "new_york",
            "philadelphia",
            "richmond",
            "san_francisco",
            "st_louis",
        ]
    ),
    help="Filter by Federal Reserve district",
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def fed_data(ctx, district, json_output):
    """Show Federal Reserve regional indicators

    Displays economic indicators from Federal Reserve banks.

    Examples:
        investigator macro fed
        investigator macro fed --district atlanta
    """
    from investigator.domain.services.data_sources.facade import get_data_source_facade

    facade = get_data_source_facade()
    analysis_data = facade.get_historical_data_sync(
        symbol="_MACRO",
        as_of_date=date.today(),
    )

    regional_fed = analysis_data.regional_fed_indicators or {}

    if json_output:
        output = regional_fed
        if district:
            output = regional_fed.get("by_district", {}).get(district, {})
        click.echo(json.dumps(output, indent=2, default=str))
        return

    click.echo("\n" + "=" * 60)
    click.echo("FEDERAL RESERVE INDICATORS")
    click.echo("=" * 60)

    by_district = regional_fed.get("by_district", {}) if isinstance(regional_fed, dict) else {}

    if district:
        if district in by_district:
            click.echo(f"\n{district.upper().replace('_', ' ')} FED")
            click.echo("-" * 40)
            for indicator, data in by_district[district].items():
                val = data.get("value")
                if val is not None:
                    click.echo(f"  {indicator}: {val}")
        else:
            click.echo(f"No data for district: {district}")
            click.echo(f"Available: {', '.join(by_district.keys())}")
    else:
        for dist, indicators in by_district.items():
            click.echo(f"\n{dist.upper().replace('_', ' ')} FED")
            click.echo("-" * 40)
            for indicator, data in list(indicators.items())[:5]:  # Show top 5
                val = data.get("value")
                if val is not None:
                    click.echo(f"  {indicator}: {val}")

    click.echo("\n" + "=" * 60)


@macro.command("indicators")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["gdp", "inflation", "employment", "financial", "manufacturing"]),
    help="Filter by category",
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def indicators(ctx, category, json_output):
    """List economic indicators by category

    Shows available economic indicators organized by category.
    """
    # Define indicator categories
    categories = {
        "gdp": [
            ("GDPNow", "atlanta_fed", "Real-time GDP growth estimate"),
            ("CFNAI", "chicago_fed", "Chicago Fed National Activity Index"),
        ],
        "inflation": [
            ("Inflation Expectations", "cleveland_fed", "1-year inflation expectations"),
            ("Trimmed Mean PCE", "dallas_fed", "Core inflation measure"),
        ],
        "employment": [
            ("Empire State Mfg", "new_york_fed", "Manufacturing employment gauge"),
            ("Labor Market Conditions", "kansas_city_fed", "KCFSI labor component"),
        ],
        "financial": [
            ("NFCI", "chicago_fed", "National Financial Conditions Index"),
            ("KCFSI", "kansas_city_fed", "Financial Stress Index"),
            ("VIX", "cboe", "Market volatility index"),
            ("SKEW", "cboe", "Tail risk measure"),
        ],
        "manufacturing": [
            ("Empire State Mfg", "new_york_fed", "NY manufacturing survey"),
            ("Philly Fed", "philadelphia_fed", "Philadelphia manufacturing"),
        ],
    }

    if json_output:
        if category:
            output = {category: categories.get(category, [])}
        else:
            output = categories
        click.echo(json.dumps(output, indent=2))
        return

    click.echo("\n" + "=" * 60)
    click.echo("ECONOMIC INDICATORS CATALOG")
    click.echo("=" * 60)

    cats_to_show = [category] if category else categories.keys()

    for cat in cats_to_show:
        if cat in categories:
            click.echo(f"\n{cat.upper()}")
            click.echo("-" * 40)
            for name, source, desc in categories[cat]:
                click.echo(f"  {name}")
                click.echo(f"    Source: {source}")
                click.echo(f"    {desc}")

    click.echo("\n" + "=" * 60)


@macro.command("treasury")
@click.option("--view", type=click.Choice(["curve", "spread", "history"]), default="curve", help="View type")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def treasury(ctx, view, json_output):
    """Show Treasury yield curve and spreads

    Displays current Treasury yields and yield curve analysis.

    Examples:
        investigator macro treasury
        investigator macro treasury --view spread
    """
    from sqlalchemy import text

    from investigator.infrastructure.database.db import get_engine

    engine = get_engine()

    with engine.connect() as conn:
        # Get latest Treasury yields
        result = conn.execute(
            text(
                """
            SELECT series_id, value, observation_date
            FROM fred_economic_data
            WHERE series_id IN ('DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30')
            AND observation_date = (
                SELECT MAX(observation_date) FROM fred_economic_data
                WHERE series_id = 'DGS10'
            )
            ORDER BY
                CASE series_id
                    WHEN 'DGS1MO' THEN 1
                    WHEN 'DGS3MO' THEN 2
                    WHEN 'DGS6MO' THEN 3
                    WHEN 'DGS1' THEN 4
                    WHEN 'DGS2' THEN 5
                    WHEN 'DGS5' THEN 6
                    WHEN 'DGS10' THEN 7
                    WHEN 'DGS30' THEN 8
                END
        """
            )
        )
        yields = {row[0]: {"value": float(row[1]), "date": row[2]} for row in result}

    if not yields:
        click.echo("No Treasury data available")
        return

    if json_output:
        click.echo(json.dumps(yields, indent=2, default=str))
        return

    click.echo("\n" + "=" * 60)
    click.echo("TREASURY YIELD CURVE")
    click.echo("=" * 60)

    obs_date = list(yields.values())[0]["date"] if yields else "N/A"
    click.echo(f"As of: {obs_date}")

    if view == "curve":
        click.echo("\nYIELD CURVE")
        click.echo("-" * 40)
        labels = {
            "DGS1MO": "1 Month",
            "DGS3MO": "3 Month",
            "DGS6MO": "6 Month",
            "DGS1": "1 Year",
            "DGS2": "2 Year",
            "DGS5": "5 Year",
            "DGS10": "10 Year",
            "DGS30": "30 Year",
        }
        for series, label in labels.items():
            if series in yields:
                val = yields[series]["value"]
                bar = "*" * int(val * 5)
                click.echo(f"  {label:12s}: {val:5.2f}% {bar}")

    elif view == "spread":
        click.echo("\nKEY SPREADS")
        click.echo("-" * 40)

        y10 = yields.get("DGS10", {}).get("value", 0)
        y2 = yields.get("DGS2", {}).get("value", 0)
        y3m = yields.get("DGS3MO", {}).get("value", 0)

        spread_10_2 = y10 - y2
        spread_10_3m = y10 - y3m

        status_10_2 = "INVERTED" if spread_10_2 < 0 else "Normal"
        status_10_3m = "INVERTED" if spread_10_3m < 0 else "Normal"

        click.echo(f"  10Y-2Y Spread:  {spread_10_2:+.2f}% ({status_10_2})")
        click.echo(f"  10Y-3M Spread:  {spread_10_3m:+.2f}% ({status_10_3m})")

        if spread_10_2 < 0 or spread_10_3m < 0:
            click.echo("\n  NOTE: Inverted curve historically signals recession risk")

    click.echo("\n" + "=" * 60)


@macro.command("volatility")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def volatility(ctx, json_output):
    """Show CBOE volatility indicators

    Displays VIX, SKEW, and volatility regime analysis.
    """
    from investigator.domain.services.data_sources.facade import get_data_source_facade

    facade = get_data_source_facade()
    analysis_data = facade.get_historical_data_sync(
        symbol="_MACRO",
        as_of_date=date.today(),
    )

    cboe = analysis_data.cboe_data or {}

    if json_output:
        click.echo(json.dumps(cboe, indent=2, default=str))
        return

    click.echo("\n" + "=" * 60)
    click.echo("CBOE VOLATILITY INDICATORS")
    click.echo("=" * 60)

    vix = cboe.get("vix")
    vix3m = cboe.get("vix3m")
    skew = cboe.get("skew")
    regime = cboe.get("volatility_regime", "unknown")

    click.echo("\nVOLATILITY LEVELS")
    click.echo("-" * 40)

    if vix:
        if vix < 15:
            status = "LOW (complacency)"
        elif vix < 20:
            status = "NORMAL"
        elif vix < 30:
            status = "ELEVATED"
        else:
            status = "HIGH (fear)"
        click.echo(f"  VIX:    {vix:.2f} - {status}")

    if vix3m:
        click.echo(f"  VIX3M:  {vix3m:.2f}")

    if vix and vix3m:
        term_structure = vix3m / vix
        if term_structure < 1:
            ts_status = "BACKWARDATION (short-term fear)"
        else:
            ts_status = "CONTANGO (normal)"
        click.echo(f"  Term Structure: {term_structure:.2f} - {ts_status}")

    click.echo("\nTAIL RISK")
    click.echo("-" * 40)

    if skew:
        if skew > 140:
            status = "ELEVATED tail risk"
        elif skew > 130:
            status = "Moderate tail risk"
        else:
            status = "Normal"
        click.echo(f"  SKEW:   {skew:.2f} - {status}")

    click.echo(f"\nVOLATILITY REGIME: {regime.upper()}")

    click.echo("\n" + "=" * 60)
