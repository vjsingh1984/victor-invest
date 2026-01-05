"""
XBRL Parser Module
Parses XBRL data from SEC filings
"""

import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class XBRLParser:
    """
    Parser for XBRL (eXtensible Business Reporting Language) data
    """

    def __init__(self):
        self.namespaces = {
            "xbrli": "http://www.xbrl.org/2003/instance",
            "us-gaap": "http://fasb.org/us-gaap/",
            "dei": "http://xbrl.sec.gov/dei/",
        }

    async def parse_filing(self, xbrl_content: str) -> Dict[str, Any]:
        """
        Parse XBRL content from a filing
        """
        try:
            if not xbrl_content:
                return {}

            # Try to parse as XML
            root = ET.fromstring(xbrl_content)

            parsed_data = {"financial_data": {}, "document_info": {}, "contexts": {}, "units": {}}

            # Extract basic document information
            parsed_data["document_info"] = self._extract_document_info(root)

            # Extract financial data
            parsed_data["financial_data"] = self._extract_financial_data(root)

            return parsed_data

        except ET.ParseError as e:
            logger.error(f"Failed to parse XBRL: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing XBRL: {e}")
            return {}

    def _extract_document_info(self, root: ET.Element) -> Dict:
        """Extract document information from XBRL"""
        doc_info = {}

        try:
            # Common DEI elements
            dei_elements = [
                "EntityRegistrantName",
                "EntityCentralIndexKey",
                "CurrentFiscalYearEndDate",
                "DocumentType",
                "DocumentPeriodEndDate",
                "AmendmentFlag",
            ]

            for element in dei_elements:
                nodes = root.findall(f".//*[@name='{element}']")
                if nodes:
                    doc_info[element] = nodes[0].text

        except Exception as e:
            logger.error(f"Error extracting document info: {e}")

        return doc_info

    def _extract_financial_data(self, root: ET.Element) -> Dict:
        """Extract financial data from XBRL"""
        financial_data = {}

        try:
            # Common financial elements
            financial_elements = [
                "Assets",
                "AssetsCurrent",
                "Liabilities",
                "LiabilitiesCurrent",
                "StockholdersEquity",
                "Revenues",
                "NetIncomeLoss",
                "EarningsPerShareBasic",
                "EarningsPerShareDiluted",
                "CashAndCashEquivalentsAtCarryingValue",
            ]

            for element in financial_elements:
                nodes = root.findall(f".//*[@name='us-gaap:{element}']")
                if nodes:
                    values = []
                    for node in nodes:
                        context_ref = node.get("contextRef", "")
                        unit_ref = node.get("unitRef", "")
                        value = node.text

                        values.append({"value": value, "context": context_ref, "unit": unit_ref})

                    if values:
                        financial_data[element] = values

        except Exception as e:
            logger.error(f"Error extracting financial data: {e}")

        return financial_data

    async def extract_metrics(self, parsed_data: Dict) -> Dict[str, float]:
        """
        Extract key financial metrics from parsed XBRL data
        """
        metrics = {}

        try:
            financial_data = parsed_data.get("financial_data", {})

            # Extract the most recent values for each metric
            for key, values in financial_data.items():
                if isinstance(values, list) and values:
                    # Take the first value (usually most recent)
                    try:
                        value = float(values[0]["value"])
                        metrics[key] = value
                    except (ValueError, KeyError):
                        pass

            # Calculate derived metrics if possible
            if "Assets" in metrics and "Liabilities" in metrics:
                metrics["Equity"] = metrics["Assets"] - metrics["Liabilities"]

            if "AssetsCurrent" in metrics and "LiabilitiesCurrent" in metrics:
                metrics["WorkingCapital"] = metrics["AssetsCurrent"] - metrics["LiabilitiesCurrent"]
                if metrics["LiabilitiesCurrent"] > 0:
                    metrics["CurrentRatio"] = metrics["AssetsCurrent"] / metrics["LiabilitiesCurrent"]

        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")

        return metrics

    def format_financial_statement(self, metrics: Dict[str, float]) -> str:
        """
        Format financial metrics into a readable statement
        """
        try:
            statement = []
            statement.append("Financial Summary:")
            statement.append("-" * 40)

            # Format metrics with proper labels
            metric_labels = {
                "Assets": "Total Assets",
                "AssetsCurrent": "Current Assets",
                "Liabilities": "Total Liabilities",
                "LiabilitiesCurrent": "Current Liabilities",
                "StockholdersEquity": "Stockholders Equity",
                "Revenues": "Revenue",
                "NetIncomeLoss": "Net Income",
                "EarningsPerShareBasic": "EPS (Basic)",
                "EarningsPerShareDiluted": "EPS (Diluted)",
                "CashAndCashEquivalentsAtCarryingValue": "Cash & Equivalents",
                "Equity": "Total Equity",
                "WorkingCapital": "Working Capital",
                "CurrentRatio": "Current Ratio",
            }

            for key, label in metric_labels.items():
                if key in metrics:
                    value = metrics[key]
                    if key in ["CurrentRatio"]:
                        statement.append(f"{label}: {value:.2f}")
                    elif key in ["EarningsPerShareBasic", "EarningsPerShareDiluted"]:
                        statement.append(f"{label}: ${value:.2f}")
                    else:
                        # Format large numbers
                        if abs(value) >= 1e9:
                            statement.append(f"{label}: ${value/1e9:.2f}B")
                        elif abs(value) >= 1e6:
                            statement.append(f"{label}: ${value/1e6:.2f}M")
                        else:
                            statement.append(f"{label}: ${value:,.0f}")

            return "\n".join(statement)

        except Exception as e:
            logger.error(f"Error formatting financial statement: {e}")
            return "Unable to format financial statement"
