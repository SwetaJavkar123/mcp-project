"""Tests for agents/market_data_agent.py — data fetching (mocked)."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestGetData:
    @patch("agents.market_data_agent.yf.download")
    def test_returns_dataframe(self, mock_download, sample_prices):
        mock_download.return_value = sample_prices
        from agents.market_data_agent import get_data

        result = get_data("AAPL", start="2025-01-01", end="2025-06-01")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_download.assert_called_once_with("AAPL", start="2025-01-01", end="2025-06-01")

    @patch("agents.market_data_agent.yf.download")
    def test_empty_result(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        from agents.market_data_agent import get_data

        result = get_data("INVALID")
        assert result.empty

    @patch("agents.market_data_agent.yf.download")
    def test_no_dates_passed(self, mock_download, sample_prices):
        mock_download.return_value = sample_prices
        from agents.market_data_agent import get_data

        get_data("AAPL")
        mock_download.assert_called_once_with("AAPL", start=None, end=None)
