"""Cache external reads independently of Streamlit display controls."""

import pandas as pd
import streamlit as st

from ind320.data.exchange import fetch_exchange_rates
from ind320.data.weather import fetch_weather
from ind320.data.weeks import WEEK, weather_week_range, weekly_fx, weekly_flows
from ind320.storage.mongo_store import available_transfers, query_annual, query_months

transfer_catalog = st.cache_data(ttl=300)(available_transfers)
transfer_annual = st.cache_data(ttl=300)(query_annual)
transfer_months = st.cache_data(ttl=300)(query_months)


@st.cache_data(ttl=86400, show_spinner="Loading the ERA5 weather archive. First downloads may take several minutes.")
def weather_for_year(area: str, year: int) -> pd.DataFrame:
    """Use full years for the daily quality analyses."""
    return fetch_weather(area, f"{year}-01-01", f"{year}-12-31")


@st.cache_data(ttl=86400, show_spinner="Loading weekly ERA5 weather…")
def weather_weeks(area: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Bound each request to a year while keeping weeks intact across New Year."""
    weeks = pd.date_range(first, last, freq="W-SUN")
    return pd.concat([
        weather_week_range(area, group.index[0], group.index[-1])
        for _, group in pd.Series(weeks.year, index=weeks).groupby(weeks.year)
    ]) if len(weeks) else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading Norges Bank rates…")
def exchange_weeks(first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    # The one-week lag needs quotes from the business days before that Sunday.
    return weekly_fx(fetch_exchange_rates(str((first - 2 * WEEK).date()), str(last.date())))


@st.cache_data(ttl=300, show_spinner="Reading weekly net imports…")
def flow_weeks(connection: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    months = pd.period_range(first - WEEK, last + WEEK, freq="M")
    frames = [transfer_months(connection, int(year), group.dt.month.tolist())
              for year, group in pd.Series(months).groupby(months.year)]
    return weekly_flows(pd.concat(frames, ignore_index=True))
