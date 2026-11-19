"""Business-day exchange rates from Norges Bank's public SDMX API."""

from io import StringIO

import pandas as pd
import requests

EXCHANGE_API = "https://data.norges-bank.no/api/data/EXR"


def fetch_exchange_rates(
    start_date: str, end_date: str, currencies: tuple[str, ...] = ("EUR", "USD")
) -> pd.DataFrame:
    """Return dates and observed NOK per foreign currency unit."""
    if not currencies or set(currencies) - {"EUR", "USD"}:
        raise ValueError("Choose EUR, USD, or both.")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("The exchange-rate start date must precede the end date.")
    series = "+".join(sorted(set(currencies)))
    response = requests.get(
        f"{EXCHANGE_API}/B.{series}.NOK.SP",
        params={"format": "csv", "locale": "en", "startPeriod": start_date, "endPeriod": end_date},
        timeout=(10, 45),
    )
    response.raise_for_status()
    # English locale uses decimal points. Norges Bank still separates columns with semicolons.
    rows = pd.read_csv(StringIO(response.content.decode("utf-8-sig")), sep=";")
    result = pd.DataFrame({
        "date": pd.to_datetime(rows.TIME_PERIOD, format="%Y-%m-%d"),
        "pair": rows.BASE_CUR + "/NOK",
        "nok_per_unit": pd.to_numeric(rows.OBS_VALUE) * 10.0 ** pd.to_numeric(rows.UNIT_MULT),
    }).dropna()
    if result.duplicated(["date", "pair"]).any():
        raise ValueError("Norges Bank returned duplicate dates for a currency pair.")
    return result.sort_values(["date", "pair"]).reset_index(drop=True)
