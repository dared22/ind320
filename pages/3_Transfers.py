"""Query published MongoDB data; never start ingestion on Streamlit reruns."""

import calendar

import matplotlib.pyplot as plt
import streamlit as st

from pymongo.errors import (
    ConfigurationError,
    ConnectionFailure,
    OperationFailure,
    PyMongoError,
)
from ind320.storage.mongo_store import available_transfers, query_annual, query_months
from ind320.ui.transfer_charts import energy_pie, transfer_lines

# Cache database reads briefly, separately from widget choices and local processing.
choices = st.cache_data(ttl=300)(available_transfers)
annual = st.cache_data(ttl=300)(query_annual)
intervals = st.cache_data(ttl=300)(query_months)
st.title("Cross-border electricity transfers")
st.caption("Physical flows · transferred energy, not electricity production")
left, right = st.columns([1, 2])
try:
    catalog = choices()
    if catalog.empty:
        st.info(
            "No transfers published yet. Run the notebook ingestion and publication cells."
        )
    else:
        with left:
            connection = st.radio("Connection", sorted(catalog.connection.unique()))
            years = sorted(
                catalog.loc[catalog.connection == connection, "year"].unique().tolist()
            )
            year = st.selectbox(
                "Year", years, index=years.index(2025) if 2025 in years else 0
            )
            summary = annual(connection, year)
            if summary.empty:
                st.info("No annual summary is available.")
            else:
                if not summary.complete.all():
                    st.warning(
                        "Incomplete coverage: energy totals include observed intervals only."
                    )
                st.caption("Totals include all observed months of the selected year.")
                figure = energy_pie(summary)
                st.pyplot(figure)
                plt.close(figure)
        with right:
            months = st.pills(
                "Months",
                list(range(1, 13)),
                selection_mode="multi",
                default=[1],
                format_func=lambda m: calendar.month_name[m],
            )
            if not months:
                st.info("Select one or more months to see directional flows.")
            else:
                data = intervals(connection, year, months)
                if data.empty:
                    st.info("No observations are available for these months.")
                else:
                    figure = transfer_lines(data)
                    st.pyplot(figure)
                    plt.close(figure)
                    st.caption(
                        f"{len(data):,} observed intervals. Gaps and omitted months are not connected."
                    )
except (PyMongoError, ValueError) as error:
    # Distinguish connection failures without exposing the URI or driver message.
    # Driver messages can include private hostnames and connection details.
    if isinstance(error, OperationFailure) and error.code == 18:
        detail = "MongoDB authentication failed. Check the database user's credentials."
    elif isinstance(error, OperationFailure) and error.code == 13:
        detail = "MongoDB denied access. Check the database user's read permissions."
    elif isinstance(error, ConnectionFailure):
        detail = (
            "MongoDB network connection failed. Check Atlas Network Access for "
            "the Streamlit Cloud connection, and check DNS/TLS connectivity."
        )
    elif isinstance(error, ConfigurationError):
        detail = "MongoDB connection configuration failed. Check the URI and DNS settings."
    elif isinstance(error, ValueError):
        detail = "Check the MongoDB settings and the format of the published transfer data."
    else:
        detail = "The MongoDB read failed. Check the database service and user permissions."
    st.error(f"Could not read MongoDB. {detail}")
    # Only show the exception class and numeric code, never the raw exception.
    code = getattr(error, "code", None)
    code_label = f", code {code}" if isinstance(code, int) else ""
    st.caption(f"Error type: {type(error).__name__}{code_label}")
# Keep the required source explanation visible even when credentials are missing.
with st.expander("Sources, coverage, and processing"):
    st.write(
        "ENTSO-E A11 cross-border physical flows. Import flows into Norway; export flows out. "
        "Energy (MWh) equals power (MW) × actual interval hours. UTC storage and Europe/Oslo "
        "month/year boundaries account for daylight saving. Missing data are never replaced by zero. "
        "The notebook retrieves monthly chunks for 2024–2025, writes and reads Cassandra with Spark, "
        "and upserts MongoDB intervals and summaries. This page reads only MongoDB."
    )
    st.link_button(
        "ENTSO-E API reference",
        "https://documenter.getpostman.com/view/7009892/2s93JtP3F6",
    )
