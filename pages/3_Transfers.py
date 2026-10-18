"""Query published MongoDB data; never start ingestion on Streamlit reruns."""

import calendar

import matplotlib.pyplot as plt
import streamlit as st

from config import ConfigurationError
from mongo_store import available_transfers, query_annual, query_coverage, query_months
from transfer_charts import energy_pie, transfer_lines
from transfers import TransferError

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
                figure = energy_pie(summary)
                st.pyplot(figure)
                plt.close(figure)
                st.dataframe(
                    summary[
                        [
                            "direction",
                            "energy_mwh",
                            "observed_hours",
                            "expected_hours",
                            "complete",
                        ]
                    ],
                    hide_index=True,
                )
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
                coverage = query_coverage(connection, year, months)
                if not coverage.empty:
                    st.dataframe(
                        coverage[
                            [
                                "month",
                                "direction",
                                "observed_hours",
                                "expected_hours",
                                "complete",
                                "message",
                            ]
                        ],
                        hide_index=True,
                    )
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
except (ConfigurationError, TransferError) as exc:
    st.error(str(exc))
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
