# IND320 — Data to Decision

An NMBU course project exploring Norwegian reservoir filling, cross-border
physical electricity flows, weather quality, and reservoir forecasts. The
Jupyter notebooks document the methods and checks. Streamlit uses the same
analysis functions.

- [Live Streamlit app](https://cklqlkatpe9mqyrnuch2ma.streamlit.app/)
- [Public repository](https://github.com/dared22/ind320)
- [Official course material](https://khliland.github.io/IND320/)

## Run locally

Use Python **3.10** and [uv](https://docs.astral.sh/uv/). Run from the repository root:

```sh
uv sync --locked --group notebook
uv run streamlit run app.py
```

For the dashboard alone, `uv sync --locked` omits notebook/GIS/Spark dependencies.
Reservoir, weather, and exchange-rate downloads use public APIs and need internet
access. The first weather download can be slow or rate-limited because it covers
many grid cells. Completed cells are cached under `.local/weather/`; rerunning
resumes downloads without showing an incomplete area average. Cloud caches can
be lost on app restart.

## Project structure

| Path | Contents |
| --- | --- |
| `app.py`, `pages/` | Map & Overview, Explore & Quality, and Forecast |
| `notebooks/project_notebook_ca1.ipynb` | Part 1 exploration; now retrieves the source through the API instead of a missing CSV |
| `notebooks/project_notebook.ipynb` | Part 2: NVE, ENTSO-E, Spark/Cassandra and MongoDB |
| `notebooks/project_notebook_part3.ipynb` | Part 3: sampling, weather, DCT/SPC, LOF, STL and spectrogram |
| `notebooks/project_notebook_part4.ipynb` | Part 4: time alignment, correlation, SARIMAX backtesting and River forecast intervals |
| `data/reservoir_areas.geojson` | WGS84 national, price-area and approximate water-area boundaries |
| `ind320/data/` | API downloads, spatial sampling and preparation |
| `ind320/analysis/` | Shared weather and reservoir calculations/figures |
| `ind320/ui/` | Shared area selection and chart controls |
| `ind320/storage/` | Local Spark/Cassandra pipeline and MongoDB publication/queries |
| `data/weather_coordinates.json` | All 8,000 reproducible sampled locations; no raw CSV is committed |
| `Budomraader/`, `NVE_Region/` | Supplied shapefiles used to generate the sample |
| `tests/` | Offline regression checks with simulated API responses |
| `docs/screencast-checklist.md` | Follow-up video demonstration, including spectrogram settings |
| `output/pdf/` | Executed notebook exports, including Part 4 |

## Notebooks

```sh
uv run jupyter lab notebooks/
```

The setup cell locates the repository, so imports work from either the root or
`notebooks/`. Part 3 recreates the committed JSON using seed 320 and 1,000 points
per area. All sampled locations contribute to the weather average: nearest ERA5
cells are weighted by the number of sampled points in each cell. Requests use
`cell_selection=nearest` and `elevation=nan`, with the 0.25° ERA5 model. Spatial
averages retain the limitations of that grid and the approximate grouped NVE
boundaries. East/West/North map to VASS 1/2/3 in the shared selector. National
reservoir data uses its own NVE series. Weather overlays are available in the
price-area and water-area views.

Run **all cells with live sources** before exporting an updated submission PDF.
For example, after selecting the project's Python kernel:

```sh
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/project_notebook_part4.ipynb
uv run jupyter nbconvert --to webpdf --allow-chromium-download --output-dir output/pdf notebooks/project_notebook_part4.ipynb
```

The optional `webpdf` export needs the nbconvert WebPDF extra and its browser:
`uv pip install 'nbconvert[webpdf]'`. Exporting through Jupyter's browser print
view is another option. Part 1's log is historical; current changes and AI use
are described in the Part 4 notebook.

## Transfers and private settings

The Transfers section and the optional net-import forecast input use the
existing MongoDB publication. Create
local `.env` settings or Streamlit Cloud secrets using the following names:

| Setting | Purpose |
| --- | --- |
| `MONGODB_URI` | Private MongoDB connection URI |
| `MONGODB_DATABASE` | Database name; default `ind320` |
| `ENTSOE_API_TOKEN` | Required only when ingesting fresh ENTSO-E data in Part 2 |
| `CASSANDRA_HOST`, `CASSANDRA_PORT` | Local Cassandra endpoint; defaults `127.0.0.1`, `9042` |
| `CASSANDRA_KEYSPACE` | Default `ind320` |
| `CASSANDRA_USERNAME`, `CASSANDRA_PASSWORD` | Optional local Cassandra authentication |
| `JAVA_HOME` | Optional Java installation override for Spark |

Running the complete Part 2 pipeline also needs a running Cassandra service and
Java compatible with Spark 3.5 (for example Java 17). It downloads the Spark
Cassandra connector on first use. The notebook ingests ENTSO-E, writes and reads
Cassandra through Spark, then publishes to MongoDB. The app reads the published MongoDB collections during queries. Never commit credentials, `.env`,
Streamlit secrets or local caches.

## Verify changes

```sh
MPLBACKEND=Agg uv run python -m unittest discover -s tests -v
```

Tests check sampled-location weights, complete-hourly-data rejection, caching,
original-scale SPC plots, and shared area selection across Streamlit pages.
Simulated API responses make these checks deterministic; they do not verify
live API availability or replace running the final notebook.
