# 🏔️ New York State Licensed Guides — Data Analysis Project

> An end-to-end R data analysis project exploring the geography, activity patterns, and workforce dynamics of New York State's licensed outdoor guide community.

---

## 📖 What Is This Project?

New York State requires professional outdoor guides to be licensed through the Department of Environmental Conservation (DEC). These are the people who lead clients through the Adirondacks, the Catskills, and the state's wild waterways — for fishing, hiking, hunting, camping, whitewater rafting, and technical climbing.

This project takes a publicly available dataset of **6,276 license records** (representing **2,516 individual guides**) and transforms it from a raw government CSV into two polished, interactive data products:

1. **A professional HTML report** — a narrative-driven analysis with publication-quality charts, written in R Markdown.
2. **An interactive Shiny dashboard** — a full multi-tab web application with real-time filters, interactive plots (via Plotly), and a searchable data table.

The goal is to answer meaningful questions about the state of outdoor guiding in New York:

- Where are guides concentrated — and why?
- What activities dominate the licensing landscape?
- How many guides come from outside New York State?
- When do current licenses expire, and what does that mean for the industry?
- How professionally diverse are individual guides?

---

## 📁 Project Structure

```
ny-licensed-guides-analysis/
│
├── app.R                    # Shiny dashboard application
├── data/
│   └── raw/
│       └── Guides_Currently_Licensed_in_New_York_State_20250127.csv
├── reports/
│   ├── Who-Guides-New-York.Rmd    # Narrative HTML report (R Markdown)
│   ├── Who-Guides-New-York.html   # Rendered HTML report snapshot
│   └── report_style.css           # Custom CSS for the HTML report
├── requirements.R
└── README.md
```

---

## 📊 The Data

**Source:** [NY Open Data — Currently Licensed Guides in New York State](https://data.ny.gov/Recreation/Guides-Currently-Licensed-in-New-York-State/msy3-bnhe)

**Last Updated:** January 27, 2025

**Records:** 6,276 license records | 2,516 unique guides (by badge number)

### Key Fields

| Column | Description |
|--------|-------------|
| `Last Name` / `First Name` | Guide's name |
| `County` | County of operation (NY county) |
| `City` / `State` / `Zip` | Guide's home address |
| `Activity Type Description` | Type of guiding certification |
| `Expiration Date` | License expiration date |
| `Badge Number` | Unique guide identifier (guides can have multiple records) |
| `Business Name` | Guide's business name (if applicable) |
| `Georeference` | Lat/lon point for the guide's location |

### Activity Types

The dataset covers 12 distinct activity certifications:

- **Boats and Canoes** (1,552 records) — most common
- **Fishing** (1,326)
- **Hiking** (1,022)
- **Camping** (881)
- **Hunting** (777)
- **WW Rafting** (370)
- Tier I/II Rock & Ice Climbing, WW Kayaking, WW Canoeing

---

## 🔍 Key Findings

1. **The Adirondacks dominate.** Essex, Warren, and Franklin counties — all Adirondack counties — are the top three by guide count. The Adirondack Park's 6-million-acre footprint makes it the undisputed center of NY's guiding industry.

2. **Water activities lead certifications.** Boats & Canoes and Fishing together account for nearly half of all license records, reflecting New York's exceptional network of lakes, rivers, and coastal waterways.

3. **1 in 6 licensed guides lives outside New York.** PA, NJ, VT, and CT contribute the most out-of-state licensees — guides who work in NY's border wilderness regions.

4. **A major renewal wave is approaching.** Over 2,900 license records (nearly 47%) expire in 2028–2029, creating a concentrated administrative burden for the DEC.

5. **Most guides hold multiple certifications.** Over 75% of individual guides (by badge number) hold 2 or more activity certifications — a marker of professional versatility and experience.

---

## 🛠️ How to Run This Project Locally

### Prerequisites

Install R (≥ 4.2) and RStudio, then install the required packages:

```r
source("requirements.R")
```

### 1. Clone the Repository

```bash
git clone https://github.com/pkyalley/dataverse.git
cd dataverse/ny-licensed-guides-analysis
```

### 2. Run the HTML Report

Open `reports/Who-Guides-New-York.Rmd` in RStudio and click **Knit**, or run from the console:

```r
rmarkdown::render("reports/Who-Guides-New-York.Rmd")
```

This will generate `reports/Who-Guides-New-York.html` — open it in any browser.

### 3. Launch the Shiny Dashboard

Open `app.R` in RStudio and click **Run App**, or from the console:

```r
shiny::runApp(".")
```

The dashboard will open at `http://127.0.0.1:XXXX` in your browser.

> **Note:** Both the report and the app expect the CSV to be in `data/raw/` relative to the project root.

---

## 📸 Screenshots

*(Add screenshots here after running the apps locally.)*

| View | Description |
|------|-------------|
| `screenshot_report.png` | HTML report — overview section |
| `screenshot_dashboard_overview.png` | Shiny dashboard — Overview tab |
| `screenshot_dashboard_geo.png` | Shiny dashboard — Geographic View tab |
| `screenshot_dashboard_activity.png` | Shiny dashboard — Activity Analysis tab |
| `screenshot_dashboard_expiration.png` | Shiny dashboard — Expiration Watch tab |
| `screenshot_dashboard_explorer.png` | Shiny dashboard — Data Explorer tab |

---

## 💡 Why This Project

This started as a homework assignment on data visualization in R. The original work used basic `ggplot2` charts to explore the same dataset. For this project, I rebuilt it from the ground up as a real analytical product — with a clear narrative purpose, professional design, interactive visualizations, and a full Shiny application.

The dataset is genuinely interesting: it captures a regulated professional community that most people don't think about but that plays a real role in New York's outdoor recreation economy and conservation ecosystem.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **R** | Core language |
| **R Markdown** | Literate programming for the HTML report |
| **ggplot2** | Static visualizations |
| **plotly** | Interactive charts in the dashboard |
| **Shiny + shinydashboard** | Web application framework |
| **dplyr / tidyr** | Data manipulation |
| **lubridate** | Date parsing |
| **DT** | Interactive data tables |
| **kableExtra** | Report tables |

---

## 📜 Data License

This dataset is published by the New York State Department of Environmental Conservation under the [NY Open Data Terms of Use](https://data.ny.gov/download/77gx-ii52/application%2Fpdf). It is freely available for public use.

---

## 👤 Author

**Prince Peter Yalley**  
Data analysis, visualization, and Shiny development.

---

*Built with R, ggplot2, and Shiny. Data courtesy of NY Open Data.*
