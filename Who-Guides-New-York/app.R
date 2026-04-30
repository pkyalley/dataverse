# =============================================================================
# New York State Licensed Guides — Interactive Dashboard
# Author: Prince Peter Yalley
# =============================================================================

library(shiny)
library(shinydashboard)
library(dplyr)
library(ggplot2)
library(readr)
library(lubridate)
library(stringr)
library(scales)
library(forcats)
library(DT)
library(plotly)

# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
raw <- read_csv("data/raw/Guides_Currently_Licensed_in_New_York_State_20250127.csv",
                show_col_types = FALSE)

guides <- raw %>%
  filter(!is.na(County), str_trim(County) != "") %>%
  mutate(
    County      = str_to_title(str_trim(County)),
    City        = str_to_title(str_trim(City)),
    Activity    = str_trim(`Activity Type Description`),
    Exp_Date    = mdy(`Expiration Date`),
    Exp_Year    = year(Exp_Date),
    NY_Resident = ifelse(State == "NY", "New York", "Out-of-State"),
    `Last Name`  = str_to_title(str_trim(`Last Name`)),
    `First Name` = str_to_title(str_trim(`First Name`))
  )

all_counties   <- sort(unique(guides$County))
all_activities <- sort(unique(guides$Activity))
all_years      <- sort(unique(na.omit(guides$Exp_Year)))

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
ACT_COLORS <- c(
  "Boats and Canoes" = "#1a7abf",
  "Fishing"          = "#2eaa62",
  "Hiking"           = "#e87722",
  "Camping"          = "#9b59b6",
  "Hunting"          = "#c0392b",
  "WW Rafting"       = "#16a085",
  "Tier I Rock"      = "#7f8c8d",
  "Tier I Ice"       = "#5dade2",
  "Tier II Rock"     = "#4d5656",
  "Tier II Ice"      = "#85c1e9",
  "WW Kayaking"      = "#48c9b0",
  "WW Canoeing"      = "#58d68d"
)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
ui <- dashboardPage(
  skin = "blue",

  # ── Header ──────────────────────────────────
  dashboardHeader(
    title = tags$span(
      tags$img(src = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/New_York_State_seal.svg/120px-New_York_State_seal.svg.png",
               height = "30px", style = "margin-right:8px;"),
      "NY Licensed Guides"
    ),
    titleWidth = 280
  ),

  # ── Sidebar ─────────────────────────────────
  dashboardSidebar(
    width = 280,
    tags$div(
      style = "padding: 12px 16px 4px; color:#adb5bd; font-size:11px; text-transform:uppercase; letter-spacing:.08em;",
      "Filters"
    ),

    selectizeInput("sel_county", "County",
                   choices  = c("All Counties" = "", all_counties),
                   selected = "",
                   multiple = FALSE,
                   options  = list(placeholder = "All Counties")),

    checkboxGroupInput("sel_activity", "Activity Type",
                       choices  = all_activities,
                       selected = all_activities),

    sliderInput("sel_year", "Expiration Year Range",
                min   = min(all_years),
                max   = max(all_years),
                value = c(min(all_years), max(all_years)),
                step  = 1,
                sep   = ""),

    radioButtons("sel_residency", "Guide Residency",
                 choices  = c("All", "New York", "Out-of-State"),
                 selected = "All"),

    tags$hr(style = "border-color:#444;"),

    sidebarMenu(
      menuItem("Overview",         tabName = "overview",   icon = icon("chart-bar")),
      menuItem("Geographic View",  tabName = "geo",        icon = icon("map-marker-alt")),
      menuItem("Activity Analysis",tabName = "activity",   icon = icon("hiking")),
      menuItem("Expiration Watch", tabName = "expiration", icon = icon("calendar-alt")),
      menuItem("Data Explorer",    tabName = "explorer",   icon = icon("table"))
    ),

    tags$div(
      style = "padding: 14px 16px; color:#888; font-size:11px; margin-top:20px;",
      "Data: NY Open Data",
      tags$br(),
      "Last updated: Jan 27, 2025"
    )
  ),

  # ── Body ─────────────────────────────────────
  dashboardBody(
    tags$head(
      tags$style(HTML("
        body { font-family: 'Segoe UI', Arial, sans-serif; }
        .content-wrapper { background-color: #f5f7fa; }
        .box { border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.07); }
        .small-box { border-radius: 8px; }
        .small-box:hover { transform: translateY(-2px); transition:.2s; }
        .info-text { font-size:13px; color:#555; line-height:1.6; }
        .section-header { font-size:17px; font-weight:600; color:#1a3a5c; margin-bottom:4px; }
      "))
    ),

    tabItems(

      # ══════════════════════════
      # TAB 1 — OVERVIEW
      # ══════════════════════════
      tabItem(tabName = "overview",

        fluidRow(
          valueBoxOutput("vb_records",   width = 3),
          valueBoxOutput("vb_guides",    width = 3),
          valueBoxOutput("vb_counties",  width = 3),
          valueBoxOutput("vb_outstate",  width = 3)
        ),

        fluidRow(
          box(width = 7, title = "Top 15 Counties by License Records",
              status = "primary", solidHeader = FALSE,
              plotlyOutput("plot_county_bar", height = "400px")),

          box(width = 5, title = "Activity Type Breakdown",
              status = "primary", solidHeader = FALSE,
              plotlyOutput("plot_activity_pie", height = "400px"))
        ),

        fluidRow(
          box(width = 12, title = "About This Dashboard",
              status = "info", solidHeader = FALSE,
              tags$div(class = "info-text",
                tags$p("New York State requires professional outdoor guides to be licensed through the Department of Environmental Conservation (DEC). This dashboard explores the current roster of licensed guides — ",
                       tags$strong("who they are, where they're based, what they teach, and when their licenses expire.")),
                tags$p("Use the sidebar filters to drill into specific counties, activity types, expiration years, or residency status. Each tab focuses on a different angle of the data.")
              ))
        )
      ),

      # ══════════════════════════
      # TAB 2 — GEOGRAPHIC
      # ══════════════════════════
      tabItem(tabName = "geo",

        fluidRow(
          box(width = 8, title = "Counties Ranked by Licensed Guides",
              status = "primary",
              plotlyOutput("plot_county_full", height = "550px")),

          box(width = 4, title = "Residency Breakdown",
              status = "warning",
              plotlyOutput("plot_residency", height = "260px"),
              tags$hr(),
              tags$div(class = "section-header", "Top Out-of-State States"),
              plotlyOutput("plot_outstate", height = "220px"))
        )
      ),

      # ══════════════════════════
      # TAB 3 — ACTIVITY
      # ══════════════════════════
      tabItem(tabName = "activity",

        fluidRow(
          box(width = 6, title = "Certifications by Activity Type",
              status = "success",
              plotlyOutput("plot_activity_bar", height = "380px")),

          box(width = 6, title = "Activity Mix — Top 10 Counties (Heatmap)",
              status = "success",
              plotlyOutput("plot_heatmap", height = "380px"))
        ),

        fluidRow(
          box(width = 6, title = "Certifications Per Guide",
              status = "warning",
              plotlyOutput("plot_multicert", height = "300px")),

          box(width = 6, title = "Activity Share by Residency",
              status = "warning",
              plotlyOutput("plot_act_residency", height = "300px"))
        )
      ),

      # ══════════════════════════
      # TAB 4 — EXPIRATION
      # ══════════════════════════
      tabItem(tabName = "expiration",

        fluidRow(
          box(width = 8, title = "License Records Expiring by Year",
              status = "danger",
              plotlyOutput("plot_exp_year", height = "360px")),

          box(width = 4, title = "Expiration Context",
              status = "info",
              tags$div(class = "info-text",
                tags$p(tags$strong("Why this matters:")),
                tags$p("Guide licenses are issued on a multi-year rolling basis. The expiration distribution tells the DEC and guiding businesses when to expect renewal surges and workforce gaps."),
                tags$p("The 2028–2029 cohort is the largest — nearly half of all current licenses expire in that two-year window, creating a significant administrative renewal load.")
              ))
        ),

        fluidRow(
          box(width = 12, title = "Expiration Breakdown by Activity Type & Year",
              status = "danger",
              plotlyOutput("plot_exp_activity", height = "380px"))
        )
      ),

      # ══════════════════════════
      # TAB 5 — DATA EXPLORER
      # ══════════════════════════
      tabItem(tabName = "explorer",
        fluidRow(
          box(width = 12, title = "Browse & Search the Guide Registry",
              status = "primary",
              tags$p(class = "info-text",
                     "Use column headers to sort. Use the search box to find a specific guide, county, or business. All sidebar filters apply."),
              DTOutput("table_main"))
        )
      )

    ) # end tabItems
  ) # end dashboardBody
) # end dashboardPage


# ─────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────
server <- function(input, output, session) {

  # ── Reactive filtered data ──────────────────
  filtered <- reactive({
    d <- guides %>%
      filter(Activity %in% input$sel_activity)

    if (!is.null(input$sel_county) && input$sel_county != "")
      d <- d %>% filter(County == input$sel_county)

    if (input$sel_residency != "All")
      d <- d %>% filter(NY_Resident == input$sel_residency)

    d <- d %>%
      filter(!is.na(Exp_Year),
             Exp_Year >= input$sel_year[1],
             Exp_Year <= input$sel_year[2])
    d
  })

  # ── VALUE BOXES ─────────────────────────────
  output$vb_records <- renderValueBox({
    valueBox(format(nrow(filtered()), big.mark=","), "License Records",
             icon = icon("id-card"), color = "blue")
  })
  output$vb_guides <- renderValueBox({
    valueBox(format(n_distinct(filtered()$`Badge Number`), big.mark=","),
             "Individual Guides", icon = icon("users"), color = "green")
  })
  output$vb_counties <- renderValueBox({
    valueBox(n_distinct(filtered()$County), "Counties",
             icon = icon("map"), color = "yellow")
  })
  output$vb_outstate <- renderValueBox({
    n   <- sum(filtered()$State != "NY")
    pct <- ifelse(nrow(filtered()) > 0, round(100*n/nrow(filtered()),1), 0)
    valueBox(paste0(pct, "%"), "Out-of-State Guides",
             icon = icon("globe-americas"), color = "purple")
  })

  # ── OVERVIEW: County bar ────────────────────
  output$plot_county_bar <- renderPlotly({
    d <- filtered() %>%
      count(County, sort = TRUE) %>%
      slice_head(n = 15) %>%
      mutate(County = fct_reorder(County, n))

    p <- ggplot(d, aes(x = County, y = n, text = paste(County, ":", n, "records"))) +
      geom_col(fill = "#2c7bb6", width = 0.72) +
      coord_flip() +
      scale_y_continuous(labels = comma) +
      labs(x = NULL, y = "License Records") +
      theme_minimal(base_size = 12) +
      theme(panel.grid.major.y = element_blank())

    ggplotly(p, tooltip = "text") %>%
      layout(margin = list(l = 10))
  })

  # ── OVERVIEW: Activity pie ──────────────────
  output$plot_activity_pie <- renderPlotly({
    d <- filtered() %>% count(Activity, sort = TRUE)
    plot_ly(d, labels = ~Activity, values = ~n, type = "pie",
            marker = list(colors = unname(ACT_COLORS[d$Activity])),
            textinfo = "label+percent",
            hoverinfo = "label+value") %>%
      layout(showlegend = FALSE,
             margin = list(t = 10, b = 10))
  })

  # ── GEO: Full county bar ────────────────────
  output$plot_county_full <- renderPlotly({
    d <- filtered() %>%
      count(County, sort = TRUE) %>%
      mutate(County = fct_reorder(County, n))

    p <- ggplot(d, aes(x = County, y = n,
                       text = paste(County, ":", n, "records"))) +
      geom_col(fill = "#2c7bb6", width = 0.72) +
      coord_flip() +
      scale_y_continuous(labels = comma) +
      labs(x = NULL, y = "License Records") +
      theme_minimal(base_size = 11) +
      theme(panel.grid.major.y = element_blank(),
            axis.text.y = element_text(size = 9))

    ggplotly(p, tooltip = "text", height = 550) %>%
      layout(margin = list(l = 10))
  })

  # ── GEO: Residency donut ────────────────────
  output$plot_residency <- renderPlotly({
    d <- filtered() %>% count(NY_Resident)
    plot_ly(d, labels = ~NY_Resident, values = ~n, type = "pie",
            hole = 0.5,
            marker = list(colors = c("#2eaa62", "#e87722")),
            textinfo = "label+percent") %>%
      layout(showlegend = FALSE, margin = list(t=5,b=5))
  })

  # ── GEO: Out-of-state bar ───────────────────
  output$plot_outstate <- renderPlotly({
    d <- filtered() %>%
      filter(State != "NY") %>%
      count(State, sort = TRUE) %>%
      slice_head(n = 8) %>%
      mutate(State = fct_reorder(State, n))

    p <- ggplot(d, aes(x = State, y = n,
                       text = paste(State, ":", n))) +
      geom_col(fill = "#27ae60", width = 0.68) +
      coord_flip() +
      labs(x = NULL, y = NULL) +
      theme_minimal(base_size = 11) +
      theme(panel.grid.major.y = element_blank())

    ggplotly(p, tooltip = "text") %>%
      layout(margin = list(l=5, b=5))
  })

  # ── ACTIVITY: Bar ───────────────────────────
  output$plot_activity_bar <- renderPlotly({
    d <- filtered() %>%
      count(Activity, sort = TRUE) %>%
      mutate(Activity = fct_reorder(Activity, n))

    colors_vec <- unname(ACT_COLORS[as.character(d$Activity)])

    p <- ggplot(d, aes(x = Activity, y = n, fill = Activity,
                       text = paste(Activity, ":", n))) +
      geom_col(width = 0.72, show.legend = FALSE) +
      coord_flip() +
      scale_fill_manual(values = setNames(ACT_COLORS, names(ACT_COLORS))) +
      scale_y_continuous(labels = comma) +
      labs(x = NULL, y = "License Records") +
      theme_minimal(base_size = 12) +
      theme(panel.grid.major.y = element_blank())

    ggplotly(p, tooltip = "text")
  })

  # ── ACTIVITY: Heatmap ───────────────────────
  output$plot_heatmap <- renderPlotly({
    top10 <- filtered() %>%
      count(County, sort = TRUE) %>%
      slice_head(n = 10) %>%
      pull(County)

    d <- filtered() %>%
      filter(County %in% top10) %>%
      group_by(County, Activity) %>%
      summarise(Count = n(), .groups = "drop") %>%
      group_by(County) %>%
      mutate(Pct = round(Count / sum(Count) * 100, 1)) %>%
      ungroup()

    plot_ly(d, x = ~Activity, y = ~County, z = ~Pct,
            type = "heatmap",
            colorscale = list(c(0,"#d6eaf8"), c(1,"#1a5276")),
            hovertemplate = "%{y} — %{x}<br>%{z}%<extra></extra>") %>%
      layout(xaxis = list(tickangle = -35),
             margin = list(b = 100))
  })

  # ── ACTIVITY: Multi-cert ────────────────────
  output$plot_multicert <- renderPlotly({
    d <- filtered() %>%
      count(`Badge Number`) %>%
      count(n, name = "Num_Guides") %>%
      rename(Num_Certs = n)

    p <- ggplot(d, aes(x = factor(Num_Certs), y = Num_Guides,
                       text = paste(Num_Certs, "cert(s):", Num_Guides, "guides"))) +
      geom_col(fill = "#8e44ad", width = 0.65) +
      labs(x = "Number of Certifications", y = "Number of Guides") +
      theme_minimal(base_size = 12) +
      theme(panel.grid.major.x = element_blank())

    ggplotly(p, tooltip = "text")
  })

  # ── ACTIVITY: By residency ──────────────────
  output$plot_act_residency <- renderPlotly({
    d <- filtered() %>%
      group_by(Activity, NY_Resident) %>%
      summarise(Count = n(), .groups = "drop")

    p <- ggplot(d, aes(x = fct_reorder(Activity, Count, sum),
                       y = Count, fill = NY_Resident,
                       text = paste(Activity, "/", NY_Resident, ":", Count))) +
      geom_col(position = "stack", width = 0.72) +
      coord_flip() +
      scale_fill_manual(values = c("New York" = "#2c7bb6", "Out-of-State" = "#e87722")) +
      scale_y_continuous(labels = comma) +
      labs(x = NULL, y = "Count", fill = NULL) +
      theme_minimal(base_size = 11) +
      theme(panel.grid.major.y = element_blank(),
            legend.position = "top")

    ggplotly(p, tooltip = "text") %>%
      layout(legend = list(orientation = "h", y = 1.05))
  })

  # ── EXPIRATION: By year ─────────────────────
  output$plot_exp_year <- renderPlotly({
    d <- filtered() %>%
      filter(!is.na(Exp_Year)) %>%
      count(Exp_Year) %>%
      mutate(label = format(n, big.mark = ","))

    p <- ggplot(d, aes(x = factor(Exp_Year), y = n,
                       text = paste("Year:", Exp_Year, "\nRecords:", format(n, big.mark=",")))) +
      geom_col(fill = "#e67e22", width = 0.65) +
      scale_y_continuous(labels = comma) +
      labs(x = "Expiration Year", y = "License Records") +
      theme_minimal(base_size = 13) +
      theme(panel.grid.major.x = element_blank())

    ggplotly(p, tooltip = "text")
  })

  # ── EXPIRATION: By activity & year ──────────
  output$plot_exp_activity <- renderPlotly({
    d <- filtered() %>%
      filter(!is.na(Exp_Year)) %>%
      group_by(Exp_Year, Activity) %>%
      summarise(Count = n(), .groups = "drop")

    p <- ggplot(d, aes(x = factor(Exp_Year), y = Count,
                       fill = Activity,
                       text = paste(Activity, "\nYear:", Exp_Year, "\nCount:", Count))) +
      geom_col(position = "stack", width = 0.72) +
      scale_fill_manual(values = ACT_COLORS, na.value = "#cccccc") +
      scale_y_continuous(labels = comma) +
      labs(x = "Expiration Year", y = "License Records", fill = NULL) +
      theme_minimal(base_size = 12) +
      theme(panel.grid.major.x = element_blank(),
            legend.position = "bottom")

    ggplotly(p, tooltip = "text") %>%
      layout(legend = list(orientation = "h"))
  })

  # ── DATA TABLE ──────────────────────────────
  output$table_main <- renderDT({
    filtered() %>%
      select(
        `Last Name`, `First Name`, County, City, State,
        Activity, `Expiration Date`, `Badge Number`,
        `Business Name`, `Phone `
      ) %>%
      arrange(County, `Last Name`) %>%
      datatable(
        rownames = FALSE,
        filter   = "top",
        options  = list(
          pageLength = 20,
          scrollX    = TRUE,
          dom        = "lfrtip"
        ),
        class = "compact stripe"
      )
  })

} # end server


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
shinyApp(ui = ui, server = server)
