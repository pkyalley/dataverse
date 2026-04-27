# =========================
# LIBRARIES
# =========================
library(shiny)
library(tidyverse)
library(plotly)
library(forecast)
library(DT)

# =========================
# LOAD DATA
# =========================
data <- read_csv("State_Park_Annual_Attendance_Figures_by_Facility___Beginning_2003_20250211.csv")

# Clean column names
colnames(data) <- make.names(colnames(data))

# Remove missing values
data <- na.omit(data)

# Ensure Year is numeric
data$Year <- as.numeric(data$Year)

# =========================
# AGGREGATED DATA
# =========================

# Yearly attendance
yearly_attendance <- data %>%
  group_by(Year) %>%
  summarise(Total = sum(Attendance)) %>%
  arrange(Year)

# Facility ranking (FIXED: Facility NOT Facility.Name)
facility_data <- data %>%
  group_by(Facility) %>%
  summarise(Total = sum(Attendance)) %>%
  arrange(desc(Total))

# =========================
# UI
# =========================
ui <- fluidPage(
  
  titlePanel("State Park Analytics Dashboard (EB-2 NIW Project)"),
  
  sidebarLayout(
    sidebarPanel(
      
      sliderInput(
        "yearRange",
        "Select Year Range:",
        min = min(data$Year),
        max = max(data$Year),
        value = c(min(data$Year), max(data$Year))
      ),
      
      selectInput(
        "facility",
        "Select Facility:",
        choices = unique(data$Facility),
        selected = unique(data$Facility)[1]
      ),
      
      numericInput(
        "forecast_horizon",
        "Forecast Years:",
        value = 5,
        min = 1,
        max = 20
      )
    ),
    
    mainPanel(
      tabsetPanel(
        
        # =========================
        # OVERVIEW TAB
        # =========================
        tabPanel("Overview",
                 
                 plotlyOutput("trendPlot"),
                 plotlyOutput("growthPlot")
        ),
        
        # =========================
        # FACILITY TAB
        # =========================
        tabPanel("Facility Analysis",
                 
                 plotlyOutput("facilityPlot"),
                 dataTableOutput("facilityTable")
        ),
        
        # =========================
        # FORECAST TAB
        # =========================
        tabPanel("Forecasting",
                 
                 plotlyOutput("forecastPlot")
        ),
        
        # =========================
        # INSIGHTS TAB
        # =========================
        tabPanel("Insights",
                 
                 verbatimTextOutput("insightsText")
        )
      )
    )
  )
)

# =========================
# SERVER
# =========================
server <- function(input, output) {
  
  # -------------------------
  # FILTERED DATA
  # -------------------------
  filtered_data <- reactive({
    
    df <- data %>%
      filter(
        Year >= input$yearRange[1],
        Year <= input$yearRange[2]
      )
    
    if (!is.null(input$facility)) {
      df <- df %>% filter(Facility == input$facility)
    }
    
    df
  })
  
  # -------------------------
  # TREND PLOT
  # -------------------------
  output$trendPlot <- renderPlotly({
    
    df <- filtered_data() %>%
      group_by(Year) %>%
      summarise(Total = sum(Attendance))
    
    p <- ggplot(df, aes(x = Year, y = Total)) +
      geom_line() +
      geom_point() +
      ggtitle("Attendance Trend")
    
    ggplotly(p)
  })
  
  # -------------------------
  # GROWTH PLOT
  # -------------------------
  output$growthPlot <- renderPlotly({
    
    df <- filtered_data() %>%
      group_by(Year) %>%
      summarise(Total = sum(Attendance)) %>%
      arrange(Year) %>%
      mutate(Growth = (Total - lag(Total)) / lag(Total))
    
    p <- ggplot(df, aes(x = Year, y = Growth)) +
      geom_line() +
      geom_point() +
      ggtitle("Year-over-Year Growth")
    
    ggplotly(p)
  })
  
  # -------------------------
  # TOP FACILITIES PLOT
  # -------------------------
  output$facilityPlot <- renderPlotly({
    
    df <- facility_data %>% head(10)
    
    p <- ggplot(df, aes(x = reorder(Facility, Total), y = Total)) +
      geom_col() +
      coord_flip() +
      ggtitle("Top 10 Facilities")
    
    ggplotly(p)
  })
  
  # -------------------------
  # FACILITY TABLE
  # -------------------------
  output$facilityTable <- renderDataTable({
    datatable(facility_data)
  })
  
  # -------------------------
  # FORECAST
  # -------------------------
  output$forecastPlot <- renderPlotly({
    
    ts_data <- ts(
      yearly_attendance$Total,
      start = min(yearly_attendance$Year)
    )
    
    model <- auto.arima(ts_data)
    forecasted <- forecast(model, h = input$forecast_horizon)
    
    p <- autoplot(forecasted)
    
    ggplotly(p)
  })
  
  # -------------------------
  # INSIGHTS
  # -------------------------
  output$insightsText <- renderText({
    
    paste(
      "Key Insights:",
      "\n- Attendance shows long-term growth trends.",
      "\n- A small number of facilities account for most visitors.",
      "\n- Forecasting supports planning and resource allocation.",
      "\n\nPolicy Impact:",
      "\nThis dashboard supports data-driven infrastructure planning",
      "and tourism management decisions.",
      "\n\nEB-2 NIW Relevance:",
      "\nDemonstrates advanced data analytics applied to public sector optimization."
    )
  })
}

# =========================
# RUN APP
# =========================
shinyApp(ui = ui, server = server)