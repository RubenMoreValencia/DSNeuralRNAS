# ============================================================
# DSNeuralRNASLab - app.R
# Laboratorio interactivo por capítulos
# ============================================================

ui <- fluidPage(
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "estilos.css")
  ),

  titlePanel("DSNeuralRNASLab: laboratorio interactivo del libro"),

  sidebarLayout(
    sidebarPanel(
      width = 3,

      h4("Módulos del libro"),

      selectInput(
        inputId = "modulo",
        label = "Seleccione capítulo",
        choices = c(
          "Inicio" = "inicio",
          "Cap. 3 Activaciones y neurona" = "cap03",
          "Cap. 4 Pérdida y gradiente" = "cap04",
          "Cap. 5 Entrenamiento de neurona" = "cap05",
          "Cap. 6 MLP simple" = "cap06",
          "Cap. 7 Dinámica continua" = "cap07",
          "Cap. 8 Geometría del aprendizaje" = "cap08",
          "Cap. 9 Regímenes dinámicos" = "cap09",
          "Cap. 10 Control de tasa" = "cap10",
          "Cap. 11 Control óptimo y meta-aprendizaje" = "cap11",
          "Cap. 12 Integración STD2, SimuDS y FNL" = "cap12",
          "Cap. 13 Casos aplicados" = "cap13"
        ),
        selected = "inicio"
      ),

      hr(),

      div(
        class = "nota-lateral",
        strong("DSNeuralRNASLab"),
        br(),
        "Laboratorio académico asociado al libro DS Neural RNAS.",
        br(), br(),
        "Autor: Rubén Alexander More Valencia.",
        br(),
        "Universidad Nacional de Piura.",
        br(), br(),
        "Seleccione un capítulo para ejecutar sus funciones, revisar código de uso, tablas, gráficas e interpretación."
      )
    ),

    mainPanel(
      width = 9,

      conditionalPanel(
        condition = "input.modulo == 'inicio'",
        mod_inicio_ui("inicio")
      ),

      conditionalPanel(
        condition = "input.modulo == 'cap03'",
        mod_cap03_ui("cap03")
      ),

      conditionalPanel(
        condition = "input.modulo == 'cap04'",
        mod_cap04_ui("cap04")
      ),

      conditionalPanel(
        condition = "input.modulo == 'cap05'",
        mod_cap05_ui("cap05")
       ),
      conditionalPanel(
        condition = "input.modulo == 'cap06'",
        mod_cap06_ui("cap06")
      ),

      conditionalPanel(
        condition = "input.modulo == 'cap07'",
        mod_cap07_ui("cap07")
      ),
      conditionalPanel(
        condition = "input.modulo == 'cap08'",
        mod_cap08_ui("cap08")
      ),
      conditionalPanel(
        condition = "input.modulo == 'cap09'",
        mod_cap09_ui("cap09")
      ),
      conditionalPanel(
        condition = "input.modulo == 'cap10'",
        mod_cap10_ui("cap10")
      ),
      conditionalPanel(
        condition = "input.modulo == 'cap11'",
        mod_cap11_ui("cap11")
      ),
      conditionalPanel(
        condition = "input.modulo == 'cap12'",
        mod_cap12_ui("cap12")
      ),

      conditionalPanel(
        condition = "input.modulo == 'cap13'",
        mod_cap13_ui("cap13")

      )
    )
  )
)

server <- function(input, output, session) {
  mod_inicio_server("inicio")
  mod_cap03_server("cap03")
  mod_cap04_server("cap04")
  mod_cap05_server("cap05")
  mod_cap06_server("cap06")
  mod_cap07_server("cap07")
  mod_cap08_server("cap08")
  mod_cap09_server("cap09")
  mod_cap10_server("cap10")
  mod_cap11_server("cap11")
  mod_cap12_server("cap12")
  mod_cap13_server("cap13")
}

shinyApp(ui = ui, server = server)
