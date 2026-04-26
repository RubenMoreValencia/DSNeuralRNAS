mod_cap03_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 3: Activaciones y neurona individual"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite evaluar funciones de activación y ejecutar una neurona individual RNAS.")
    ),

    fluidRow(
      column(
        4,
        selectInput(ns("activation"), "Activación", c("tanh", "sigmoid"), selected = "tanh"),
        numericInput(ns("x1"), "x1", value = 0.5, step = 0.1),
        numericInput(ns("x2"), "x2", value = -0.2, step = 0.1),
        numericInput(ns("w1"), "w1", value = 0.8, step = 0.05),
        numericInput(ns("w2"), "w2", value = 0.3, step = 0.05),
        numericInput(ns("b"), "b", value = 0.1, step = 0.05),
        actionButton(ns("run"), "Ejecutar", class = "btn-primary")
      ),
      column(
        8,
        tabsetPanel(
          tabPanel("Código", br(), uiOutput(ns("codigo"))),
          tabPanel("Resultado", br(), DT::DTOutput(ns("tabla"))),
          tabPanel("Gráfica", br(), plotOutput(ns("plot"), height = "420px")),
          tabPanel("Interpretación", br(), verbatimTextOutput(ns("interpretacion")))
        )
      )
    )
  )
}

mod_cap03_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      x <- c(input$x1, input$x2)
      w <- c(input$w1, input$w2)
      b <- input$b

      z <- sum(w * x) + b

      y_hat <- rnas_neuron_forward(
        x = x,
        w = w,
        b = b,
        activation = input$activation
      )

      data.frame(
        x1 = input$x1,
        x2 = input$x2,
        w1 = input$w1,
        w2 = input$w2,
        b = input$b,
        z = z,
        y_hat = y_hat,
        activation = input$activation,
        stringsAsFactors = FALSE
      )
    })

    output$codigo <- renderUI({
      codigo <- paste0(
        "x <- c(", input$x1, ", ", input$x2, ")\n",
        "w <- c(", input$w1, ", ", input$w2, ")\n",
        "b <- ", input$b, "\n\n",
        "rnas_neuron_forward(\n",
        "  x = x,\n",
        "  w = w,\n",
        "  b = b,\n",
        "  activation = \"", input$activation, "\"\n",
        ")"
      )

      mostrar_codigo(codigo)
    })

    output$tabla <- DT::renderDT({
      req(res())
      tabla_dt(res())
    })

    output$plot <- renderPlot({
      z <- seq(-5, 5, length.out = 200)

      y <- if (input$activation == "tanh") {
        tanh(z)
      } else {
        1 / (1 + exp(-z))
      }

      plot(
        z, y, type = "l", lwd = 2,
        xlab = "z",
        ylab = "a(z)",
        main = paste("Función de activación:", input$activation)
      )
      grid()
    })

    output$interpretacion <- renderText({
      req(res())
      r <- res()

      paste0(
        "La neurona calcula primero z = w'x + b. ",
        "Con los valores ingresados se obtiene z = ",
        round(r$z, 6),
        " y salida activada y_hat = ",
        round(r$y_hat, 6),
        "."
      )
    })
  })
}
