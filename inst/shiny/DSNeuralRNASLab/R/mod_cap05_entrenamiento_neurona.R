# ============================================================
# Modulo Capitulo 5: Entrenamiento discreto de neurona
# ============================================================

mod_cap05_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 5: Entrenamiento discreto de neurona"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite entrenar una neurona RNAS con datos controlados, modificando tasa de aprendizaje, iteraciones, activación, pesos iniciales y sesgo.")
    ),

    fluidRow(
      column(
        4,

        selectInput(
          ns("activation"),
          "Activación",
          choices = c("tanh", "sigmoid"),
          selected = "tanh"
        ),

        numericInput(
          ns("eta0"),
          "Eta inicial",
          value = 0.1,
          min = 0.001,
          max = 1,
          step = 0.005
        ),

        numericInput(
          ns("T_iter"),
          "Número de iteraciones",
          value = 100,
          min = 5,
          max = 1000,
          step = 5
        ),

        numericInput(ns("w1"), "w1 inicial", value = 0.8, step = 0.05),
        numericInput(ns("w2"), "w2 inicial", value = 0.3, step = 0.05),
        numericInput(ns("b"), "b inicial", value = 0.1, step = 0.05),

        actionButton(ns("run"), "Ejecutar entrenamiento", class = "btn-primary")
      ),

      column(
        8,

        tabsetPanel(
          tabPanel("Código", br(), uiOutput(ns("codigo"))),
          tabPanel("Datos", br(), DT::DTOutput(ns("tabla_datos"))),
          tabPanel("Resumen", br(), DT::DTOutput(ns("tabla_resumen"))),
          tabPanel("Trayectoria", br(), DT::DTOutput(ns("tabla_trayectoria"))),
          tabPanel("Gráficas", br(), plotOutput(ns("plot"), height = "430px")),
          tabPanel("Interpretación", br(), verbatimTextOutput(ns("interpretacion")))
        )
      )
    )
  )
}

mod_cap05_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      datos <-generar_datos_neurona_base()

      X <- datos$X
      y <- datos$y
      theta0 <- c(input$w1, input$w2, input$b)

      modelo <- DSNeuralRNAS::rnas_train_neuron_control_eta(
        X = X,
        y = y,
        theta0 = theta0,
        eta0 = input$eta0,
        policy = "constante",
        eta_min = input$eta0,
        eta_max = input$eta0,
        T = as.integer(input$T_iter),
        activation = input$activation
      )

      list(
        datos = datos,
        modelo = modelo
      )
    })

    output$codigo <- renderUI({
      codigo <- paste0(
        "datos <- generar_datos_neurona_base()\n",
        "X <- datos$X\n",
        "y <- datos$y\n",
        "theta0 <- c(", input$w1, ", ", input$w2, ", ", input$b, ")\n\n",
        "modelo <- rnas_train_neuron_control_eta(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  theta0 = theta0,\n",
        "  eta0 = ", input$eta0, ",\n",
        "  policy = \"constante\",\n",
        "  eta_min = ", input$eta0, ",\n",
        "  eta_max = ", input$eta0, ",\n",
        "  T = ", input$T_iter, ",\n",
        "  activation = \"", input$activation, "\"\n",
        ")"
      )

      mostrar_codigo(codigo)
    })

    output$tabla_datos <- DT::renderDT({
      req(res())

      datos <- res()$datos

      tab <- data.frame(
        obs = seq_len(nrow(datos$X)),
        x1 = datos$X[, 1],
        x2 = datos$X[, 2],
        y_obs = datos$y
      )

      tabla_dt(tab)
    })

    output$tabla_resumen <- DT::renderDT({
      req(res())

      m <- res()$modelo$metricas

      tab <- data.frame(
        indicador = c(
          "loss_inicial",
          "loss_final",
          "reduccion_abs",
          "reduccion_rel",
          "eta_media",
          "eta_min_obs",
          "eta_max_obs",
          "grad_norm_inicial",
          "grad_norm_final",
          "descendente_global"
        ),
        valor = c(
          m$loss_inicial,
          m$loss_final,
          m$reduccion_abs,
          m$reduccion_rel,
          m$eta_media,
          m$eta_min_obs,
          m$eta_max_obs,
          m$grad_norm_inicial,
          m$grad_norm_final,
          as.numeric(m$descendente_global)
        )
      )

      tabla_dt(tab)
    })

    output$tabla_trayectoria <- DT::renderDT({
      req(res())

      tray <- res()$modelo$trayectoria

      columnas <- intersect(
        c("iter", "loss", "grad_norm", "eta", "speed", "b", "w1", "w2"),
        names(tray)
      )

      tabla_dt(utils::head(tray[, columnas, drop = FALSE], 20))
    })

    output$plot <- renderPlot({
      req(res())

      tray <- res()$modelo$trayectoria

      op <- par(mfrow = c(1, 2))
      on.exit(par(op), add = TRUE)

      plot(
        tray$iter,
        tray$loss,
        type = "l",
        lwd = 2,
        xlab = "Iteración",
        ylab = "Pérdida MSE",
        main = "Curva de pérdida"
      )
      grid()

      plot(
        tray$iter,
        tray$grad_norm,
        type = "l",
        lwd = 2,
        xlab = "Iteración",
        ylab = "Norma del gradiente",
        main = "Norma del gradiente"
      )
      grid()
    })

    output$interpretacion <- renderText({
      req(res())

      m <- res()$modelo$metricas

      paste0(
        "El entrenamiento de la neurona RNAS inició con una pérdida de ",
        round(m$loss_inicial, 6),
        " y finalizó con una pérdida de ",
        round(m$loss_final, 6),
        ". La reducción relativa fue ",
        round(m$reduccion_rel, 4),
        ". El gradiente pasó de ",
        round(m$grad_norm_inicial, 6),
        " a ",
        round(m$grad_norm_final, 6),
        "."
      )
    })
  })
}
