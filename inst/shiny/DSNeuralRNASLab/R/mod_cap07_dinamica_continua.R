# ============================================================
# Modulo Capitulo 7: Dinamica continua del aprendizaje
# DSNeuralRNASLab
# ============================================================

mod_cap07_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 7: Aprendizaje como dinámica continua"),

    div(
      class = "bloque-explicacion",
      p("Este módulo interpreta el aprendizaje de una neurona RNAS como una dinámica continua aproximada. El usuario puede modificar eta, paso temporal, iteraciones, activación y parámetros iniciales.")
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

        numericInput(ns("eta"), "Eta", value = 0.1, min = 0.001, max = 1, step = 0.005),
        numericInput(ns("dt"), "Paso temporal dt", value = 0.1, min = 0.001, max = 1, step = 0.01),
        numericInput(ns("T_iter"), "Número de pasos", value = 100, min = 5, max = 1000, step = 5),

        numericInput(ns("w1"), "w1 inicial", value = 0.8, step = 0.05),
        numericInput(ns("w2"), "w2 inicial", value = 0.3, step = 0.05),
        numericInput(ns("b"), "b inicial", value = 0.1, step = 0.05),

        actionButton(ns("run"), "Ejecutar dinámica", class = "btn-primary")
      ),

      column(
        8,
        tabsetPanel(
          tabPanel("Código", br(), uiOutput(ns("codigo"))),
          tabPanel("Datos", br(), DT::DTOutput(ns("tabla_datos"))),
          tabPanel("Resumen", br(), DT::DTOutput(ns("tabla_resumen"))),
          tabPanel("Trayectoria", br(), DT::DTOutput(ns("tabla_trayectoria"))),
          tabPanel("Gráficas", br(), plotOutput(ns("plot"), height = "460px")),
          tabPanel("Interpretación", br(), verbatimTextOutput(ns("interpretacion")))
        )
      )
    )
  )
}

mod_cap07_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      datos <- generar_datos_neurona_base()

      X <- datos$X
      y <- datos$y
      theta0 <- c(input$w1, input$w2, input$b)

      dinamica <- rnas_integrar_dinamica_neuron(
        X = X,
        y = y,
        theta0 = theta0,
        eta = input$eta,
        dt = input$dt,
        T = as.integer(input$T_iter),
        activation = input$activation
      )

      list(
        datos = datos,
        dinamica = dinamica
      )
    })

    output$codigo <- renderUI({
      codigo <- paste0(
        "datos <- generar_datos_neurona_base()\n",
        "X <- datos$X\n",
        "y <- datos$y\n",
        "theta0 <- c(", input$w1, ", ", input$w2, ", ", input$b, ")\n\n",
        "dinamica <- rnas_integrar_dinamica_neuron(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  theta0 = theta0,\n",
        "  eta = ", input$eta, ",\n",
        "  dt = ", input$dt, ",\n",
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
        y_obs = datos$y,
        stringsAsFactors = FALSE
      )

      tabla_dt(tab)
    })

    output$tabla_resumen <- DT::renderDT({
      req(res())

      d <- res()$dinamica

      if (!is.null(d$metricas)) {
        tab <- data.frame(
          indicador = names(d$metricas),
          valor = unlist(d$metricas, use.names = FALSE),
          stringsAsFactors = FALSE
        )
      } else {
        tray <- d$trayectoria
        tab <- data.frame(
          indicador = c("loss_inicial", "loss_final", "n_pasos"),
          valor = c(
            tray$loss[1],
            tray$loss[nrow(tray)],
            nrow(tray)
          ),
          stringsAsFactors = FALSE
        )
      }

      tabla_dt(tab)
    })

    output$tabla_trayectoria <- DT::renderDT({
      req(res())

      tray <- res()$dinamica$trayectoria

      columnas <- intersect(
        c("iter", "time", "loss", "grad_norm", "speed", "w1", "w2", "b"),
        names(tray)
      )

      tabla_dt(utils::head(tray[, columnas, drop = FALSE], 25))
    })

    output$plot <- renderPlot({
      req(res())

      tray <- res()$dinamica$trayectoria

      op <- par(mfrow = c(2, 2))
      on.exit(par(op), add = TRUE)

      plot(
        tray$time,
        tray$loss,
        type = "l",
        lwd = 2,
        xlab = "Tiempo",
        ylab = "Pérdida",
        main = "Dinámica: pérdida"
      )
      grid()

      plot(
        tray$time,
        tray$grad_norm,
        type = "l",
        lwd = 2,
        xlab = "Tiempo",
        ylab = "Norma gradiente",
        main = "Dinámica: gradiente"
      )
      grid()

      if ("speed" %in% names(tray)) {
        plot(
          tray$time,
          tray$speed,
          type = "l",
          lwd = 2,
          xlab = "Tiempo",
          ylab = "Velocidad",
          main = "Dinámica: velocidad"
        )
        grid()
      }

      if (all(c("w1", "w2") %in% names(tray))) {
        plot(
          tray$w1,
          tray$w2,
          type = "l",
          lwd = 2,
          xlab = "w1",
          ylab = "w2",
          main = "Trayectoria paramétrica"
        )
        points(tray$w1[1], tray$w2[1], pch = 19)
        points(tray$w1[nrow(tray)], tray$w2[nrow(tray)], pch = 17)
        grid()
      }
    })

    output$interpretacion <- renderText({
      req(res())

      tray <- res()$dinamica$trayectoria

      loss_ini <- tray$loss[1]
      loss_fin <- tray$loss[nrow(tray)]

      paste0(
        "La dinámica continua aproximada inició con pérdida ",
        round(loss_ini, 6),
        " y finalizó con pérdida ",
        round(loss_fin, 6),
        ". El proceso se integró con dt = ",
        input$dt,
        " durante ",
        input$T_iter,
        " pasos. La trayectoria permite observar pérdida, gradiente, velocidad y desplazamiento paramétrico."
      )
    })
  })
}
