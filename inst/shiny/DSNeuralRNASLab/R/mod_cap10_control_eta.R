# ============================================================
# Modulo Capitulo 10: Control de tasa de aprendizaje
# DSNeuralRNASLab
# ============================================================

mod_cap10_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 10: Control de tasa de aprendizaje"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite entrenar una neurona RNAS con políticas de control de tasa de aprendizaje. El usuario puede modificar eta inicial, iteraciones, activación, pesos iniciales y política de control.")
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

        numericInput(ns("eta0"), "Eta inicial", value = 0.1, min = 0.001, max = 1, step = 0.005),
        numericInput(ns("eta_min"), "Eta mínima", value = 0.001, min = 0.0001, max = 1, step = 0.001),
        numericInput(ns("eta_max"), "Eta máxima", value = 0.1, min = 0.001, max = 1, step = 0.005),

        numericInput(ns("T_iter"), "Número de iteraciones", value = 100, min = 5, max = 1000, step = 5),

        numericInput(ns("w1"), "w1 inicial", value = 0.8, step = 0.05),
        numericInput(ns("w2"), "w2 inicial", value = 0.3, step = 0.05),
        numericInput(ns("b"), "b inicial", value = 0.1, step = 0.05),

        selectInput(
          ns("policy_type"),
          "Política de control",
          choices = c(
            "Constante" = "constante",
            "Temporal" = "temporal",
            "Mejora relativa" = "mejora",
            "Régimen dinámico" = "regimen",
            "Inestabilidad" = "inestabilidad"
          ),
          selected = "constante"
        ),

        numericInput(ns("alpha"), "Alpha temporal", value = 0.01, min = 0, step = 0.01),
        numericInput(ns("tau_loss"), "Umbral tau_loss", value = 0.01, min = 0, step = 0.005),
        numericInput(ns("gamma_mejora"), "Gamma mejora", value = 0.95, min = 0.01, max = 1, step = 0.01),
        numericInput(ns("gamma_ref"), "Gamma refinamiento", value = 0.98, min = 0.01, max = 1, step = 0.01),
        numericInput(ns("gamma_sat"), "Gamma saturación", value = 0.90, min = 0.01, max = 1, step = 0.01),
        numericInput(ns("gamma_ine"), "Gamma inestabilidad", value = 0.50, min = 0.01, max = 1, step = 0.01),
        numericInput(ns("gamma_est"), "Gamma estabilización", value = 0.95, min = 0.01, max = 1, step = 0.01),

        actionButton(ns("run"), "Ejecutar control", class = "btn-primary")
      ),

      column(
        8,
        tabsetPanel(
          tabPanel("Código", br(), uiOutput(ns("codigo"))),
          tabPanel("Datos", br(), DT::DTOutput(ns("tabla_datos"))),
          tabPanel("Resumen", br(), DT::DTOutput(ns("tabla_resumen"))),
          tabPanel("Trayectoria", br(), DT::DTOutput(ns("tabla_trayectoria"))),
          tabPanel("Gráficas", br(), plotOutput(ns("plot"), height = "440px")),
          tabPanel("Interpretación", br(), verbatimTextOutput(ns("interpretacion")))
        )
      )
    )
  )
}

mod_cap10_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    construir_policy_cap10 <- function() {
      switch(
        input$policy_type,
        constante = "constante",
        temporal = list(
          tipo = "temporal",
          alpha = input$alpha
        ),
        mejora = list(
          tipo = "mejora",
          tau_loss = input$tau_loss,
          gamma = input$gamma_mejora
        ),
        regimen = list(
          tipo = "regimen",
          gamma_refinamiento = input$gamma_ref,
          gamma_saturacion = input$gamma_sat,
          gamma_inestabilidad = input$gamma_ine,
          gamma_estabilizacion = input$gamma_est
        ),
        inestabilidad = list(
          tipo = "inestabilidad",
          gamma = input$gamma_ine
        )
      )
    }

    res <- eventReactive(input$run, {
      datos <- generar_datos_neurona_base()

      X <- datos$X
      y <- datos$y
      theta0 <- c(input$w1, input$w2, input$b)
      policy <- construir_policy_cap10()

      modelo <- DSNeuralRNAS::rnas_train_neuron_control_eta(
        X = X,
        y = y,
        theta0 = theta0,
        eta0 = input$eta0,
        policy = policy,
        eta_min = input$eta_min,
        eta_max = input$eta_max,
        T = as.integer(input$T_iter),
        activation = input$activation
      )

      list(
        datos = datos,
        modelo = modelo,
        policy = policy
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
        "  policy = \"", input$policy_type, "\",\n",
        "  eta_min = ", input$eta_min, ",\n",
        "  eta_max = ", input$eta_max, ",\n",
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
          "eta_cambios",
          "velocidad_media",
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
          m$eta_cambios,
          m$velocidad_media,
          m$grad_norm_inicial,
          m$grad_norm_final,
          as.numeric(m$descendente_global)
        ),
        stringsAsFactors = FALSE
      )

      tabla_dt(tab)
    })

    output$tabla_trayectoria <- DT::renderDT({
      req(res())

      tray <- res()$modelo$trayectoria

      columnas <- intersect(
        c("iter", "loss", "grad_norm", "eta", "speed", "accion_eta", "regimen", "b", "w1", "w2"),
        names(tray)
      )

      tabla_dt(utils::head(tray[, columnas, drop = FALSE], 25))
    })

    output$plot <- renderPlot({
      req(res())

      tray <- res()$modelo$trayectoria

      op <- par(mfrow = c(2, 2))
      on.exit(par(op), add = TRUE)

      plot(tray$iter, tray$loss, type = "l", lwd = 2,
           xlab = "Iteración", ylab = "Pérdida",
           main = "Curva de pérdida")
      grid()

      plot(tray$iter, tray$eta, type = "l", lwd = 2,
           xlab = "Iteración", ylab = "Eta",
           main = "Evolución de eta")
      grid()

      plot(tray$iter, tray$grad_norm, type = "l", lwd = 2,
           xlab = "Iteración", ylab = "Norma gradiente",
           main = "Norma del gradiente")
      grid()

      if ("speed" %in% names(tray)) {
        plot(tray$iter, tray$speed, type = "l", lwd = 2,
             xlab = "Iteración", ylab = "Velocidad",
             main = "Velocidad dinámica")
        grid()
      }
    })

    output$interpretacion <- renderText({
      req(res())

      m <- res()$modelo$metricas

      paste0(
        "La política de control seleccionada fue '", input$policy_type, "'. ",
        "La pérdida inicial fue ", round(m$loss_inicial, 6),
        " y la pérdida final fue ", round(m$loss_final, 6),
        ". La reducción relativa fue ", round(m$reduccion_rel, 4),
        ". La tasa media observada fue ", round(m$eta_media, 6),
        ", con ", m$eta_cambios, " cambios detectados en eta."
      )
    })
  })
}
