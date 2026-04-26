# ============================================================
# Modulo Capitulo 6: MLP simple
# DSNeuralRNASLab
# Version ajustada a funciones reales del paquete:
# rnas_train_mlp()
# rnas_predict_mlp()
# rnas_resumen_entrenamiento_mlp()
# ============================================================

mod_cap06_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 6: MLP simple"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite ejecutar una arquitectura MLP simple del modelo DS Neural RNAS. El usuario puede modificar la activación, la tasa de aprendizaje, el número de iteraciones, la cantidad de neuronas ocultas, la escala de inicialización, la semilla y los criterios de registro.")
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
          ns("eta"),
          "Eta",
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

        numericInput(
          ns("d_hidden"),
          "Neuronas ocultas",
          value = 3,
          min = 1,
          max = 20,
          step = 1
        ),

        numericInput(
          ns("init_sd"),
          "Desviación inicial init_sd",
          value = 0.1,
          min = 0.001,
          max = 2,
          step = 0.01
        ),

        numericInput(
          ns("seed"),
          "Semilla",
          value = 123,
          min = 1,
          step = 1
        ),

        numericInput(
          ns("registrar_cada"),
          "Registrar cada",
          value = 1,
          min = 1,
          max = 100,
          step = 1
        ),

        checkboxInput(
          ns("usar_tol_loss"),
          "Usar tolerancia de pérdida",
          value = FALSE
        ),

        conditionalPanel(
          condition = sprintf("input['%s'] == true", ns("usar_tol_loss")),
          numericInput(
            ns("tol_loss"),
            "tol_loss",
            value = 1e-8,
            min = 1e-12,
            step = 1e-8
          )
        ),

        checkboxInput(
          ns("usar_tol_grad"),
          "Usar tolerancia de gradiente",
          value = FALSE
        ),

        conditionalPanel(
          condition = sprintf("input['%s'] == true", ns("usar_tol_grad")),
          numericInput(
            ns("tol_grad"),
            "tol_grad",
            value = 1e-8,
            min = 1e-12,
            step = 1e-8
          )
        ),

        actionButton(
          ns("run"),
          "Ejecutar MLP",
          class = "btn-primary"
        )
      ),

      column(
        8,

        tabsetPanel(
          tabPanel(
            "Código",
            br(),
            uiOutput(ns("codigo"))
          ),

          tabPanel(
            "Datos",
            br(),
            DT::DTOutput(ns("tabla_datos"))
          ),

          tabPanel(
            "Resumen",
            br(),
            DT::DTOutput(ns("tabla_resumen"))
          ),

          tabPanel(
            "Trayectoria",
            br(),
            DT::DTOutput(ns("tabla_trayectoria"))
          ),

          tabPanel(
            "Predicciones",
            br(),
            DT::DTOutput(ns("tabla_predicciones"))
          ),

          tabPanel(
            "Gráficas",
            br(),
            plotOutput(ns("plot"), height = "460px")
          ),

          tabPanel(
            "Interpretación",
            br(),
            verbatimTextOutput(ns("interpretacion"))
          ),

          tabPanel(
            "Objeto",
            br(),
            verbatimTextOutput(ns("estructura"))
          )
        )
      )
    )
  )
}


mod_cap06_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    # --------------------------------------------------------
    # Ejecución principal
    # --------------------------------------------------------

    res <- eventReactive(input$run, {
      set.seed(as.integer(input$seed))

      X <- matrix(
        c(
          0.5, -0.2,
          1.0,  0.3,
          -0.7,  0.8,
          0.0,  0.0,
          0.4,  0.6,
          -0.3, -0.5
        ),
        ncol = 2,
        byrow = TRUE
      )

      y <- c(0.4, 0.8, -0.2, 0.1, 0.6, -0.3)

      tol_loss_val <- if (isTRUE(input$usar_tol_loss)) input$tol_loss else NULL
      tol_grad_val <- if (isTRUE(input$usar_tol_grad)) input$tol_grad else NULL

      modelo <- rnas_train_mlp(
        X = X,
        y = y,
        d_hidden = as.integer(input$d_hidden),
        params0 = NULL,
        eta = input$eta,
        T = as.integer(input$T_iter),
        activation = input$activation,
        init_sd = input$init_sd,
        seed = as.integer(input$seed),
        tol_loss = tol_loss_val,
        tol_grad = tol_grad_val,
        registrar_cada = as.integer(input$registrar_cada)
      )

      pred <- rnas_predict_mlp(
        object = modelo,
        X = X,
        activation = input$activation
      )

      resumen <- tryCatch(
        {
          rnas_resumen_entrenamiento_mlp(modelo)
        },
        error = function(e) {
          NULL
        }
      )

      trayectoria <- extraer_trayectoria_cap06(modelo)
      metricas <- extraer_metricas_cap06(modelo, resumen, trayectoria)

      list(
        X = X,
        y = y,
        modelo = modelo,
        pred = as.numeric(pred),
        resumen = resumen,
        trayectoria = trayectoria,
        metricas = metricas,
        configuracion = list(
          d_hidden = as.integer(input$d_hidden),
          eta = input$eta,
          T = as.integer(input$T_iter),
          activation = input$activation,
          init_sd = input$init_sd,
          seed = as.integer(input$seed),
          registrar_cada = as.integer(input$registrar_cada),
          tol_loss = tol_loss_val,
          tol_grad = tol_grad_val
        )
      )
    })

    # --------------------------------------------------------
    # Código mostrado al usuario
    # --------------------------------------------------------

    output$codigo <- renderUI({
      tol_loss_txt <- if (isTRUE(input$usar_tol_loss)) {
        as.character(input$tol_loss)
      } else {
        "NULL"
      }

      tol_grad_txt <- if (isTRUE(input$usar_tol_grad)) {
        as.character(input$tol_grad)
      } else {
        "NULL"
      }

      codigo <- paste0(
        "X <- matrix(\n",
        "  c(\n",
        "    0.5, -0.2,\n",
        "    1.0,  0.3,\n",
        "   -0.7,  0.8,\n",
        "    0.0,  0.0,\n",
        "    0.4,  0.6,\n",
        "   -0.3, -0.5\n",
        "  ),\n",
        "  ncol = 2,\n",
        "  byrow = TRUE\n",
        ")\n\n",
        "y <- c(0.4, 0.8, -0.2, 0.1, 0.6, -0.3)\n\n",
        "modelo <- rnas_train_mlp(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  d_hidden = ", input$d_hidden, ",\n",
        "  params0 = NULL,\n",
        "  eta = ", input$eta, ",\n",
        "  T = ", input$T_iter, ",\n",
        "  activation = \"", input$activation, "\",\n",
        "  init_sd = ", input$init_sd, ",\n",
        "  seed = ", input$seed, ",\n",
        "  tol_loss = ", tol_loss_txt, ",\n",
        "  tol_grad = ", tol_grad_txt, ",\n",
        "  registrar_cada = ", input$registrar_cada, "\n",
        ")\n\n",
        "pred <- rnas_predict_mlp(\n",
        "  object = modelo,\n",
        "  X = X,\n",
        "  activation = \"", input$activation, "\"\n",
        ")\n\n",
        "resumen <- rnas_resumen_entrenamiento_mlp(modelo)"
      )

      mostrar_codigo(codigo)
    })

    # --------------------------------------------------------
    # Datos de entrada
    # --------------------------------------------------------

    output$tabla_datos <- DT::renderDT({
      req(res())

      tab <- data.frame(
        obs = seq_len(nrow(res()$X)),
        x1 = res()$X[, 1],
        x2 = res()$X[, 2],
        y_obs = res()$y,
        stringsAsFactors = FALSE
      )

      tabla_dt(tab)
    })

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    output$tabla_resumen <- DT::renderDT({
      req(res())

      m <- res()$metricas

      tab <- data.frame(
        indicador = names(m),
        valor = unlist(m, use.names = FALSE),
        stringsAsFactors = FALSE
      )

      tabla_dt(tab)
    })

    # --------------------------------------------------------
    # Trayectoria
    # --------------------------------------------------------

    output$tabla_trayectoria <- DT::renderDT({
      req(res())

      tray <- res()$trayectoria

      if (is.null(tray) || !is.data.frame(tray) || nrow(tray) == 0) {
        tab <- data.frame(
          mensaje = "El objeto MLP no contiene una trayectoria reconocible.",
          stringsAsFactors = FALSE
        )
      } else {
        columnas <- intersect(
          c("iter", "time", "loss", "grad_norm", "eta", "velocidad", "speed"),
          names(tray)
        )

        if (length(columnas) == 0) {
          tab <- utils::head(tray, 25)
        } else {
          tab <- utils::head(tray[, columnas, drop = FALSE], 25)
        }
      }

      tabla_dt(tab)
    })

    # --------------------------------------------------------
    # Predicciones
    # --------------------------------------------------------

    output$tabla_predicciones <- DT::renderDT({
      req(res())

      pred <- res()$pred

      tab <- data.frame(
        obs = seq_along(res()$y),
        y_obs = res()$y,
        y_hat = pred,
        error = pred - res()$y,
        error_abs = abs(pred - res()$y),
        error_cuadrado = (pred - res()$y)^2,
        stringsAsFactors = FALSE
      )

      tabla_dt(tab)
    })

    # --------------------------------------------------------
    # Gráficas
    # --------------------------------------------------------

    output$plot <- renderPlot({
      req(res())

      tray <- res()$trayectoria
      pred <- res()$pred

      op <- par(mfrow = c(1, 2))
      on.exit(par(op), add = TRUE)

      if (!is.null(tray) &&
          is.data.frame(tray) &&
          nrow(tray) > 0 &&
          "loss" %in% names(tray)) {

        eje_x <- if ("iter" %in% names(tray)) {
          tray$iter
        } else if ("time" %in% names(tray)) {
          tray$time
        } else {
          seq_len(nrow(tray))
        }

        plot(
          eje_x,
          tray$loss,
          type = "l",
          lwd = 2,
          xlab = "Iteración",
          ylab = "Pérdida",
          main = "MLP: curva de pérdida"
        )
        grid()

      } else {
        plot.new()
        title("MLP: pérdida")
        text(0.5, 0.5, "Trayectoria no disponible.")
      }

      plot(
        res()$y,
        pred,
        pch = 19,
        xlab = "Observado",
        ylab = "Predicho",
        main = "MLP: observado vs predicho"
      )
      abline(0, 1, lty = 2)
      grid()
    })

    # --------------------------------------------------------
    # Interpretación
    # --------------------------------------------------------

    output$interpretacion <- renderText({
      req(res())

      m <- res()$metricas

      loss_inicial <- obtener_valor_cap06(m, c("loss_inicial", "loss_ini", "L0"))
      loss_final <- obtener_valor_cap06(m, c("loss_final", "loss_fin", "LT"))
      reduccion_rel <- obtener_valor_cap06(m, c("reduccion_rel", "reduccion_relativa"))

      texto_loss_ini <- ifelse(is.na(loss_inicial), "no disponible", round(loss_inicial, 6))
      texto_loss_fin <- ifelse(is.na(loss_final), "no disponible", round(loss_final, 6))
      texto_red <- ifelse(is.na(reduccion_rel), "no disponible", round(reduccion_rel, 4))

      paste0(
        "El MLP simple fue entrenado con ",
        input$d_hidden,
        " neuronas ocultas, activación ",
        input$activation,
        ", eta = ",
        input$eta,
        " y ",
        input$T_iter,
        " iteraciones. La pérdida inicial fue ",
        texto_loss_ini,
        " y la pérdida final fue ",
        texto_loss_fin,
        ". La reducción relativa fue ",
        texto_red,
        ". La tabla de predicciones permite comparar y observado contra y_hat."
      )
    })

    # --------------------------------------------------------
    # Estructura del objeto
    # --------------------------------------------------------

    output$estructura <- renderPrint({
      req(res())
      str(res()$modelo, max.level = 3)
    })
  })
}


# ============================================================
# Funciones auxiliares internas del modulo Capitulo 6
# ============================================================

extraer_trayectoria_cap06 <- function(modelo) {
  if (!is.null(modelo$trayectoria) && is.data.frame(modelo$trayectoria)) {
    return(modelo$trayectoria)
  }

  if (!is.null(modelo$historial) && is.data.frame(modelo$historial)) {
    return(modelo$historial)
  }

  if (!is.null(modelo$history) && is.data.frame(modelo$history)) {
    return(modelo$history)
  }

  if (!is.null(modelo$loss) && is.numeric(modelo$loss)) {
    return(data.frame(
      iter = seq_along(modelo$loss) - 1L,
      loss = as.numeric(modelo$loss),
      stringsAsFactors = FALSE
    ))
  }

  data.frame()
}


extraer_metricas_cap06 <- function(modelo, resumen, trayectoria) {
  if (!is.null(resumen)) {
    if (is.data.frame(resumen)) {
      if (all(c("indicador", "valor") %in% names(resumen))) {
        vals <- resumen$valor
        names(vals) <- resumen$indicador
        return(as.list(vals))
      }

      if (nrow(resumen) == 1L) {
        return(as.list(resumen[1, , drop = TRUE]))
      }
    }

    if (is.list(resumen)) {
      return(resumen)
    }
  }

  if (!is.null(modelo$metricas) && is.list(modelo$metricas)) {
    return(modelo$metricas)
  }

  if (!is.null(modelo$resumen) && is.list(modelo$resumen)) {
    return(modelo$resumen)
  }

  if (!is.null(trayectoria) &&
      is.data.frame(trayectoria) &&
      nrow(trayectoria) > 0 &&
      "loss" %in% names(trayectoria)) {

    loss_inicial <- trayectoria$loss[1]
    loss_final <- trayectoria$loss[nrow(trayectoria)]
    reduccion_abs <- loss_inicial - loss_final
    reduccion_rel <- reduccion_abs / (abs(loss_inicial) + .Machine$double.eps)

    return(list(
      loss_inicial = loss_inicial,
      loss_final = loss_final,
      reduccion_abs = reduccion_abs,
      reduccion_rel = reduccion_rel,
      n_registros = nrow(trayectoria)
    ))
  }

  list(
    estado = "modelo_ejecutado",
    mensaje = "No se encontraron metricas estandar ni trayectoria con columna loss."
  )
}


obtener_valor_cap06 <- function(metricas, nombres) {
  if (is.null(metricas) || length(metricas) == 0L) {
    return(NA_real_)
  }

  for (nm in nombres) {
    if (!is.null(metricas[[nm]])) {
      val <- suppressWarnings(as.numeric(metricas[[nm]][1]))
      return(val)
    }
  }

  NA_real_
}
