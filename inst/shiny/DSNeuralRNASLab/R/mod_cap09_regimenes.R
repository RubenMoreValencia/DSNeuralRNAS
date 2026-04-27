# ============================================================
# Modulo Capitulo 9: Regimenes dinamicos
# DSNeuralRNASLab
# Version corregida:
# rnas_analizar_regimenes_neuron() requiere objeto
# de clase rnas_neuron_dynamics, por tanto se usa
# rnas_integrar_dinamica_neuron().
# ============================================================

mod_cap09_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 9: Regímenes dinámicos del aprendizaje"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite analizar los regímenes dinámicos de una trayectoria de aprendizaje RNAS generada como dinámica continua. El usuario puede modificar eta, dt, iteraciones, activación, ventana, umbrales de pérdida, gradiente, velocidad y suavizado.")
    ),

    fluidRow(
      column(
        4,

        h4("Dinámica base"),

        selectInput(
          ns("activation"),
          "Activación",
          choices = c("tanh", "sigmoid"),
          selected = "tanh"
        ),

        numericInput(
          ns("eta"),
          "Eta dinámica",
          value = 0.1,
          min = 0.001,
          max = 1,
          step = 0.005
        ),

        numericInput(
          ns("dt"),
          "Paso temporal dt",
          value = 0.1,
          min = 0.001,
          max = 1,
          step = 0.01
        ),

        numericInput(
          ns("T_iter"),
          "Número de pasos",
          value = 100,
          min = 10,
          max = 1000,
          step = 5
        ),

        numericInput(
          ns("w1"),
          "w1 inicial",
          value = 0.8,
          step = 0.05
        ),

        numericInput(
          ns("w2"),
          "w2 inicial",
          value = 0.3,
          step = 0.05
        ),

        numericInput(
          ns("b"),
          "b inicial",
          value = 0.1,
          step = 0.05
        ),

        hr(),

        h4("Parámetros de régimen"),

        numericInput(
          ns("ventana"),
          "Ventana de suavizado",
          value = 5,
          min = 1,
          max = 50,
          step = 1
        ),

        numericInput(
          ns("tau_loss"),
          "tau_loss",
          value = 0.01,
          min = 0,
          step = 0.001
        ),

        numericInput(
          ns("tau_grad"),
          "tau_grad",
          value = 0.001,
          min = 0,
          step = 0.0005
        ),

        numericInput(
          ns("eps_loss"),
          "eps_loss",
          value = 1e-8,
          min = 1e-12,
          step = 1e-8
        ),

        numericInput(
          ns("eps_grad"),
          "eps_grad",
          value = 1e-4,
          min = 1e-8,
          step = 1e-4
        ),

        checkboxInput(
          ns("usar_tau_speed"),
          "Usar umbral de velocidad",
          value = FALSE
        ),

        conditionalPanel(
          condition = sprintf("input['%s'] == true", ns("usar_tau_speed")),
          numericInput(
            ns("tau_speed"),
            "tau_speed",
            value = 0.05,
            min = 0,
            step = 0.005
          )
        ),

        checkboxInput(
          ns("usar_loss_alta"),
          "Usar umbral de pérdida alta",
          value = FALSE
        ),

        conditionalPanel(
          condition = sprintf("input['%s'] == true", ns("usar_loss_alta")),
          numericInput(
            ns("loss_alta"),
            "loss_alta",
            value = 0.1,
            min = 0,
            step = 0.01
          )
        ),

        checkboxInput(
          ns("usar_suavizado"),
          "Usar suavizado",
          value = TRUE
        ),

        actionButton(
          ns("run"),
          "Analizar regímenes",
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
            "Dinámica",
            br(),
            DT::DTOutput(ns("tabla_dinamica"))
          ),

          tabPanel(
            "Señales",
            br(),
            DT::DTOutput(ns("tabla_senales"))
          ),

          tabPanel(
            "Trayectoria con regímenes",
            br(),
            DT::DTOutput(ns("tabla_regimenes"))
          ),

          tabPanel(
            "Resumen",
            br(),
            DT::DTOutput(ns("tabla_resumen"))
          ),

          tabPanel(
            "Segmentos",
            br(),
            DT::DTOutput(ns("tabla_segmentos"))
          ),

          tabPanel(
            "Gráficas",
            br(),
            plotOutput(ns("plot"), height = "500px")
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


mod_cap09_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      datos <- generar_datos_neurona_base()

      X <- datos$X
      y <- datos$y
      theta0 <- c(input$w1, input$w2, input$b)

      dinamica <- DSNeuralRNAS::rnas_integrar_dinamica_neuron(
        X = X,
        y = y,
        theta0 = theta0,
        eta = input$eta,
        dt = input$dt,
        T = as.integer(input$T_iter),
        activation = input$activation
      )

      trayectoria <- dinamica$trayectoria

      tau_speed_val <- if (isTRUE(input$usar_tau_speed)) {
        input$tau_speed
      } else {
        Inf
      }

      loss_alta_val <- if (isTRUE(input$usar_loss_alta)) {
        input$loss_alta
      } else {
        NULL
      }

      senales <- DSNeuralRNAS::rnas_calcular_senales_regimen(
        trayectoria = trayectoria,
        loss_col = "loss",
        grad_col = "grad_norm",
        speed_col = "speed",
        ventana = as.integer(input$ventana),
        eps = 1e-12
      )

      trayectoria_regimen <- DSNeuralRNAS::rnas_clasificar_regimenes(
        senales = senales,
        tau_loss = input$tau_loss,
        tau_grad = input$tau_grad,
        eps_loss = input$eps_loss,
        eps_grad = input$eps_grad,
        tau_speed = tau_speed_val,
        loss_alta = loss_alta_val,
        usar_suavizado = isTRUE(input$usar_suavizado)
      )

      resumen <- DSNeuralRNAS::rnas_resumen_regimenes(
        trayectoria_regimen = trayectoria_regimen,
        loss_col = "loss",
        grad_col = "grad_norm",
        speed_col = "speed"
      )

      segmentos <- DSNeuralRNAS::rnas_segmentar_regimenes(
        trayectoria_regimen = trayectoria_regimen,
        iter_col = "iter",
        loss_col = "loss",
        grad_col = "grad_norm",
        speed_col = "speed"
      )

      analisis <- DSNeuralRNAS::rnas_analizar_regimenes_neuron(
        object = dinamica,
        ventana = as.integer(input$ventana),
        tau_loss = input$tau_loss,
        tau_grad = input$tau_grad,
        eps_loss = input$eps_loss,
        eps_grad = input$eps_grad,
        tau_speed = tau_speed_val,
        loss_alta = loss_alta_val,
        usar_suavizado = isTRUE(input$usar_suavizado)
      )

      list(
        datos = datos,
        dinamica = dinamica,
        trayectoria = trayectoria,
        senales = senales,
        trayectoria_regimen = trayectoria_regimen,
        resumen = resumen,
        segmentos = segmentos,
        analisis = analisis,
        configuracion = list(
          activation = input$activation,
          eta = input$eta,
          dt = input$dt,
          T = as.integer(input$T_iter),
          theta0 = theta0,
          ventana = as.integer(input$ventana),
          tau_loss = input$tau_loss,
          tau_grad = input$tau_grad,
          eps_loss = input$eps_loss,
          eps_grad = input$eps_grad,
          tau_speed = tau_speed_val,
          loss_alta = loss_alta_val,
          usar_suavizado = isTRUE(input$usar_suavizado)
        )
      )
    })

    output$codigo <- renderUI({
      tau_speed_txt <- if (isTRUE(input$usar_tau_speed)) {
        as.character(input$tau_speed)
      } else {
        "Inf"
      }

      loss_alta_txt <- if (isTRUE(input$usar_loss_alta)) {
        as.character(input$loss_alta)
      } else {
        "NULL"
      }

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
        ")\n\n",
        "senales <- rnas_calcular_senales_regimen(\n",
        "  trayectoria = dinamica$trayectoria,\n",
        "  loss_col = \"loss\",\n",
        "  grad_col = \"grad_norm\",\n",
        "  speed_col = \"speed\",\n",
        "  ventana = ", input$ventana, ",\n",
        "  eps = 1e-12\n",
        ")\n\n",
        "trayectoria_regimen <- rnas_clasificar_regimenes(\n",
        "  senales = senales,\n",
        "  tau_loss = ", input$tau_loss, ",\n",
        "  tau_grad = ", input$tau_grad, ",\n",
        "  eps_loss = ", input$eps_loss, ",\n",
        "  eps_grad = ", input$eps_grad, ",\n",
        "  tau_speed = ", tau_speed_txt, ",\n",
        "  loss_alta = ", loss_alta_txt, ",\n",
        "  usar_suavizado = ", isTRUE(input$usar_suavizado), "\n",
        ")\n\n",
        "resumen <- rnas_resumen_regimenes(\n",
        "  trayectoria_regimen = trayectoria_regimen,\n",
        "  loss_col = \"loss\",\n",
        "  grad_col = \"grad_norm\",\n",
        "  speed_col = \"speed\"\n",
        ")\n\n",
        "segmentos <- rnas_segmentar_regimenes(\n",
        "  trayectoria_regimen = trayectoria_regimen,\n",
        "  iter_col = \"iter\",\n",
        "  loss_col = \"loss\",\n",
        "  grad_col = \"grad_norm\",\n",
        "  speed_col = \"speed\"\n",
        ")\n\n",
        "analisis <- rnas_analizar_regimenes_neuron(\n",
        "  object = dinamica,\n",
        "  ventana = ", input$ventana, ",\n",
        "  tau_loss = ", input$tau_loss, ",\n",
        "  tau_grad = ", input$tau_grad, ",\n",
        "  eps_loss = ", input$eps_loss, ",\n",
        "  eps_grad = ", input$eps_grad, ",\n",
        "  tau_speed = ", tau_speed_txt, ",\n",
        "  loss_alta = ", loss_alta_txt, ",\n",
        "  usar_suavizado = ", isTRUE(input$usar_suavizado), "\n",
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

    output$tabla_dinamica <- DT::renderDT({
      req(res())

      trayectoria <- res()$trayectoria

      if (is.data.frame(trayectoria)) {
        columnas <- intersect(
          c("iter", "time", "loss", "grad_norm", "speed", "w1", "w2", "b"),
          names(trayectoria)
        )

        if (length(columnas) == 0L) {
          tab <- utils::head(trayectoria, 40)
        } else {
          tab <- utils::head(trayectoria[, columnas, drop = FALSE], 40)
        }
      } else {
        tab <- convertir_lista_a_tabla_cap09(trayectoria)
      }

      tabla_dt(tab)
    })

    output$tabla_senales <- DT::renderDT({
      req(res())

      senales <- res()$senales

      if (is.data.frame(senales)) {
        tab <- utils::head(senales, 40)
      } else if (is.list(senales)) {
        tab <- convertir_lista_a_tabla_cap09(senales)
      } else {
        tab <- data.frame(
          resultado = as.character(senales),
          stringsAsFactors = FALSE
        )
      }

      tabla_dt(tab)
    })

    output$tabla_regimenes <- DT::renderDT({
      req(res())

      tr <- res()$trayectoria_regimen

      if (is.data.frame(tr)) {
        tab <- utils::head(tr, 50)
      } else if (is.list(tr)) {
        tab <- convertir_lista_a_tabla_cap09(tr)
      } else {
        tab <- data.frame(
          resultado = as.character(tr),
          stringsAsFactors = FALSE
        )
      }

      tabla_dt(tab)
    })

    output$tabla_resumen <- DT::renderDT({
      req(res())

      resumen <- res()$resumen

      if (is.data.frame(resumen)) {
        tab <- resumen
      } else if (is.list(resumen)) {
        tab <- convertir_lista_a_tabla_cap09(resumen)
      } else {
        tab <- data.frame(
          resultado = as.character(resumen),
          stringsAsFactors = FALSE
        )
      }

      tabla_dt(tab)
    })

    output$tabla_segmentos <- DT::renderDT({
      req(res())

      seg <- res()$segmentos

      if (is.data.frame(seg)) {
        tab <- seg
      } else if (is.list(seg)) {
        tab <- convertir_lista_a_tabla_cap09(seg)
      } else {
        tab <- data.frame(
          resultado = as.character(seg),
          stringsAsFactors = FALSE
        )
      }

      tabla_dt(tab)
    })

    output$plot <- renderPlot({
      req(res())

      tr <- res()$trayectoria_regimen
      resumen <- res()$resumen

      op <- par(mfrow = c(2, 2))
      on.exit(par(op), add = TRUE)

      plot_columna_iter_cap09(
        data = tr,
        y_col = "loss",
        ylab = "Pérdida",
        titulo = "Pérdida por iteración"
      )

      plot_columna_iter_cap09(
        data = tr,
        y_col = "grad_norm",
        ylab = "Norma gradiente",
        titulo = "Gradiente por iteración"
      )

      plot_columna_iter_cap09(
        data = tr,
        y_col = "speed",
        ylab = "Velocidad",
        titulo = "Velocidad dinámica"
      )

      plot_frecuencia_regimen_cap09(tr, resumen)
    })

    output$interpretacion <- renderText({
      req(res())

      tr <- res()$trayectoria_regimen
      seg <- res()$segmentos

      regimen_col <- detectar_columna_regimen_cap09(tr)

      if (!is.null(regimen_col) && is.data.frame(tr)) {
        freq <- sort(table(tr[[regimen_col]]), decreasing = TRUE)
        regimen_pred <- names(freq)[1]
        n_regimenes <- length(freq)
      } else {
        regimen_pred <- "no disponible"
        n_regimenes <- NA
      }

      n_segmentos <- if (is.data.frame(seg)) {
        nrow(seg)
      } else {
        NA
      }

      loss_ini <- obtener_loss_cap09(tr, posicion = "inicial")
      loss_fin <- obtener_loss_cap09(tr, posicion = "final")

      paste0(
        "El análisis de regímenes se aplicó sobre una trayectoria de dinámica continua generada por rnas_integrar_dinamica_neuron(). ",
        "Con ventana = ", input$ventana,
        ", tau_loss = ", input$tau_loss,
        " y tau_grad = ", input$tau_grad,
        ", el régimen predominante fue: ",
        regimen_pred,
        ". Número de regímenes detectados: ",
        ifelse(is.na(n_regimenes), "no disponible", n_regimenes),
        ". Número de segmentos identificados: ",
        ifelse(is.na(n_segmentos), "no disponible", n_segmentos),
        ". La pérdida pasó de ",
        ifelse(is.na(loss_ini), "no disponible", round(loss_ini, 6)),
        " a ",
        ifelse(is.na(loss_fin), "no disponible", round(loss_fin, 6)),
        "."
      )
    })

    output$estructura <- renderPrint({
      req(res())
      str(res(), max.level = 3)
    })
  })
}


# ============================================================
# Auxiliares Capitulo 9
# ============================================================

convertir_lista_a_tabla_cap09 <- function(x) {
  if (is.null(x)) {
    return(data.frame(
      indicador = "resultado",
      valor = NA_character_,
      stringsAsFactors = FALSE
    ))
  }

  if (is.data.frame(x)) {
    return(x)
  }

  if (is.atomic(x) && !is.list(x)) {
    return(data.frame(
      indicador = paste0("valor_", seq_along(x)),
      valor = as.character(x),
      stringsAsFactors = FALSE
    ))
  }

  if (is.list(x)) {
    if (is.null(names(x))) {
      names(x) <- paste0("elemento_", seq_along(x))
    }

    tab <- do.call(
      rbind,
      lapply(names(x), function(nm) {
        val <- x[[nm]]

        if (is.null(val)) {
          val_txt <- NA_character_
        } else if (is.data.frame(val)) {
          val_txt <- paste0("data.frame[", nrow(val), "x", ncol(val), "]")
        } else if (is.matrix(val)) {
          val_txt <- paste0("matrix[", nrow(val), "x", ncol(val), "]")
        } else if (length(val) == 1L) {
          val_txt <- as.character(val)
        } else {
          val_txt <- paste(as.character(val), collapse = ", ")
        }

        data.frame(
          indicador = nm,
          valor = val_txt,
          stringsAsFactors = FALSE
        )
      })
    )

    rownames(tab) <- NULL
    return(tab)
  }

  data.frame(
    indicador = "resultado",
    valor = as.character(x),
    stringsAsFactors = FALSE
  )
}


detectar_columna_regimen_cap09 <- function(trayectoria_regimen) {
  if (!is.data.frame(trayectoria_regimen)) {
    return(NULL)
  }

  candidatos <- c(
    "regimen",
    "regimen_detalle",
    "clase_regimen",
    "estado_regimen",
    "fase"
  )

  encontrado <- candidatos[candidatos %in% names(trayectoria_regimen)]

  if (length(encontrado) == 0L) {
    return(NULL)
  }

  encontrado[1]
}


plot_columna_iter_cap09 <- function(data, y_col, ylab, titulo) {
  if (!is.data.frame(data) || nrow(data) == 0L || !(y_col %in% names(data))) {
    plot.new()
    title(titulo)
    text(0.5, 0.5, "No disponible.")
    return(invisible(NULL))
  }

  eje_x <- if ("iter" %in% names(data)) {
    data$iter
  } else if ("time" %in% names(data)) {
    data$time
  } else {
    seq_len(nrow(data))
  }

  plot(
    eje_x,
    data[[y_col]],
    type = "l",
    lwd = 2,
    xlab = "Iteración",
    ylab = ylab,
    main = titulo
  )
  grid()

  invisible(NULL)
}


plot_frecuencia_regimen_cap09 <- function(trayectoria_regimen, resumen = NULL) {
  regimen_col <- detectar_columna_regimen_cap09(trayectoria_regimen)

  if (!is.null(regimen_col) && is.data.frame(trayectoria_regimen)) {
    freq <- table(trayectoria_regimen[[regimen_col]])

    barplot(
      freq,
      las = 2,
      ylab = "Frecuencia",
      main = "Frecuencia de regímenes"
    )
    grid()
    return(invisible(NULL))
  }

  if (is.data.frame(resumen)) {
    nms <- names(resumen)
    reg_col <- nms[grepl("regimen|fase|estado", nms, ignore.case = TRUE)][1]
    freq_col <- nms[grepl("frecuencia|n|conteo", nms, ignore.case = TRUE)][1]

    if (!is.na(reg_col) && !is.na(freq_col)) {
      vals <- resumen[[freq_col]]
      names(vals) <- resumen[[reg_col]]

      barplot(
        vals,
        las = 2,
        ylab = "Frecuencia",
        main = "Frecuencia de regímenes"
      )
      grid()
      return(invisible(NULL))
    }
  }

  plot.new()
  title("Frecuencia de regímenes")
  text(0.5, 0.5, "No disponible.")
}


obtener_loss_cap09 <- function(trayectoria_regimen, posicion = c("inicial", "final")) {
  posicion <- match.arg(posicion)

  if (!is.data.frame(trayectoria_regimen) ||
      nrow(trayectoria_regimen) == 0L ||
      !("loss" %in% names(trayectoria_regimen))) {
    return(NA_real_)
  }

  if (posicion == "inicial") {
    return(as.numeric(trayectoria_regimen$loss[1]))
  }

  as.numeric(trayectoria_regimen$loss[nrow(trayectoria_regimen)])
}
