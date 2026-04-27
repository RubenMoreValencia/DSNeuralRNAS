# ============================================================
# Modulo Capitulo 8: Geometria del aprendizaje
# DSNeuralRNASLab
# Version corregida:
# rnas_geometria_trayectoria_neuron() requiere objeto
# de clase rnas_neuron_dynamics, por tanto se usa
# rnas_integrar_dinamica_neuron().
# ============================================================

mod_cap08_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 8: Geometría del aprendizaje"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite analizar el paisaje de pérdida de una neurona RNAS, calcular Hessiano numérico, autovalores, curvatura direccional y geometría sobre una trayectoria dinámica de aprendizaje.")
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

        h4("Punto de evaluación"),

        numericInput(
          ns("w1"),
          "w1 evaluado",
          value = 0.8,
          step = 0.05
        ),

        numericInput(
          ns("w2"),
          "w2 evaluado",
          value = 0.3,
          step = 0.05
        ),

        numericInput(
          ns("b"),
          "b evaluado",
          value = 0.1,
          step = 0.05
        ),

        hr(),

        h4("Paisaje de pérdida"),

        numericInput(
          ns("w1_min"),
          "w1 mínimo grid",
          value = -1.5,
          step = 0.1
        ),

        numericInput(
          ns("w1_max"),
          "w1 máximo grid",
          value = 1.5,
          step = 0.1
        ),

        numericInput(
          ns("w2_min"),
          "w2 mínimo grid",
          value = -1.5,
          step = 0.1
        ),

        numericInput(
          ns("w2_max"),
          "w2 máximo grid",
          value = 1.5,
          step = 0.1
        ),

        numericInput(
          ns("n_grid"),
          "Resolución grid",
          value = 41,
          min = 11,
          max = 101,
          step = 2
        ),

        hr(),

        h4("Hessiano y curvatura"),

        numericInput(
          ns("h"),
          "Paso numérico Hessiano",
          value = 1e-4,
          min = 1e-8,
          step = 1e-4
        ),

        numericInput(
          ns("v1"),
          "Dirección v1",
          value = 1,
          step = 0.1
        ),

        numericInput(
          ns("v2"),
          "Dirección v2",
          value = 0,
          step = 0.1
        ),

        numericInput(
          ns("v3"),
          "Dirección v3",
          value = 0,
          step = 0.1
        ),

        hr(),

        h4("Trayectoria dinámica"),

        checkboxInput(
          ns("usar_trayectoria"),
          "Calcular geometría sobre trayectoria dinámica",
          value = TRUE
        ),

        conditionalPanel(
          condition = sprintf("input['%s'] == true", ns("usar_trayectoria")),

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
            "Pasos de dinámica",
            value = 100,
            min = 5,
            max = 1000,
            step = 5
          )
        ),

        actionButton(
          ns("run"),
          "Ejecutar geometría",
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
            "Grid de pérdida",
            br(),
            DT::DTOutput(ns("tabla_grid"))
          ),

          tabPanel(
            "Resumen geométrico",
            br(),
            DT::DTOutput(ns("tabla_resumen"))
          ),

          tabPanel(
            "Hessiano y autovalores",
            br(),
            DT::DTOutput(ns("tabla_hessian"))
          ),

          tabPanel(
            "Trayectoria geométrica",
            br(),
            DT::DTOutput(ns("tabla_trayectoria"))
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


mod_cap08_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      datos <- generar_datos_neurona_base()

      X <- datos$X
      y <- datos$y

      theta <- c(input$w1, input$w2, input$b)
      direccion <- c(input$v1, input$v2, input$v3)

      w1_seq <- seq(
        input$w1_min,
        input$w1_max,
        length.out = as.integer(input$n_grid)
      )

      w2_seq <- seq(
        input$w2_min,
        input$w2_max,
        length.out = as.integer(input$n_grid)
      )

      grid_loss <- DSNeuralRNAS::rnas_loss_grid_neuron(
        X = X,
        y = y,
        w1_seq = w1_seq,
        w2_seq = w2_seq,
        b = input$b,
        activation = input$activation
      )

      H <- DSNeuralRNAS::rnas_hessian_num_neuron(
        X = X,
        y = y,
        theta = theta,
        activation = input$activation,
        h = input$h
      )

      autovalores <- DSNeuralRNAS::rnas_autovalores_hessian(
        H = H
      )

      curvatura <- DSNeuralRNAS::rnas_curvatura_direccional(
        H = H,
        v = direccion
      )

      resumen <- DSNeuralRNAS::rnas_resumen_geometria_neuron(
        X = X,
        y = y,
        theta = theta,
        activation = input$activation,
        h = input$h,
        direccion = direccion
      )

      dinamica <- NULL
      trayectoria_geom <- NULL

      if (isTRUE(input$usar_trayectoria)) {
        dinamica <- DSNeuralRNAS::rnas_integrar_dinamica_neuron(
          X = X,
          y = y,
          theta0 = theta,
          eta = input$eta,
          dt = input$dt,
          T = as.integer(input$T_iter),
          activation = input$activation
        )

        trayectoria_geom <- DSNeuralRNAS::rnas_geometria_trayectoria_neuron(
          object = dinamica,
          X = X,
          y = y,
          iteraciones = NULL,
          h = input$h
        )
      }

      list(
        datos = datos,
        theta = theta,
        direccion = direccion,
        w1_seq = w1_seq,
        w2_seq = w2_seq,
        grid_loss = grid_loss,
        H = H,
        autovalores = autovalores,
        curvatura = curvatura,
        resumen = resumen,
        dinamica = dinamica,
        trayectoria_geom = trayectoria_geom,
        configuracion = list(
          activation = input$activation,
          theta = theta,
          direccion = direccion,
          h = input$h,
          usar_trayectoria = isTRUE(input$usar_trayectoria),
          eta = if (isTRUE(input$usar_trayectoria)) input$eta else NA_real_,
          dt = if (isTRUE(input$usar_trayectoria)) input$dt else NA_real_,
          T = if (isTRUE(input$usar_trayectoria)) as.integer(input$T_iter) else NA_integer_
        )
      )
    })

    output$codigo <- renderUI({
      codigo <- paste0(
        "datos <- generar_datos_neurona_base()\n",
        "X <- datos$X\n",
        "y <- datos$y\n\n",
        "theta <- c(", input$w1, ", ", input$w2, ", ", input$b, ")\n",
        "direccion <- c(", input$v1, ", ", input$v2, ", ", input$v3, ")\n\n",
        "grid_loss <- rnas_loss_grid_neuron(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  w1_seq = seq(", input$w1_min, ", ", input$w1_max, ", length.out = ", input$n_grid, "),\n",
        "  w2_seq = seq(", input$w2_min, ", ", input$w2_max, ", length.out = ", input$n_grid, "),\n",
        "  b = ", input$b, ",\n",
        "  activation = \"", input$activation, "\"\n",
        ")\n\n",
        "H <- rnas_hessian_num_neuron(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  theta = theta,\n",
        "  activation = \"", input$activation, "\",\n",
        "  h = ", input$h, "\n",
        ")\n\n",
        "autovalores <- rnas_autovalores_hessian(H)\n",
        "curvatura <- rnas_curvatura_direccional(H, direccion)\n\n",
        "resumen <- rnas_resumen_geometria_neuron(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  theta = theta,\n",
        "  activation = \"", input$activation, "\",\n",
        "  h = ", input$h, ",\n",
        "  direccion = direccion\n",
        ")"
      )

      if (isTRUE(input$usar_trayectoria)) {
        codigo <- paste0(
          codigo,
          "\n\n",
          "dinamica <- rnas_integrar_dinamica_neuron(\n",
          "  X = X,\n",
          "  y = y,\n",
          "  theta0 = theta,\n",
          "  eta = ", input$eta, ",\n",
          "  dt = ", input$dt, ",\n",
          "  T = ", input$T_iter, ",\n",
          "  activation = \"", input$activation, "\"\n",
          ")\n\n",
          "trayectoria_geom <- rnas_geometria_trayectoria_neuron(\n",
          "  object = dinamica,\n",
          "  X = X,\n",
          "  y = y,\n",
          "  iteraciones = NULL,\n",
          "  h = ", input$h, "\n",
          ")"
        )
      }

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

    output$tabla_grid <- DT::renderDT({
      req(res())

      grid <- res()$grid_loss

      tab <- convertir_grid_a_tabla_cap08(grid)

      tabla_dt(utils::head(tab, 80))
    })

    output$tabla_resumen <- DT::renderDT({
      req(res())

      resumen <- res()$resumen

      tab <- convertir_objeto_a_tabla_cap08(resumen)

      tabla_dt(tab)
    })

    output$tabla_hessian <- DT::renderDT({
      req(res())

      H <- res()$H
      aut <- res()$autovalores
      curv <- res()$curvatura

      H_vec <- apply(H, 1, function(z) paste(round(z, 8), collapse = ", "))

      tab_H <- data.frame(
        bloque = "Hessiano",
        indicador = paste0("fila_", seq_along(H_vec)),
        valor = H_vec,
        stringsAsFactors = FALSE
      )

      tab_aut <- convertir_objeto_a_tabla_cap08(aut)
      tab_aut <- data.frame(
        bloque = "Autovalores",
        indicador = tab_aut$indicador,
        valor = tab_aut$valor,
        stringsAsFactors = FALSE
      )

      tab_curv <- data.frame(
        bloque = "Curvatura",
        indicador = "curvatura_direccional",
        valor = paste(as.character(curv), collapse = ", "),
        stringsAsFactors = FALSE
      )

      tab <- rbind(tab_H, tab_aut, tab_curv)

      tabla_dt(tab)
    })

    output$tabla_trayectoria <- DT::renderDT({
      req(res())

      tg <- res()$trayectoria_geom

      if (is.null(tg)) {
        tab <- data.frame(
          mensaje = "No se calculó geometría sobre trayectoria. Active la opción correspondiente.",
          stringsAsFactors = FALSE
        )
      } else if (is.data.frame(tg)) {
        tab <- utils::head(tg, 40)
      } else if (is.list(tg)) {
        tab <- convertir_objeto_a_tabla_cap08(tg)
      } else {
        tab <- data.frame(
          resultado = as.character(tg),
          stringsAsFactors = FALSE
        )
      }

      tabla_dt(tab)
    })

    output$plot <- renderPlot({
      req(res())

      grid <- res()$grid_loss
      tg <- res()$trayectoria_geom

      op <- par(mfrow = c(1, 2))
      on.exit(par(op), add = TRUE)

      plot_grid_loss_cap08(grid)

      plot_trayectoria_geom_cap08(tg)
    })

    output$interpretacion <- renderText({
      req(res())

      curv <- obtener_curvatura_cap08(res()$curvatura)
      lambdas <- obtener_lambdas_cap08(res()$autovalores)

      lambda_min <- lambdas$lambda_min
      lambda_max <- lambdas$lambda_max

      texto_trayectoria <- if (is.null(res()$trayectoria_geom)) {
        "No se calculó la geometría sobre trayectoria dinámica."
      } else {
        "Se calculó además la geometría sobre una trayectoria dinámica compatible con rnas_neuron_dynamics."
      }

      paste0(
        "El análisis geométrico evalúa el paisaje local de pérdida en theta = (",
        paste(round(res()$theta, 4), collapse = ", "),
        "). La curvatura direccional calculada fue ",
        ifelse(is.na(curv), "no disponible", round(curv, 6)),
        ". El autovalor mínimo fue ",
        ifelse(is.na(lambda_min), "no disponible", round(lambda_min, 6)),
        " y el autovalor máximo fue ",
        ifelse(is.na(lambda_max), "no disponible", round(lambda_max, 6)),
        ". ",
        texto_trayectoria,
        " Estos indicadores permiten interpretar la forma local del paisaje de aprendizaje y su evolución dinámica."
      )
    })

    output$estructura <- renderPrint({
      req(res())
      str(res(), max.level = 3)
    })
  })
}


# ============================================================
# Auxiliares Capitulo 8
# ============================================================

convertir_objeto_a_tabla_cap08 <- function(x) {
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

  if (is.matrix(x)) {
    tab <- as.data.frame(x)
    tab$indicador <- paste0("fila_", seq_len(nrow(tab)))
    tab <- tab[, c("indicador", setdiff(names(tab), "indicador")), drop = FALSE]
    return(tab)
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


convertir_grid_a_tabla_cap08 <- function(grid) {
  if (is.data.frame(grid)) {
    return(grid)
  }

  if (is.matrix(grid)) {
    tab <- as.data.frame(as.table(grid))
    names(tab) <- c("w1_idx", "w2_idx", "loss")
    return(tab)
  }

  if (is.list(grid)) {
    if (!is.null(grid$grid) && is.data.frame(grid$grid)) {
      return(grid$grid)
    }

    if (!is.null(grid$loss_grid) && is.data.frame(grid$loss_grid)) {
      return(grid$loss_grid)
    }

    if (!is.null(grid$loss_matrix) && is.matrix(grid$loss_matrix)) {
      tab <- as.data.frame(as.table(grid$loss_matrix))
      names(tab) <- c("w1_idx", "w2_idx", "loss")
      return(tab)
    }

    return(convertir_objeto_a_tabla_cap08(grid))
  }

  data.frame(
    resultado = as.character(grid),
    stringsAsFactors = FALSE
  )
}


obtener_curvatura_cap08 <- function(curvatura) {
  val <- suppressWarnings(as.numeric(curvatura[1]))

  if (length(val) == 0L || !is.finite(val)) {
    return(NA_real_)
  }

  val
}


obtener_lambdas_cap08 <- function(autovalores) {
  if (is.null(autovalores)) {
    return(list(lambda_min = NA_real_, lambda_max = NA_real_))
  }

  if (is.numeric(autovalores)) {
    return(list(
      lambda_min = min(autovalores, na.rm = TRUE),
      lambda_max = max(autovalores, na.rm = TRUE)
    ))
  }

  if (is.data.frame(autovalores)) {
    nms <- names(autovalores)

    val_col <- nms[grepl("valor|lambda|autoval", nms, ignore.case = TRUE)][1]

    if (!is.na(val_col)) {
      vals <- suppressWarnings(as.numeric(autovalores[[val_col]]))

      return(list(
        lambda_min = min(vals, na.rm = TRUE),
        lambda_max = max(vals, na.rm = TRUE)
      ))
    }
  }

  if (is.list(autovalores)) {
    if (!is.null(autovalores$lambda_min) && !is.null(autovalores$lambda_max)) {
      return(list(
        lambda_min = as.numeric(autovalores$lambda_min[1]),
        lambda_max = as.numeric(autovalores$lambda_max[1])
      ))
    }

    vals <- suppressWarnings(as.numeric(unlist(autovalores)))

    vals <- vals[is.finite(vals)]

    if (length(vals) > 0L) {
      return(list(
        lambda_min = min(vals),
        lambda_max = max(vals)
      ))
    }
  }

  list(lambda_min = NA_real_, lambda_max = NA_real_)
}


plot_grid_loss_cap08 <- function(grid) {
  if (is.data.frame(grid)) {
    nms <- names(grid)

    w1_col <- nms[grepl("^w1$|w1", nms, ignore.case = TRUE)][1]
    w2_col <- nms[grepl("^w2$|w2", nms, ignore.case = TRUE)][1]
    loss_col <- nms[grepl("loss|perdida", nms, ignore.case = TRUE)][1]

    if (!is.na(w1_col) && !is.na(w2_col) && !is.na(loss_col)) {
      x <- grid[[w1_col]]
      y <- grid[[w2_col]]
      z <- grid[[loss_col]]

      plot(
        x,
        y,
        pch = 15,
        cex = 0.55,
        xlab = "w1",
        ylab = "w2",
        main = "Grid de pérdida"
      )

      if (length(z) > 0L && any(is.finite(z))) {
        points(
          x[which.min(z)],
          y[which.min(z)],
          pch = 19,
          cex = 1.4
        )
      }

      grid()
      return(invisible(NULL))
    }
  }

  if (is.matrix(grid)) {
    image(
      grid,
      xlab = "w1",
      ylab = "w2",
      main = "Paisaje de pérdida"
    )
    contour(grid, add = TRUE)
    return(invisible(NULL))
  }

  if (is.list(grid)) {
    if (!is.null(grid$loss_matrix) && is.matrix(grid$loss_matrix)) {
      image(
        grid$loss_matrix,
        xlab = "w1",
        ylab = "w2",
        main = "Paisaje de pérdida"
      )
      contour(grid$loss_matrix, add = TRUE)
      return(invisible(NULL))
    }

    if (!is.null(grid$grid) && is.data.frame(grid$grid)) {
      plot_grid_loss_cap08(grid$grid)
      return(invisible(NULL))
    }
  }

  plot.new()
  title("Grid de pérdida")
  text(0.5, 0.5, "Estructura de grid no reconocida.")
}


plot_trayectoria_geom_cap08 <- function(tg) {
  if (is.null(tg)) {
    plot.new()
    title("Trayectoria geométrica")
    text(0.5, 0.5, "No disponible.")
    return(invisible(NULL))
  }

  if (is.list(tg) && !is.data.frame(tg)) {
    if (!is.null(tg$trayectoria) && is.data.frame(tg$trayectoria)) {
      tg <- tg$trayectoria
    } else if (!is.null(tg$resultados) && is.data.frame(tg$resultados)) {
      tg <- tg$resultados
    } else {
      plot.new()
      title("Trayectoria geométrica")
      text(0.5, 0.5, "Estructura no reconocida.")
      return(invisible(NULL))
    }
  }

  if (!is.data.frame(tg) || nrow(tg) == 0L) {
    plot.new()
    title("Trayectoria geométrica")
    text(0.5, 0.5, "Sin registros.")
    return(invisible(NULL))
  }

  eje_x <- if ("iter" %in% names(tg)) {
    tg$iter
  } else if ("time" %in% names(tg)) {
    tg$time
  } else {
    seq_len(nrow(tg))
  }

  if (all(c("lambda_min", "lambda_max") %in% names(tg))) {
    plot(
      eje_x,
      tg$lambda_max,
      type = "l",
      lwd = 2,
      xlab = "Iteración",
      ylab = "Autovalor",
      main = "Autovalores en trayectoria"
    )
    lines(eje_x, tg$lambda_min, lwd = 2, lty = 2)
    legend(
      "topright",
      legend = c("lambda_max", "lambda_min"),
      lty = c(1, 2),
      lwd = 2,
      bty = "n"
    )
    grid()
    return(invisible(NULL))
  }

  if ("curvatura" %in% names(tg)) {
    plot(
      eje_x,
      tg$curvatura,
      type = "l",
      lwd = 2,
      xlab = "Iteración",
      ylab = "Curvatura",
      main = "Curvatura en trayectoria"
    )
    grid()
    return(invisible(NULL))
  }

  if ("loss" %in% names(tg)) {
    plot(
      eje_x,
      tg$loss,
      type = "l",
      lwd = 2,
      xlab = "Iteración",
      ylab = "Pérdida",
      main = "Pérdida en trayectoria"
    )
    grid()
    return(invisible(NULL))
  }

  plot.new()
  title("Trayectoria geométrica")
  text(0.5, 0.5, "Columnas geométricas no reconocidas.")
}
