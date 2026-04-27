# ============================================================
# Modulo Capitulo 11: Control optimo y meta-aprendizaje geometrico
# DSNeuralRNASLab
#
# Version corregida:
# - rnas_train_neuron_meta_geo(): opt_local, curvatura, lambda_max.
# - rnas_comparar_meta_politicas(): opt_local, curvatura, lambda_max.
# - rnas_eta_policy_geo(): curvatura, lambda_max, hibrida.
# - Si metodo = opt_local, modo_geo = hibrida solo para referencia geometrica.
# - Maneja salidas tipo numerico, lista o data.frame.
# ============================================================

mod_cap11_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 11: Control óptimo y meta-aprendizaje geométrico"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite explorar el control óptimo local de la tasa de aprendizaje y el meta-aprendizaje geométrico en una neurona RNAS. El usuario puede modificar eta inicial, rango de búsqueda, método meta-geométrico, activación, pesos iniciales, parámetros de penalización y frecuencia de evaluación geométrica.")
    ),

    fluidRow(
      column(
        4,

        h4("Datos y neurona"),

        selectInput(
          ns("activation"),
          "Activación",
          choices = c("tanh", "sigmoid"),
          selected = "tanh"
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

        h4("Control de eta"),

        selectInput(
          ns("metodo"),
          "Método meta-geométrico",
          choices = c(
            "Óptimo local" = "opt_local",
            "Curvatura" = "curvatura",
            "Lambda máxima" = "lambda_max"
          ),
          selected = "opt_local"
        ),

        numericInput(
          ns("eta0"),
          "Eta inicial",
          value = 0.1,
          min = 0.0001,
          max = 1,
          step = 0.005
        ),

        numericInput(
          ns("eta_min"),
          "Eta mínima",
          value = 0.01,
          min = 0.00001,
          max = 1,
          step = 0.001
        ),

        numericInput(
          ns("eta_max"),
          "Eta máxima",
          value = 0.1,
          min = 0.0001,
          max = 2,
          step = 0.005
        ),

        numericInput(
          ns("n_eta_grid"),
          "Número de eta candidatos",
          value = 10,
          min = 3,
          max = 50,
          step = 1
        ),

        numericInput(
          ns("T_iter"),
          "Número de iteraciones",
          value = 100,
          min = 5,
          max = 1000,
          step = 5
        ),

        hr(),

        h4("Parámetros geométricos"),

        numericInput(
          ns("alpha"),
          "Alpha",
          value = 1,
          min = 0,
          max = 100,
          step = 0.1
        ),

        numericInput(
          ns("beta"),
          "Beta",
          value = 0,
          min = 0,
          max = 100,
          step = 0.1
        ),

        numericInput(
          ns("h"),
          "Paso Hessiano h",
          value = 1e-4,
          min = 1e-8,
          step = 1e-4
        ),

        numericInput(
          ns("evaluar_geo_cada"),
          "Evaluar geometría cada",
          value = 1,
          min = 1,
          max = 100,
          step = 1
        ),

        hr(),

        checkboxInput(
          ns("comparar_politicas"),
          "Comparar políticas meta",
          value = TRUE
        ),

        actionButton(
          ns("run"),
          "Ejecutar meta-control",
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
            "Costo local",
            br(),
            DT::DTOutput(ns("tabla_costo"))
          ),

          tabPanel(
            "Resumen meta",
            br(),
            DT::DTOutput(ns("tabla_resumen"))
          ),

          tabPanel(
            "Trayectoria",
            br(),
            DT::DTOutput(ns("tabla_trayectoria"))
          ),

          tabPanel(
            "Comparación",
            br(),
            DT::DTOutput(ns("tabla_comparacion"))
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


mod_cap11_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      datos <- generar_datos_neurona_base()

      X <- datos$X
      y <- datos$y
      theta0 <- c(input$w1, input$w2, input$b)

      eta_grid <- seq(
        input$eta_min,
        input$eta_max,
        length.out = as.integer(input$n_eta_grid)
      )

      grad0 <- DSNeuralRNAS::rnas_grad_neuron(
        X = X,
        y = y,
        w = theta0[1:2],
        b = theta0[3],
        activation = input$activation
      )

      grad_vec <- c(grad0$grad_w, grad0$grad_b)

      costo_grid <- do.call(
        rbind,
        lapply(eta_grid, function(eta_i) {
          costo_i <- rnas_eta_costo_local(
            X = X,
            y = y,
            theta = theta0,
            grad = grad_vec,
            eta = eta_i,
            activation = input$activation,
            alpha = input$alpha,
            beta = input$beta
          )

          data.frame(
            eta = eta_i,
            costo = extraer_costo_local_cap11(costo_i),
            detalle = resumen_estructura_cap11(costo_i),
            stringsAsFactors = FALSE
          )
        })
      )

      eta_opt <- DSNeuralRNAS::rnas_eta_opt_local(
        X = X,
        y = y,
        theta = theta0,
        grad = grad_vec,
        eta_grid = eta_grid,
        activation = input$activation,
        alpha = input$alpha,
        beta = input$beta
      )

      H <- DSNeuralRNAS::rnas_hessian_num_neuron(
        X = X,
        y = y,
        theta = theta0,
        activation = input$activation,
        h = input$h
      )

      aut <- DSNeuralRNAS::rnas_autovalores_hessian(H)
      lambdas <- extraer_lambdas_cap11(aut)

      curv <- DSNeuralRNAS::rnas_curvatura_direccional(
        H = H,
        v = grad_vec
      )

      # rnas_eta_policy_geo() solo acepta:
      # "curvatura", "lambda_max" o "hibrida".
      # Si el metodo principal es "opt_local", se usa "hibrida"
      # solo como eta geometrica de referencia.
      modo_geo <- if (input$metodo %in% c("curvatura", "lambda_max")) {
        input$metodo
      } else {
        "hibrida"
      }

      eta_geo <- DSNeuralRNAS::rnas_eta_policy_geo(
        eta0 = input$eta0,
        kappa = extraer_costo_local_cap11(curv),
        lambda_max = lambdas$lambda_max,
        modo = modo_geo,
        alpha = input$alpha,
        eta_reg = NULL,
        eta_min = input$eta_min,
        eta_max = input$eta_max
      )

      modelo <- DSNeuralRNAS::rnas_train_neuron_meta_geo(
        X = X,
        y = y,
        theta0 = theta0,
        eta0 = input$eta0,
        metodo = input$metodo,
        eta_grid = eta_grid,
        eta_min = input$eta_min,
        eta_max = input$eta_max,
        T = as.integer(input$T_iter),
        activation = input$activation,
        alpha = input$alpha,
        beta = input$beta,
        h = input$h,
        evaluar_geo_cada = as.integer(input$evaluar_geo_cada)
      )

      resumen <- DSNeuralRNAS::rnas_resumen_meta_geo(
        object = modelo,
        nombre_politica = input$metodo
      )

      comparacion <- NULL

      if (isTRUE(input$comparar_politicas)) {
        politicas <- list(
          opt_local = list(
            metodo = "opt_local",
            alpha = input$alpha,
            beta = input$beta
          ),
          curvatura = list(
            metodo = "curvatura",
            alpha = input$alpha
          ),
          lambda_max = list(
            metodo = "lambda_max",
            alpha = input$alpha
          )
        )

        comparacion <- tryCatch(
          {
            rnas_comparar_meta_politicas(
              X = X,
              y = y,
              theta0 = theta0,
              politicas = politicas,
              eta0 = input$eta0,
              eta_min = input$eta_min,
              eta_max = input$eta_max,
              T = as.integer(input$T_iter),
              activation = input$activation
            )
          },
          error = function(e) {
            list(error = conditionMessage(e))
          }
        )
      }

      list(
        datos = datos,
        theta0 = theta0,
        grad0 = grad_vec,
        eta_grid = eta_grid,
        costo_grid = costo_grid,
        eta_opt = eta_opt,
        eta_opt_valor = extraer_eta_cap11(eta_opt),
        H = H,
        autovalores = aut,
        lambdas = lambdas,
        curvatura = curv,
        curvatura_valor = extraer_costo_local_cap11(curv),
        modo_geo = modo_geo,
        eta_geo = eta_geo,
        eta_geo_valor = extraer_eta_cap11(eta_geo),
        modelo = modelo,
        resumen = resumen,
        comparacion = comparacion,
        configuracion = list(
          metodo = input$metodo,
          modo_geo = modo_geo,
          eta0 = input$eta0,
          eta_min = input$eta_min,
          eta_max = input$eta_max,
          T = as.integer(input$T_iter),
          activation = input$activation,
          alpha = input$alpha,
          beta = input$beta,
          h = input$h,
          evaluar_geo_cada = as.integer(input$evaluar_geo_cada)
        )
      )
    })

    output$codigo <- renderUI({
      codigo <- paste0(
        "datos <- generar_datos_neurona_base()\n",
        "X <- datos$X\n",
        "y <- datos$y\n",
        "theta0 <- c(", input$w1, ", ", input$w2, ", ", input$b, ")\n\n",
        "eta_grid <- seq(", input$eta_min, ", ", input$eta_max,
        ", length.out = ", input$n_eta_grid, ")\n\n",
        "modelo <- rnas_train_neuron_meta_geo(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  theta0 = theta0,\n",
        "  eta0 = ", input$eta0, ",\n",
        "  metodo = \"", input$metodo, "\",\n",
        "  eta_grid = eta_grid,\n",
        "  eta_min = ", input$eta_min, ",\n",
        "  eta_max = ", input$eta_max, ",\n",
        "  T = ", input$T_iter, ",\n",
        "  activation = \"", input$activation, "\",\n",
        "  alpha = ", input$alpha, ",\n",
        "  beta = ", input$beta, ",\n",
        "  h = ", input$h, ",\n",
        "  evaluar_geo_cada = ", input$evaluar_geo_cada, "\n",
        ")\n\n",
        "resumen <- rnas_resumen_meta_geo(\n",
        "  object = modelo,\n",
        "  nombre_politica = \"", input$metodo, "\"\n",
        ")\n\n",
        "# Comparación válida:\n",
        "politicas <- list(\n",
        "  opt_local = list(metodo = \"opt_local\"),\n",
        "  curvatura = list(metodo = \"curvatura\"),\n",
        "  lambda_max = list(metodo = \"lambda_max\")\n",
        ")\n\n",
        "# Nota: rnas_eta_policy_geo() solo acepta modo =\n",
        "# 'curvatura', 'lambda_max' o 'hibrida'.\n",
        "# Si metodo = 'opt_local', la app usa modo_geo = 'hibrida'\n",
        "# solo para calcular una eta geometrica de referencia."
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

    output$tabla_costo <- DT::renderDT({
      req(res())

      eta_opt_tab <- convertir_objeto_a_tabla_cap11(res()$eta_opt)

      eta_opt_txt <- paste(
        paste0(eta_opt_tab$indicador, "=", eta_opt_tab$valor),
        collapse = "; "
      )

      tab <- res()$costo_grid
      tab$eta_opt_local <- eta_opt_txt
      tab$eta_opt_valor <- res()$eta_opt_valor
      tab$modo_geo <- res()$modo_geo
      tab$eta_geo_politica <- res()$eta_geo_valor

      tabla_dt(tab)
    })

    output$tabla_resumen <- DT::renderDT({
      req(res())

      tab <- convertir_objeto_a_tabla_cap11(res()$resumen)

      tabla_dt(tab)
    })

    output$tabla_trayectoria <- DT::renderDT({
      req(res())

      modelo <- res()$modelo
      tray <- extraer_trayectoria_cap11(modelo)

      if (is.null(tray) || !is.data.frame(tray) || nrow(tray) == 0L) {
        tab <- data.frame(
          mensaje = "El objeto de meta-aprendizaje no contiene trayectoria reconocible.",
          stringsAsFactors = FALSE
        )
      } else {
        columnas <- intersect(
          c(
            "iter", "time", "loss", "grad_norm", "eta",
            "kappa", "curvatura", "lambda_max", "lambda_min",
            "metodo", "accion_eta", "speed"
          ),
          names(tray)
        )

        if (length(columnas) == 0L) {
          tab <- utils::head(tray, 40)
        } else {
          tab <- utils::head(tray[, columnas, drop = FALSE], 40)
        }
      }

      tabla_dt(tab)
    })

    output$tabla_comparacion <- DT::renderDT({
      req(res())

      comp <- res()$comparacion

      if (is.null(comp)) {
        tab <- data.frame(
          mensaje = "Comparación de políticas no ejecutada.",
          stringsAsFactors = FALSE
        )
      } else {
        tab <- convertir_comparacion_cap11(comp)
      }

      tabla_dt(tab)
    })

    output$plot <- renderPlot({
      req(res())

      modelo <- res()$modelo
      tray <- extraer_trayectoria_cap11(modelo)
      costo <- res()$costo_grid
      comp <- res()$comparacion

      op <- par(mfrow = c(2, 2))
      on.exit(par(op), add = TRUE)

      if (is.data.frame(tray) && nrow(tray) > 0L && "loss" %in% names(tray)) {
        eje_x <- if ("iter" %in% names(tray)) tray$iter else seq_len(nrow(tray))

        plot(
          eje_x,
          tray$loss,
          type = "l",
          lwd = 2,
          xlab = "Iteración",
          ylab = "Pérdida",
          main = "Meta-geo: pérdida"
        )
        grid()
      } else {
        plot.new()
        title("Meta-geo: pérdida")
        text(0.5, 0.5, "No disponible.")
      }

      if (is.data.frame(tray) && nrow(tray) > 0L && "eta" %in% names(tray)) {
        eje_x <- if ("iter" %in% names(tray)) tray$iter else seq_len(nrow(tray))

        plot(
          eje_x,
          tray$eta,
          type = "l",
          lwd = 2,
          xlab = "Iteración",
          ylab = "Eta",
          main = "Meta-geo: eta"
        )
        grid()
      } else {
        plot.new()
        title("Meta-geo: eta")
        text(0.5, 0.5, "No disponible.")
      }

      if (is.data.frame(costo) && all(c("eta", "costo") %in% names(costo))) {
        plot(
          costo$eta,
          costo$costo,
          type = "b",
          pch = 19,
          lwd = 2,
          xlab = "Eta candidato",
          ylab = "Costo local",
          main = "Costo local por eta"
        )

        if (is.finite(res()$eta_opt_valor)) {
          abline(v = res()$eta_opt_valor, lty = 2)
        }

        grid()
      } else {
        plot.new()
        title("Costo local")
        text(0.5, 0.5, "No disponible.")
      }

      plot_comparacion_cap11(comp)
    })

    output$interpretacion <- renderText({
      req(res())

      tray <- extraer_trayectoria_cap11(res()$modelo)

      loss_ini <- obtener_valor_tray_cap11(tray, "loss", "inicial")
      loss_fin <- obtener_valor_tray_cap11(tray, "loss", "final")
      eta_ini <- obtener_valor_tray_cap11(tray, "eta", "inicial")
      eta_fin <- obtener_valor_tray_cap11(tray, "eta", "final")

      lambdas <- res()$lambdas

      paste0(
        "El meta-aprendizaje geométrico se ejecutó con método '",
        input$metodo,
        "', eta inicial = ",
        input$eta0,
        " y rango [",
        input$eta_min,
        ", ",
        input$eta_max,
        "]. La pérdida pasó de ",
        ifelse(is.na(loss_ini), "no disponible", round(loss_ini, 6)),
        " a ",
        ifelse(is.na(loss_fin), "no disponible", round(loss_fin, 6)),
        ". Eta pasó de ",
        ifelse(is.na(eta_ini), "no disponible", round(eta_ini, 6)),
        " a ",
        ifelse(is.na(eta_fin), "no disponible", round(eta_fin, 6)),
        ". En el punto inicial, lambda_max = ",
        ifelse(is.na(lambdas$lambda_max), "no disponible", round(lambdas$lambda_max, 6)),
        " y la curvatura direccional fue ",
        ifelse(is.na(res()$curvatura_valor), "no disponible", round(res()$curvatura_valor, 6)),
        ". La eta óptima local estimada fue ",
        ifelse(is.na(res()$eta_opt_valor), "no disponible", round(res()$eta_opt_valor, 6)),
        ". Para eta geométrica de referencia se usó modo_geo = '",
        res()$modo_geo,
        "'."
      )
    })

    output$estructura <- renderPrint({
      req(res())
      str(res(), max.level = 3)
    })
  })
}


# ============================================================
# Auxiliares Capitulo 11
# ============================================================

resumen_estructura_cap11 <- function(x) {
  if (is.null(x)) {
    return("NULL")
  }

  if (is.numeric(x) && length(x) == 1L) {
    return("numeric[1]")
  }

  if (is.numeric(x)) {
    return(paste0("numeric[", length(x), "]"))
  }

  if (is.data.frame(x)) {
    return(paste0("data.frame[", nrow(x), "x", ncol(x), "]"))
  }

  if (is.matrix(x)) {
    return(paste0("matrix[", nrow(x), "x", ncol(x), "]"))
  }

  if (is.list(x)) {
    nms <- names(x)
    if (is.null(nms)) {
      nms <- paste0("elemento_", seq_along(x))
    }
    return(paste0("list{", paste(nms, collapse = ", "), "}"))
  }

  class(x)[1]
}


extraer_costo_local_cap11 <- function(x) {
  if (is.null(x)) {
    return(NA_real_)
  }

  if (is.numeric(x) && length(x) >= 1L) {
    val <- as.numeric(x[1])
    if (is.finite(val)) return(val)
  }

  if (is.data.frame(x)) {
    nms <- names(x)
    col <- nms[grepl("costo|cost|valor|loss|objetivo|J|criterio", nms, ignore.case = TRUE)][1]

    if (!is.na(col)) {
      val <- suppressWarnings(as.numeric(x[[col]][1]))
      if (is.finite(val)) return(val)
    }
  }

  if (is.list(x)) {
    candidatos <- c(
      "costo",
      "cost",
      "valor",
      "valor_costo",
      "loss",
      "loss_eta",
      "objetivo",
      "J",
      "criterio"
    )

    for (nm in candidatos) {
      if (!is.null(x[[nm]])) {
        val <- suppressWarnings(as.numeric(x[[nm]][1]))
        if (is.finite(val)) return(val)
      }
    }

    vals <- suppressWarnings(
      as.numeric(unlist(x, recursive = TRUE, use.names = FALSE))
    )

    vals <- vals[is.finite(vals)]

    if (length(vals) > 0L) {
      return(vals[1])
    }
  }

  NA_real_
}


extraer_eta_cap11 <- function(x) {
  if (is.null(x)) {
    return(NA_real_)
  }

  if (is.numeric(x) && length(x) >= 1L) {
    val <- as.numeric(x[1])
    if (is.finite(val)) return(val)
  }

  if (is.data.frame(x)) {
    nms <- names(x)
    col <- nms[grepl("^eta$|eta_opt|eta_star|eta_geo|mejor_eta|eta_optima", nms, ignore.case = TRUE)][1]

    if (!is.na(col)) {
      val <- suppressWarnings(as.numeric(x[[col]][1]))
      if (is.finite(val)) return(val)
    }
  }

  if (is.list(x)) {
    candidatos <- c(
      "eta",
      "eta_opt",
      "eta_star",
      "eta_geo",
      "mejor_eta",
      "eta_mejor",
      "eta_optima"
    )

    for (nm in candidatos) {
      if (!is.null(x[[nm]])) {
        val <- suppressWarnings(as.numeric(x[[nm]][1]))
        if (is.finite(val)) return(val)
      }
    }

    vals <- suppressWarnings(
      as.numeric(unlist(x, recursive = TRUE, use.names = FALSE))
    )

    vals <- vals[is.finite(vals)]

    if (length(vals) > 0L) {
      return(vals[1])
    }
  }

  NA_real_
}


convertir_objeto_a_tabla_cap11 <- function(x) {
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


extraer_trayectoria_cap11 <- function(object) {
  if (is.null(object)) {
    return(data.frame())
  }

  if (!is.null(object$trayectoria) && is.data.frame(object$trayectoria)) {
    return(object$trayectoria)
  }

  if (!is.null(object$historial) && is.data.frame(object$historial)) {
    return(object$historial)
  }

  if (!is.null(object$history) && is.data.frame(object$history)) {
    return(object$history)
  }

  if (!is.null(object$resultados$trayectoria) &&
      is.data.frame(object$resultados$trayectoria)) {
    return(object$resultados$trayectoria)
  }

  data.frame()
}


extraer_lambdas_cap11 <- function(autovalores) {
  if (is.null(autovalores)) {
    return(list(lambda_min = NA_real_, lambda_max = NA_real_))
  }

  if (is.numeric(autovalores)) {
    vals <- autovalores[is.finite(autovalores)]

    if (length(vals) == 0L) {
      return(list(lambda_min = NA_real_, lambda_max = NA_real_))
    }

    return(list(
      lambda_min = min(vals),
      lambda_max = max(vals)
    ))
  }

  if (is.data.frame(autovalores)) {
    nms <- names(autovalores)
    val_col <- nms[grepl("valor|lambda|autoval", nms, ignore.case = TRUE)][1]

    if (!is.na(val_col)) {
      vals <- suppressWarnings(as.numeric(autovalores[[val_col]]))
      vals <- vals[is.finite(vals)]

      if (length(vals) > 0L) {
        return(list(
          lambda_min = min(vals),
          lambda_max = max(vals)
        ))
      }
    }
  }

  if (is.list(autovalores)) {
    if (!is.null(autovalores$lambda_min) &&
        !is.null(autovalores$lambda_max)) {
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


obtener_valor_tray_cap11 <- function(trayectoria, columna, posicion = c("inicial", "final")) {
  posicion <- match.arg(posicion)

  if (!is.data.frame(trayectoria) ||
      nrow(trayectoria) == 0L ||
      !(columna %in% names(trayectoria))) {
    return(NA_real_)
  }

  if (posicion == "inicial") {
    return(suppressWarnings(as.numeric(trayectoria[[columna]][1])))
  }

  suppressWarnings(as.numeric(trayectoria[[columna]][nrow(trayectoria)]))
}


convertir_comparacion_cap11 <- function(comp) {
  if (is.null(comp)) {
    return(data.frame(
      mensaje = "Comparacion no disponible.",
      stringsAsFactors = FALSE
    ))
  }

  if (is.data.frame(comp)) {
    return(comp)
  }

  if (is.list(comp)) {
    if (!is.null(comp$error)) {
      return(data.frame(
        error = comp$error,
        stringsAsFactors = FALSE
      ))
    }

    if (!is.null(comp$resumen) && is.data.frame(comp$resumen)) {
      return(comp$resumen)
    }

    if (!is.null(comp$comparacion) && is.data.frame(comp$comparacion)) {
      return(comp$comparacion)
    }

    if (!is.null(comp$metricas) && is.data.frame(comp$metricas)) {
      return(comp$metricas)
    }

    return(convertir_objeto_a_tabla_cap11(comp))
  }

  data.frame(
    resultado = as.character(comp),
    stringsAsFactors = FALSE
  )
}


plot_comparacion_cap11 <- function(comp) {
  tab <- convertir_comparacion_cap11(comp)

  if (is.data.frame(tab) && "error" %in% names(tab)) {
    plot.new()
    title("Comparación de políticas")
    text(0.5, 0.5, "No disponible.")
    return(invisible(NULL))
  }

  if (is.data.frame(tab)) {
    nms <- names(tab)

    pol_col <- nms[grepl("politica|metodo|nombre", nms, ignore.case = TRUE)][1]
    loss_col <- nms[grepl("loss_final|perdida_final|loss", nms, ignore.case = TRUE)][1]

    if (!is.na(pol_col) && !is.na(loss_col)) {
      vals <- suppressWarnings(as.numeric(tab[[loss_col]]))
      names(vals) <- as.character(tab[[pol_col]])

      keep <- is.finite(vals)
      vals <- vals[keep]

      if (length(vals) > 0L) {
        barplot(
          vals,
          las = 2,
          ylab = "Pérdida final",
          main = "Comparación de políticas"
        )
        grid()
        return(invisible(NULL))
      }
    }
  }

  plot.new()
  title("Comparación de políticas")
  text(0.5, 0.5, "Estructura no reconocida.")
}
