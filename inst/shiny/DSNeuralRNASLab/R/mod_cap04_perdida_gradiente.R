# ============================================================
# Modulo Capitulo 4: Perdida, gradiente y verificacion
# DSNeuralRNASLab
# ============================================================

mod_cap04_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 4: Pérdida, gradiente y verificación"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite calcular la pérdida MSE, el gradiente analítico, el gradiente numérico y la verificación de consistencia entre ambos gradientes para una neurona RNAS.")
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

        numericInput(
          ns("h"),
          "Paso numérico h",
          value = 1e-5,
          min = 1e-8,
          step = 1e-5
        ),

        numericInput(
          ns("tol"),
          "Tolerancia",
          value = 1e-5,
          min = 1e-8,
          step = 1e-5
        ),

        actionButton(
          ns("run"),
          "Ejecutar verificación",
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
            "Pérdida",
            br(),
            DT::DTOutput(ns("tabla_loss"))
          ),

          tabPanel(
            "Gradientes",
            br(),
            DT::DTOutput(ns("tabla_grad"))
          ),

          tabPanel(
            "Verificación",
            br(),
            DT::DTOutput(ns("tabla_check"))
          ),

          tabPanel(
            "Interpretación",
            br(),
            verbatimTextOutput(ns("interpretacion"))
          )
        )
      )
    )
  )
}

mod_cap04_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      datos <- generar_datos_neurona_base()

      X <- datos$X
      y <- datos$y

      w <- c(input$w1, input$w2)
      b <- input$b

      loss <- DSNeuralRNAS::rnas_loss_mse_neuron(
        X = X,
        y = y,
        w = w,
        b = b,
        activation = input$activation
      )

      grad <- DSNeuralRNAS::rnas_grad_neuron(
        X = X,
        y = y,
        w = w,
        b = b,
        activation = input$activation
      )

      grad_num <- DSNeuralRNAS::rnas_grad_num_neuron(
        X = X,
        y = y,
        w = w,
        b = b,
        activation = input$activation,
        h = input$h
      )

      check <- DSNeuralRNAS::rnas_grad_check_neuron(
        X = X,
        y = y,
        w = w,
        b = b,
        activation = input$activation,
        h = input$h,
        tol = input$tol
      )

      list(
        datos = datos,
        w = w,
        b = b,
        loss = loss,
        grad = grad,
        grad_num = grad_num,
        check = check
      )
    })

    output$codigo <- renderUI({
      codigo <- paste0(
        "datos <- generar_datos_neurona_base()\n",
        "X <- datos$X\n",
        "y <- datos$y\n\n",
        "w <- c(", input$w1, ", ", input$w2, ")\n",
        "b <- ", input$b, "\n\n",
        "loss <- rnas_loss_mse_neuron(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  w = w,\n",
        "  b = b,\n",
        "  activation = \"", input$activation, "\"\n",
        ")\n\n",
        "grad <- rnas_grad_neuron(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  w = w,\n",
        "  b = b,\n",
        "  activation = \"", input$activation, "\"\n",
        ")\n\n",
        "grad_num <- rnas_grad_num_neuron(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  w = w,\n",
        "  b = b,\n",
        "  activation = \"", input$activation, "\",\n",
        "  h = ", input$h, "\n",
        ")\n\n",
        "check <- rnas_grad_check_neuron(\n",
        "  X = X,\n",
        "  y = y,\n",
        "  w = w,\n",
        "  b = b,\n",
        "  activation = \"", input$activation, "\",\n",
        "  h = ", input$h, ",\n",
        "  tol = ", input$tol, "\n",
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

    output$tabla_loss <- DT::renderDT({
      req(res())

      tab <- data.frame(
        indicador = c("loss_mse"),
        valor = c(res()$loss),
        stringsAsFactors = FALSE
      )

      tabla_dt(tab)
    })

    output$tabla_grad <- DT::renderDT({
      req(res())

      g <- res()$grad
      gn <- res()$grad_num

      grad_analitico <- c(g$grad_w, g$grad_b)

      if (is.data.frame(gn) && all(c("componente", "valor") %in% names(gn))) {
        grad_numerico <- gn$valor
      } else if (is.data.frame(gn) && "grad" %in% names(gn)) {
        grad_numerico <- gn$grad
      } else if (is.list(gn) && !is.null(gn$grad)) {
        grad_numerico <- gn$grad
      } else if (is.list(gn) && !is.null(gn$grad_w) && !is.null(gn$grad_b)) {
        grad_numerico <- c(gn$grad_w, gn$grad_b)
      } else if (is.numeric(gn)) {
        grad_numerico <- gn
      } else {
        stop("No se reconocio la estructura devuelta por rnas_grad_num_neuron().")
      }

      grad_numerico <- as.numeric(grad_numerico)

      if (length(grad_numerico) != length(grad_analitico)) {
        stop("El gradiente numerico no tiene la misma longitud que el gradiente analitico.")
      }

      tab <- data.frame(
        componente = c("w1", "w2", "b"),
        grad_analitico = grad_analitico,
        grad_numerico = grad_numerico,
        diferencia = grad_analitico - grad_numerico,
        stringsAsFactors = FALSE
      )

      tabla_dt(tab)
    })

    output$tabla_check <- DT::renderDT({
      req(res())

      chk <- res()$check

      if (is.data.frame(chk)) {

        tab <- chk

      } else if (is.list(chk)) {

        tab <- do.call(
          rbind,
          lapply(names(chk), function(nm) {
            valor <- chk[[nm]]

            if (is.null(valor)) {
              valor_txt <- NA_character_
            } else if (length(valor) == 1L) {
              valor_txt <- as.character(valor)
            } else {
              valor_txt <- paste(as.character(valor), collapse = ", ")
            }

            data.frame(
              indicador = nm,
              valor = valor_txt,
              stringsAsFactors = FALSE
            )
          })
        )

        rownames(tab) <- NULL

      } else {

        tab <- data.frame(
          indicador = "resultado",
          valor = as.character(chk),
          stringsAsFactors = FALSE
        )
      }

      tabla_dt(tab)
    })

    output$interpretacion <- renderText({
      req(res())

      chk <- res()$check

      if (is.data.frame(chk) && all(c("indicador", "valor") %in% names(chk))) {
        diff_abs <- chk$valor[chk$indicador == "diff_abs"]
        verificado <- chk$valor[chk$indicador == "verificado"]
      } else if (is.list(chk)) {
        diff_abs <- chk$diff_abs
        verificado <- chk$verificado
      } else {
        diff_abs <- NA
        verificado <- NA
      }

      if (length(diff_abs) == 0) diff_abs <- NA
      if (length(verificado) == 0) verificado <- NA

      paste0(
        "La pérdida MSE obtenida fue ",
        round(res()$loss, 8),
        ". La verificación compara el gradiente analítico con el gradiente numérico. ",
        "La diferencia absoluta reportada fue ",
        ifelse(is.na(diff_abs), "no disponible", round(as.numeric(diff_abs), 10)),
        ". El indicador de verificación fue ",
        ifelse(is.na(verificado), "no disponible", as.character(verificado)),
        "."
      )
    })
  })
}
