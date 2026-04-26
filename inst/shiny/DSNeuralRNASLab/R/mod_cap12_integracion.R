# ============================================================
# Modulo Capitulo 12: Integracion STD2, SimuDS y FNL
# DSNeuralRNASLab
# ============================================================

mod_cap12_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 12: Integración con STD2, SimuDS y FNL"),

    div(
      class = "bloque-explicacion",
      p("Este módulo permite ejecutar integraciones controladas entre RNAS y tres marcos del proyecto: señales temporales STD2, trayectorias simuladas SimuDS y formulación funcional FNL.")
    ),

    fluidRow(
      column(
        4,

        selectInput(
          ns("tipo_integracion"),
          "Tipo de integración",
          choices = c(
            "STD2-RNAS" = "std2",
            "SimuDS-RNAS" = "simuds",
            "FNL-RNAS" = "fnl"
          ),
          selected = "std2"
        ),

        selectInput(
          ns("activation"),
          "Activación",
          choices = c("tanh", "sigmoid"),
          selected = "tanh"
        ),

        numericInput(ns("eta0"), "Eta inicial", value = 0.05, min = 0.001, max = 1, step = 0.005),
        numericInput(ns("T_iter"), "Número de iteraciones", value = 100, min = 5, max = 1000, step = 5),
        numericInput(ns("horizonte"), "Horizonte", value = 1, min = 1, max = 10, step = 1),
        numericInput(ns("seed"), "Semilla", value = 123, min = 1, step = 1),

        conditionalPanel(
          condition = sprintf("input['%s'] == 'std2'", ns("tipo_integracion")),
          numericInput(ns("n_std2"), "Observaciones STD2", value = 60, min = 15, max = 500, step = 5),
          checkboxGroupInput(
            ns("features_std2"),
            "Señales STD2",
            choices = c("C", "x", "e", "gap", "speed"),
            selected = c("C", "x", "e", "gap", "speed")
          )
        ),

        conditionalPanel(
          condition = sprintf("input['%s'] == 'simuds'", ns("tipo_integracion")),
          numericInput(ns("n_simuds"), "Estados por escenario", value = 40, min = 15, max = 500, step = 5),
          selectInput(
            ns("escenario"),
            "Escenario",
            choices = c("Todos" = "todos", "Base" = "base", "Shock" = "shock", "Recuperación" = "recuperacion"),
            selected = "todos"
          )
        ),

        conditionalPanel(
          condition = sprintf("input['%s'] == 'fnl'", ns("tipo_integracion")),
          numericInput(ns("n_fnl"), "Observaciones FNL", value = 60, min = 10, max = 500, step = 5),
          numericInput(ns("w1"), "theta w1", value = 0.1, step = 0.05),
          numericInput(ns("w2"), "theta w2", value = 0.1, step = 0.05),
          numericInput(ns("b"), "theta b", value = 0.0, step = 0.05),
          numericInput(ns("c_norma"), "Cota norma theta", value = 10, min = 0.1, step = 0.5),
          numericInput(ns("c_w1"), "Cota w1", value = 3, min = 0.1, step = 0.5)
        ),

        actionButton(ns("run"), "Ejecutar integración", class = "btn-primary")
      ),

      column(
        8,
        tabsetPanel(
          tabPanel("Código", br(), uiOutput(ns("codigo"))),
          tabPanel("Datos preparados", br(), DT::DTOutput(ns("tabla_datos"))),
          tabPanel("Resumen", br(), DT::DTOutput(ns("tabla_resumen"))),
          tabPanel("Trayectoria / diagnóstico", br(), DT::DTOutput(ns("tabla_trayectoria"))),
          tabPanel("Gráficas", br(), plotOutput(ns("plot"), height = "440px")),
          tabPanel("Interpretación", br(), verbatimTextOutput(ns("interpretacion")))
        )
      )
    )
  )
}

mod_cap12_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      tipo <- input$tipo_integracion
      activation <- input$activation
      eta0 <- input$eta0
      T_iter <- as.integer(input$T_iter)
      horizonte <- as.integer(input$horizonte)
      seed <- as.integer(input$seed)

      if (tipo == "std2") {
        datos <- generar_datos_std2_app(n = as.integer(input$n_std2), seed = seed)

        features <- input$features_std2
        if (length(features) < 1) {
          stop("Debe seleccionar al menos una señal STD2.")
        }

        theta0 <- rep(0, length(features) + 1L)

        modelo <- rnas_integrar_std2(
          datos_std2 = datos,
          target = "y",
          features = features,
          horizonte = horizonte,
          theta0 = theta0,
          eta0 = eta0,
          T = T_iter,
          activation = activation
        )

        return(list(
          tipo = tipo,
          datos = datos,
          resultado = modelo,
          resumen = rnas_resumen_integracion(modelo)
        ))
      }

      if (tipo == "simuds") {
        tray <- generar_trayectorias_simuds_app(
          n = as.integer(input$n_simuds),
          escenario_sel = input$escenario
        )

        modelo <- rnas_integrar_simuds(
          trayectorias_simuds = tray,
          estado_cols = "s",
          target_estado = "s",
          escenario_col = "escenario",
          tiempo_col = "t",
          horizonte = horizonte,
          theta0 = c(0, 0),
          eta0 = eta0,
          T = T_iter,
          activation = activation
        )

        return(list(
          tipo = tipo,
          datos = tray,
          resultado = modelo,
          resumen = rnas_resumen_integracion(modelo)
        ))
      }

      if (tipo == "fnl") {
        set.seed(seed)

        n <- as.integer(input$n_fnl)
        C <- tanh(seq(-1, 1, length.out = n))
        x <- sin(seq(0, 4, length.out = n)) / 5
        y <- C + 0.2 * x + stats::rnorm(n, sd = 0.02)
        X <- cbind(C = C, x = x)

        restricciones <- list(
          norma_theta = function(theta) sum(theta^2) - input$c_norma,
          cota_w1 = function(theta) theta[1] - input$c_w1
        )

        fnl <- rnas_formular_fnl(
          X = X,
          y = y,
          activation = activation,
          restricciones = restricciones
        )

        theta_eval <- c(input$w1, input$w2, input$b)

        valor_objetivo <- fnl$objetivo(theta_eval)
        valores_restricciones <- fnl$evaluar_restricciones(theta_eval)
        factible <- fnl$es_factible(theta_eval)

        diagnostico <- data.frame(
          indicador = c("valor_objetivo", names(valores_restricciones), "factible"),
          valor = c(valor_objetivo, valores_restricciones, as.numeric(factible)),
          stringsAsFactors = FALSE
        )

        resumen <- data.frame(
          integracion = "FNL-RNAS",
          n_obs = n,
          n_features = ncol(X),
          horizonte = NA,
          loss_inicial = NA,
          loss_final = valor_objetivo,
          reduccion_rel = NA,
          factible = factible,
          stringsAsFactors = FALSE
        )

        return(list(
          tipo = tipo,
          datos = data.frame(C = C, x = x, y = y),
          resultado = fnl,
          theta_eval = theta_eval,
          diagnostico = diagnostico,
          resumen = resumen
        ))
      }
    })

    output$codigo <- renderUI({
      tipo <- input$tipo_integracion

      codigo <- switch(
        tipo,
        std2 = paste0(
          "datos <- generar_datos_std2_app(n = ", input$n_std2, ", seed = ", input$seed, ")\n\n",
          "modelo <- rnas_integrar_std2(\n",
          "  datos_std2 = datos,\n",
          "  target = \"y\",\n",
          "  features = c(\"", paste(input$features_std2, collapse = "\", \""), "\"),\n",
          "  horizonte = ", input$horizonte, ",\n",
          "  eta0 = ", input$eta0, ",\n",
          "  T = ", input$T_iter, ",\n",
          "  activation = \"", input$activation, "\"\n",
          ")"
        ),
        simuds = paste0(
          "tray <- generar_trayectorias_simuds_app(n = ", input$n_simuds, ", escenario_sel = \"", input$escenario, "\")\n\n",
          "modelo <- rnas_integrar_simuds(\n",
          "  trayectorias_simuds = tray,\n",
          "  estado_cols = \"s\",\n",
          "  target_estado = \"s\",\n",
          "  escenario_col = \"escenario\",\n",
          "  tiempo_col = \"t\",\n",
          "  horizonte = ", input$horizonte, ",\n",
          "  eta0 = ", input$eta0, ",\n",
          "  T = ", input$T_iter, ",\n",
          "  activation = \"", input$activation, "\"\n",
          ")"
        ),
        fnl = paste0(
          "fnl <- rnas_formular_fnl(\n",
          "  X = X,\n",
          "  y = y,\n",
          "  activation = \"", input$activation, "\",\n",
          "  restricciones = restricciones\n",
          ")\n\n",
          "theta_eval <- c(", input$w1, ", ", input$w2, ", ", input$b, ")\n",
          "fnl$objetivo(theta_eval)\n",
          "fnl$evaluar_restricciones(theta_eval)\n",
          "fnl$es_factible(theta_eval)"
        )
      )

      mostrar_codigo(codigo)
    })

    output$tabla_datos <- DT::renderDT({
      req(res())

      obj <- res()

      if (obj$tipo == "std2") {
        tabla_dt(utils::head(obj$resultado$preparacion$datos_alineados, 20))
      } else if (obj$tipo == "simuds") {
        tabla_dt(utils::head(obj$resultado$preparacion$pares, 20))
      } else {
        tabla_dt(utils::head(obj$datos, 20))
      }
    })

    output$tabla_resumen <- DT::renderDT({
      req(res())
      tabla_dt(res()$resumen)
    })

    output$tabla_trayectoria <- DT::renderDT({
      req(res())

      obj <- res()

      if (obj$tipo %in% c("std2", "simuds")) {
        tray <- obj$resultado$modelo_rnas$trayectoria
        columnas <- intersect(c("iter", "loss", "grad_norm", "eta", "speed", "b", "w1", "w2"), names(tray))
        tabla_dt(utils::head(tray[, columnas, drop = FALSE], 25))
      } else {
        tabla_dt(obj$diagnostico)
      }
    })

    output$plot <- renderPlot({
      req(res())

      obj <- res()

      if (obj$tipo == "std2") {
        datos <- obj$datos
        tray <- obj$resultado$modelo_rnas$trayectoria

        op <- par(mfrow = c(1, 2))
        on.exit(par(op), add = TRUE)

        plot(datos$tiempo, datos$y, type = "l", lwd = 2,
             xlab = "Tiempo", ylab = "Valor",
             main = "STD2: y y C")
        lines(datos$tiempo, datos$C, lwd = 2, lty = 2)
        legend("topleft", legend = c("y", "C"), lty = c(1, 2), lwd = 2, bty = "n")
        grid()

        plot(tray$iter, tray$loss, type = "l", lwd = 2,
             xlab = "Iteración", ylab = "Pérdida",
             main = "STD2-RNAS: pérdida")
        grid()
      }

      if (obj$tipo == "simuds") {
        datos <- obj$datos
        tray <- obj$resultado$modelo_rnas$trayectoria

        op <- par(mfrow = c(1, 2))
        on.exit(par(op), add = TRUE)

        plot(NULL,
             xlim = range(datos$t),
             ylim = range(datos$s),
             xlab = "Tiempo", ylab = "Estado s",
             main = "SimuDS: trayectorias")

        for (esc in unique(datos$escenario)) {
          sub <- datos[datos$escenario == esc, , drop = FALSE]
          lines(sub$t, sub$s, lwd = 2)
        }

        legend("topleft", legend = unique(datos$escenario), lty = 1, lwd = 2, bty = "n")
        grid()

        plot(tray$iter, tray$loss, type = "l", lwd = 2,
             xlab = "Iteración", ylab = "Pérdida",
             main = "SimuDS-RNAS: pérdida")
        grid()
      }

      if (obj$tipo == "fnl") {
        diag <- obj$diagnostico
        vals <- diag$valor
        names(vals) <- diag$indicador

        barplot(vals, las = 2, ylab = "Valor",
                main = "FNL-RNAS: objetivo y restricciones")
        abline(h = 0, lty = 2)
        grid()
      }
    })

    output$interpretacion <- renderText({
      req(res())

      obj <- res()

      if (obj$tipo == "std2") {
        r <- obj$resumen
        return(paste0(
          "La integración STD2-RNAS alineó ", r$n_obs,
          " observaciones con ", r$n_features,
          " señales dinámicas. La reducción relativa fue ",
          round(r$reduccion_rel, 4), "."
        ))
      }

      if (obj$tipo == "simuds") {
        r <- obj$resumen
        return(paste0(
          "La integración SimuDS-RNAS generó ", r$n_obs,
          " pares de transición. La reducción relativa fue ",
          round(r$reduccion_rel, 4), "."
        ))
      }

      if (obj$tipo == "fnl") {
        return(paste0(
          "La formulación FNL-RNAS evaluó una función objetivo y restricciones. ",
          "El estado evaluado fue factible: ", obj$resumen$factible, "."
        ))
      }
    })
  })
}
