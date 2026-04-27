# ============================================================
# Modulo Capitulo 13: Casos aplicados
# DSNeuralRNASLab
# ============================================================

mod_cap13_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Capítulo 13: Casos aplicados y resultados consolidados"),

    div(
      class = "bloque-explicacion",
      p("Este módulo ejecuta casos aplicados controlados del modelo DS Neural RNAS: STD2-RNAS, SimuDS-RNAS, FNL-RNAS o la ejecución integral de los tres casos.")
    ),

    fluidRow(
      column(
        4,

        selectInput(
          ns("tipo_caso"),
          "Tipo de caso aplicado",
          choices = c(
            "STD2-RNAS" = "std2",
            "SimuDS-RNAS" = "simuds",
            "FNL-RNAS" = "fnl",
            "Integral" = "integral"
          ),
          selected = "integral"
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

        numericInput(ns("n_std2"), "Observaciones STD2", value = 60, min = 15, max = 500, step = 5),
        numericInput(ns("n_simuds"), "Estados por escenario SimuDS", value = 40, min = 15, max = 500, step = 5),
        numericInput(ns("n_fnl"), "Observaciones FNL", value = 60, min = 10, max = 500, step = 5),

        actionButton(ns("run"), "Ejecutar caso", class = "btn-primary")
      ),

      column(
        8,
        tabsetPanel(
          tabPanel("Código", br(), uiOutput(ns("codigo"))),
          tabPanel("Resumen", br(), DT::DTOutput(ns("tabla_resumen"))),
          tabPanel("Datos / diagnóstico", br(), DT::DTOutput(ns("tabla_datos"))),
          tabPanel("Gráficas", br(), plotOutput(ns("plot"), height = "440px")),
          tabPanel("Interpretación", br(), verbatimTextOutput(ns("interpretacion"))),
          tabPanel("Objeto", br(), verbatimTextOutput(ns("estructura")))
        )
      )
    )
  )
}

mod_cap13_server <- function(id) {
  moduleServer(id, function(input, output, session) {

    res <- eventReactive(input$run, {
      tipo <- input$tipo_caso

      if (tipo == "std2") {
        caso <- DSNeuralRNAS::rnas_caso_std2_controlado(
          n = as.integer(input$n_std2),
          horizonte = as.integer(input$horizonte),
          eta0 = input$eta0,
          T = as.integer(input$T_iter),
          activation = input$activation,
          seed = as.integer(input$seed)
        )

        return(list(tipo = tipo, caso = caso, resumen = caso$resumen))
      }

      if (tipo == "simuds") {
        caso <- DSNeuralRNAS::rnas_caso_simuds_controlado(
          n = as.integer(input$n_simuds),
          horizonte = as.integer(input$horizonte),
          eta0 = input$eta0,
          T = as.integer(input$T_iter),
          activation = input$activation,
          seed = as.integer(input$seed)
        )

        return(list(tipo = tipo, caso = caso, resumen = caso$resumen))
      }

      if (tipo == "fnl") {
        caso <- DSNeuralRNAS::rnas_caso_fnl_controlado(
          n = as.integer(input$n_fnl),
          activation = input$activation,
          seed = as.integer(input$seed)
        )

        return(list(tipo = tipo, caso = caso, resumen = caso$resumen))
      }

      if (tipo == "integral") {
        casos <- DSNeuralRNAS::rnas_ejecutar_casos_aplicados(
          n_std2 = as.integer(input$n_std2),
          n_simuds = as.integer(input$n_simuds),
          n_fnl = as.integer(input$n_fnl),
          horizonte = as.integer(input$horizonte),
          eta0 = input$eta0,
          T = as.integer(input$T_iter),
          activation = input$activation,
          seed = as.integer(input$seed)
        )

        return(list(tipo = tipo, caso = casos, resumen = casos$resumen_casos))
      }
    })

    output$codigo <- renderUI({
      tipo <- input$tipo_caso

      codigo <- switch(
        tipo,
        std2 = paste0(
          "caso <- rnas_caso_std2_controlado(\n",
          "  n = ", input$n_std2, ",\n",
          "  horizonte = ", input$horizonte, ",\n",
          "  eta0 = ", input$eta0, ",\n",
          "  T = ", input$T_iter, ",\n",
          "  activation = \"", input$activation, "\",\n",
          "  seed = ", input$seed, "\n",
          ")"
        ),
        simuds = paste0(
          "caso <- rnas_caso_simuds_controlado(\n",
          "  n = ", input$n_simuds, ",\n",
          "  horizonte = ", input$horizonte, ",\n",
          "  eta0 = ", input$eta0, ",\n",
          "  T = ", input$T_iter, ",\n",
          "  activation = \"", input$activation, "\",\n",
          "  seed = ", input$seed, "\n",
          ")"
        ),
        fnl = paste0(
          "caso <- rnas_caso_fnl_controlado(\n",
          "  n = ", input$n_fnl, ",\n",
          "  activation = \"", input$activation, "\",\n",
          "  seed = ", input$seed, "\n",
          ")"
        ),
        integral = paste0(
          "casos <- rnas_ejecutar_casos_aplicados(\n",
          "  n_std2 = ", input$n_std2, ",\n",
          "  n_simuds = ", input$n_simuds, ",\n",
          "  n_fnl = ", input$n_fnl, ",\n",
          "  horizonte = ", input$horizonte, ",\n",
          "  eta0 = ", input$eta0, ",\n",
          "  T = ", input$T_iter, ",\n",
          "  activation = \"", input$activation, "\",\n",
          "  seed = ", input$seed, "\n",
          ")"
        )
      )

      mostrar_codigo(codigo)
    })

    output$tabla_resumen <- DT::renderDT({
      req(res())
      tabla_dt(res()$resumen)
    })

    output$tabla_datos <- DT::renderDT({
      req(res())

      obj <- res()
      caso <- obj$caso

      if (obj$tipo == "std2") {
        return(tabla_dt(utils::head(caso$integracion$preparacion$datos_alineados, 20)))
      }

      if (obj$tipo == "simuds") {
        return(tabla_dt(utils::head(caso$integracion$preparacion$pares, 20)))
      }

      if (obj$tipo == "fnl") {
        return(tabla_dt(caso$diagnostico))
      }

      if (obj$tipo == "integral") {
        diag <- data.frame(
          indicador = c("n_casos", "casos_con_descenso", "casos_factibles"),
          valor = c(
            caso$diagnostico$n_casos,
            caso$diagnostico$casos_con_descenso,
            caso$diagnostico$casos_factibles
          ),
          stringsAsFactors = FALSE
        )

        return(tabla_dt(diag))
      }
    })

    output$plot <- renderPlot({
      req(res())

      obj <- res()
      caso <- obj$caso

      if (obj$tipo == "std2") {
        datos <- caso$datos
        tray <- caso$integracion$modelo_rnas$trayectoria

        op <- par(mfrow = c(1, 2))
        on.exit(par(op), add = TRUE)

        plot(datos$tiempo, datos$y, type = "l", lwd = 2,
             xlab = "Tiempo", ylab = "Valor",
             main = "Caso STD2-RNAS")
        lines(datos$tiempo, datos$C, lwd = 2, lty = 2)
        legend("topleft", legend = c("y", "C"), lty = c(1, 2), lwd = 2, bty = "n")
        grid()

        plot(tray$iter, tray$loss, type = "l", lwd = 2,
             xlab = "Iteración", ylab = "Pérdida",
             main = "Pérdida STD2-RNAS")
        grid()
      }

      if (obj$tipo == "simuds") {
        datos <- caso$trayectorias
        tray <- caso$integracion$modelo_rnas$trayectoria

        op <- par(mfrow = c(1, 2))
        on.exit(par(op), add = TRUE)

        plot(NULL,
             xlim = range(datos$t),
             ylim = range(datos$s),
             xlab = "Tiempo", ylab = "Estado s",
             main = "Caso SimuDS-RNAS")

        for (esc in unique(datos$escenario)) {
          sub <- datos[datos$escenario == esc, , drop = FALSE]
          lines(sub$t, sub$s, lwd = 2)
        }

        legend("topleft", legend = unique(datos$escenario), lty = 1, lwd = 2, bty = "n")
        grid()

        plot(tray$iter, tray$loss, type = "l", lwd = 2,
             xlab = "Iteración", ylab = "Pérdida",
             main = "Pérdida SimuDS-RNAS")
        grid()
      }

      if (obj$tipo == "fnl") {
        diag <- caso$diagnostico
        vals <- diag$valor
        names(vals) <- diag$indicador

        barplot(vals, las = 2,
                ylab = "Valor",
                main = "FNL-RNAS: objetivo y restricciones")
        abline(h = 0, lty = 2)
        grid()
      }

      if (obj$tipo == "integral") {
        tab <- caso$resumen_casos
        tab2 <- tab[!is.na(tab$loss_final), ]

        barplot(
          height = tab2$loss_final,
          names.arg = tab2$caso,
          las = 2,
          ylab = "Pérdida / valor objetivo final",
          main = "Comparación de casos aplicados"
        )
        grid()
      }
    })

    output$interpretacion <- renderText({
      req(res())

      obj <- res()

      if (obj$tipo == "std2") {
        r <- obj$resumen
        return(paste0(
          "El caso STD2-RNAS usó ", r$n_obs,
          " observaciones alineadas y ", r$n_features,
          " señales dinámicas. La reducción relativa fue ",
          round(r$reduccion_rel, 4), "."
        ))
      }

      if (obj$tipo == "simuds") {
        r <- obj$resumen
        return(paste0(
          "El caso SimuDS-RNAS generó ", r$n_obs,
          " pares de transición. La reducción relativa fue ",
          round(r$reduccion_rel, 4), "."
        ))
      }

      if (obj$tipo == "fnl") {
        return(paste0(
          "El caso FNL-RNAS evaluó una función objetivo y restricciones. ",
          "El estado evaluado fue factible: ", obj$resumen$factible, "."
        ))
      }

      if (obj$tipo == "integral") {
        d <- obj$caso$diagnostico
        return(paste0(
          "La ejecución integral consolidó ", d$n_casos,
          " casos. Casos con descenso global: ",
          d$casos_con_descenso,
          ". Casos factibles: ", d$casos_factibles, "."
        ))
      }
    })

    output$estructura <- renderPrint({
      req(res())
      str(res(), max.level = 3)
    })
  })
}
