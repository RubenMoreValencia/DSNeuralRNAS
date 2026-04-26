#' Ejecutar caso aplicado STD2-RNAS controlado
#'
#' Genera un conjunto controlado de señales tipo STD2 y ejecuta la integración
#' STD2-RNAS. El caso permite validar alineamiento temporal, entrenamiento y
#' reducción de pérdida sobre señales dinámicas.
#'
#' @param n Número de observaciones.
#' @param horizonte Horizonte de predicción.
#' @param eta0 Tasa de aprendizaje.
#' @param T Número de iteraciones.
#' @param activation Activación de la neurona.
#' @param seed Semilla de reproducibilidad.
#'
#' @return Objeto de clase `rnas_caso_aplicado_std2`.
#'
#' @examples
#' caso <- rnas_caso_std2_controlado(n = 30, T = 10)
#' caso$resumen
#'
#' @export
rnas_caso_std2_controlado <- function(n = 60L,
                                      horizonte = 1L,
                                      eta0 = 0.05,
                                      T = 100L,
                                      activation = "tanh",
                                      seed = 123) {
  if (!is.numeric(n) || length(n) != 1L || n <= horizonte + 2L) {
    stop("`n` debe ser mayor que horizonte + 2.")
  }

  if (!is.numeric(horizonte) ||
      length(horizonte) != 1L ||
      horizonte < 1 ||
      horizonte != as.integer(horizonte)) {
    stop("`horizonte` debe ser un entero positivo.")
  }

  set.seed(seed)

  tiempo <- seq_len(n)
  base <- tanh(seq(-1, 1, length.out = n))

  datos_std2 <- data.frame(
    tiempo = tiempo,
    y = base + stats::rnorm(n, sd = 0.02),
    C = base,
    x = sin(seq(0, 4, length.out = n)) / 5,
    e = stats::rnorm(n, sd = 0.03),
    gap = stats::rnorm(n, sd = 0.02),
    speed = abs(c(NA_real_, diff(base))),
    stringsAsFactors = FALSE
  )

  datos_std2$speed[1] <- datos_std2$speed[2]

  features <- c("C", "x", "e", "gap", "speed")

  integracion <- rnas_integrar_std2(
    datos_std2 = datos_std2,
    target = "y",
    features = features,
    horizonte = horizonte,
    eta0 = eta0,
    T = T,
    activation = activation
  )

  resumen_int <- rnas_resumen_integracion(integracion)

  resumen <- data.frame(
    caso = "STD2-RNAS",
    n_original = n,
    n_obs = resumen_int$n_obs,
    n_features = resumen_int$n_features,
    horizonte = resumen_int$horizonte,
    loss_inicial = resumen_int$loss_inicial,
    loss_final = resumen_int$loss_final,
    reduccion_abs = resumen_int$loss_inicial - resumen_int$loss_final,
    reduccion_rel = resumen_int$reduccion_rel,
    descendente_global = integracion$modelo_rnas$metricas$descendente_global,
    factible = NA,
    stringsAsFactors = FALSE
  )

  res <- list(
    datos = datos_std2,
    integracion = integracion,
    resumen = resumen,
    configuracion = list(
      n = n,
      horizonte = horizonte,
      eta0 = eta0,
      T = T,
      activation = activation,
      seed = seed,
      features = features
    )
  )

  class(res) <- c("rnas_caso_aplicado_std2", class(res))
  res
}


#' Ejecutar caso aplicado SimuDS-RNAS controlado
#'
#' Genera trayectorias simuladas para escenarios base, choque y recuperación.
#' Luego convierte esas trayectorias en pares de transición y entrena RNAS
#' para aproximar el estado siguiente.
#'
#' @param n Número de estados por escenario.
#' @param horizonte Horizonte de transición.
#' @param eta0 Tasa de aprendizaje.
#' @param T Número de iteraciones.
#' @param activation Activación de la neurona.
#' @param seed Semilla de reproducibilidad.
#'
#' @return Objeto de clase `rnas_caso_aplicado_simuds`.
#'
#' @examples
#' caso <- rnas_caso_simuds_controlado(n = 30, T = 10)
#' caso$resumen
#'
#' @export
rnas_caso_simuds_controlado <- function(n = 40L,
                                        horizonte = 1L,
                                        eta0 = 0.05,
                                        T = 100L,
                                        activation = "tanh",
                                        seed = 123) {
  if (!is.numeric(n) || length(n) != 1L || n <= horizonte + 2L) {
    stop("`n` debe ser mayor que horizonte + 2.")
  }

  if (!is.numeric(horizonte) ||
      length(horizonte) != 1L ||
      horizonte < 1 ||
      horizonte != as.integer(horizonte)) {
    stop("`horizonte` debe ser un entero positivo.")
  }

  set.seed(seed)

  escenarios <- c("base", "shock", "recuperacion")

  trayectorias <- do.call(
    rbind,
    lapply(escenarios, function(esc) {
      t <- seq_len(n)
      base <- tanh(seq(-1, 1, length.out = n))

      s <- switch(
        esc,
        base = base,
        shock = base + ifelse(t > floor(n / 2), -0.15, 0),
        recuperacion = base + ifelse(t > floor(n / 2), 0.10, 0)
      )

      data.frame(
        escenario = esc,
        t = t,
        s = s,
        stringsAsFactors = FALSE
      )
    })
  )

  integracion <- rnas_integrar_simuds(
    trayectorias_simuds = trayectorias,
    estado_cols = "s",
    target_estado = "s",
    escenario_col = "escenario",
    tiempo_col = "t",
    horizonte = horizonte,
    eta0 = eta0,
    T = T,
    activation = activation
  )

  resumen_int <- rnas_resumen_integracion(integracion)

  resumen <- data.frame(
    caso = "SimuDS-RNAS",
    n_original = nrow(trayectorias),
    n_obs = resumen_int$n_obs,
    n_features = resumen_int$n_features,
    horizonte = resumen_int$horizonte,
    loss_inicial = resumen_int$loss_inicial,
    loss_final = resumen_int$loss_final,
    reduccion_abs = resumen_int$loss_inicial - resumen_int$loss_final,
    reduccion_rel = resumen_int$reduccion_rel,
    descendente_global = integracion$modelo_rnas$metricas$descendente_global,
    factible = NA,
    stringsAsFactors = FALSE
  )

  res <- list(
    trayectorias = trayectorias,
    integracion = integracion,
    resumen = resumen,
    configuracion = list(
      n = n,
      escenarios = escenarios,
      horizonte = horizonte,
      eta0 = eta0,
      T = T,
      activation = activation,
      seed = seed
    )
  )

  class(res) <- c("rnas_caso_aplicado_simuds", class(res))
  res
}


#' Ejecutar caso aplicado FNL-RNAS controlado
#'
#' Construye una formulación funcional no lineal para RNAS, evalúa una función
#' objetivo y verifica restricciones de factibilidad sobre un estado theta.
#'
#' @param n Número de observaciones.
#' @param activation Activación de la neurona.
#' @param seed Semilla de reproducibilidad.
#'
#' @return Objeto de clase `rnas_caso_aplicado_fnl`.
#'
#' @examples
#' caso <- rnas_caso_fnl_controlado(n = 30)
#' caso$resumen
#'
#' @export
rnas_caso_fnl_controlado <- function(n = 60L,
                                     activation = "tanh",
                                     seed = 123) {
  if (!is.numeric(n) || length(n) != 1L || n < 5L) {
    stop("`n` debe ser un entero mayor o igual que 5.")
  }

  set.seed(seed)

  tiempo <- seq_len(n)
  C <- tanh(seq(-1, 1, length.out = n))
  x <- sin(seq(0, 4, length.out = n)) / 5
  y <- C + 0.2 * x + stats::rnorm(n, sd = 0.02)

  X <- cbind(C = C, x = x)

  restricciones <- list(
    norma_theta = function(theta) sum(theta^2) - 10,
    cota_w1 = function(theta) theta[1] - 3
  )

  fnl <- rnas_formular_fnl(
    X = X,
    y = y,
    activation = activation,
    restricciones = restricciones
  )

  theta_eval <- c(0.1, 0.1, 0.0)

  valor_objetivo <- fnl$objetivo(theta_eval)
  valores_restricciones <- fnl$evaluar_restricciones(theta_eval)
  factible <- fnl$es_factible(theta_eval)

  diagnostico <- data.frame(
    indicador = c(
      "valor_objetivo",
      names(valores_restricciones),
      "factible"
    ),
    valor = c(
      valor_objetivo,
      valores_restricciones,
      as.numeric(factible)
    ),
    stringsAsFactors = FALSE
  )

  resumen <- data.frame(
    caso = "FNL-RNAS",
    n_original = n,
    n_obs = n,
    n_features = ncol(X),
    horizonte = NA_integer_,
    loss_inicial = NA_real_,
    loss_final = valor_objetivo,
    reduccion_abs = NA_real_,
    reduccion_rel = NA_real_,
    descendente_global = NA,
    factible = factible,
    stringsAsFactors = FALSE
  )

  res <- list(
    X = X,
    y = y,
    formulacion = fnl,
    theta_eval = theta_eval,
    diagnostico = diagnostico,
    resumen = resumen,
    configuracion = list(
      n = n,
      activation = activation,
      seed = seed,
      n_restricciones = length(restricciones)
    )
  )

  class(res) <- c("rnas_caso_aplicado_fnl", class(res))
  res
}


#' Consolidar casos aplicados RNAS
#'
#' Consolida las métricas principales de una lista de casos aplicados.
#'
#' @param casos Lista de objetos de casos aplicados.
#'
#' @return Data frame consolidado.
#'
#' @examples
#' c1 <- rnas_caso_std2_controlado(n = 30, T = 10)
#' c2 <- rnas_caso_simuds_controlado(n = 20, T = 10)
#' rnas_consolidar_casos(list(c1, c2))
#'
#' @export
rnas_consolidar_casos <- function(casos) {
  if (!is.list(casos) || length(casos) == 0L) {
    stop("`casos` debe ser una lista no vacia.")
  }

  resumenes <- lapply(casos, function(obj) {
    if (is.null(obj$resumen) || !is.data.frame(obj$resumen)) {
      stop("Cada caso debe contener un data frame llamado `resumen`.")
    }

    obj$resumen
  })

  out <- do.call(rbind, resumenes)
  rownames(out) <- NULL
  out
}


#' Ejecutar todos los casos aplicados del capítulo 13
#'
#' Ejecuta los casos STD2-RNAS, SimuDS-RNAS y FNL-RNAS bajo una configuración
#' controlada y devuelve un objeto consolidado.
#'
#' @param n_std2 Número de observaciones para el caso STD2.
#' @param n_simuds Número de estados por escenario para SimuDS.
#' @param n_fnl Número de observaciones para FNL.
#' @param horizonte Horizonte de predicción o transición.
#' @param eta0 Tasa de aprendizaje.
#' @param T Número de iteraciones.
#' @param activation Activación de la neurona.
#' @param seed Semilla de reproducibilidad.
#'
#' @return Objeto de clase `rnas_casos_aplicados`.
#'
#' @examples
#' res <- rnas_ejecutar_casos_aplicados(T = 10)
#' res$resumen_casos
#'
#' @export
rnas_ejecutar_casos_aplicados <- function(n_std2 = 60L,
                                          n_simuds = 40L,
                                          n_fnl = 60L,
                                          horizonte = 1L,
                                          eta0 = 0.05,
                                          T = 100L,
                                          activation = "tanh",
                                          seed = 123) {
  caso_std2 <- rnas_caso_std2_controlado(
    n = n_std2,
    horizonte = horizonte,
    eta0 = eta0,
    T = T,
    activation = activation,
    seed = seed
  )

  caso_simuds <- rnas_caso_simuds_controlado(
    n = n_simuds,
    horizonte = horizonte,
    eta0 = eta0,
    T = T,
    activation = activation,
    seed = seed
  )

  caso_fnl <- rnas_caso_fnl_controlado(
    n = n_fnl,
    activation = activation,
    seed = seed
  )

  resumen <- rnas_consolidar_casos(
    list(
      caso_std2,
      caso_simuds,
      caso_fnl
    )
  )

  diagnostico <- list(
    n_casos = nrow(resumen),
    casos = resumen$caso,
    casos_con_descenso = sum(resumen$descendente_global %in% TRUE, na.rm = TRUE),
    casos_factibles = sum(resumen$factible %in% TRUE, na.rm = TRUE)
  )

  res <- list(
    caso_std2 = caso_std2,
    caso_simuds = caso_simuds,
    caso_fnl = caso_fnl,
    resumen_casos = resumen,
    diagnostico = diagnostico,
    configuracion = list(
      n_std2 = n_std2,
      n_simuds = n_simuds,
      n_fnl = n_fnl,
      horizonte = horizonte,
      eta0 = eta0,
      T = T,
      activation = activation,
      seed = seed
    )
  )

  class(res) <- c("rnas_casos_aplicados", class(res))
  res
}


#' Imprimir caso aplicado STD2-RNAS
#'
#' @param x Objeto `rnas_caso_aplicado_std2`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto x de forma invisible.
#'
#' @export
print.rnas_caso_aplicado_std2 <- function(x, ...) {
  cat("Caso aplicado STD2-RNAS\n")
  cat("-----------------------\n")
  cat("Obs. alineadas :", x$resumen$n_obs, "\n")
  cat("Features       :", x$resumen$n_features, "\n")
  cat("Loss inicial   :", x$resumen$loss_inicial, "\n")
  cat("Loss final     :", x$resumen$loss_final, "\n")
  cat("Reduccion rel. :", x$resumen$reduccion_rel, "\n")

  invisible(x)
}


#' Imprimir caso aplicado SimuDS-RNAS
#'
#' @param x Objeto `rnas_caso_aplicado_simuds`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto x de forma invisible.
#'
#' @export
print.rnas_caso_aplicado_simuds <- function(x, ...) {
  cat("Caso aplicado SimuDS-RNAS\n")
  cat("-------------------------\n")
  cat("Pares          :", x$resumen$n_obs, "\n")
  cat("Features       :", x$resumen$n_features, "\n")
  cat("Loss inicial   :", x$resumen$loss_inicial, "\n")
  cat("Loss final     :", x$resumen$loss_final, "\n")
  cat("Reduccion rel. :", x$resumen$reduccion_rel, "\n")

  invisible(x)
}


#' Imprimir caso aplicado FNL-RNAS
#'
#' @param x Objeto `rnas_caso_aplicado_fnl`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto x de forma invisible.
#'
#' @export
print.rnas_caso_aplicado_fnl <- function(x, ...) {
  cat("Caso aplicado FNL-RNAS\n")
  cat("----------------------\n")
  cat("Observaciones :", x$resumen$n_obs, "\n")
  cat("Features      :", x$resumen$n_features, "\n")
  cat("Valor objetivo:", x$resumen$loss_final, "\n")
  cat("Factible      :", x$resumen$factible, "\n")

  invisible(x)
}


#' Imprimir ejecución integral de casos aplicados
#'
#' @param x Objeto `rnas_casos_aplicados`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto x de forma invisible.
#'
#' @export
print.rnas_casos_aplicados <- function(x, ...) {
  cat("Casos aplicados DS Neural RNAS\n")
  cat("------------------------------\n")
  cat("Casos ejecutados :", x$diagnostico$n_casos, "\n")
  cat("Con descenso     :", x$diagnostico$casos_con_descenso, "\n")
  cat("Factibles        :", x$diagnostico$casos_factibles, "\n")

  invisible(x)
}
