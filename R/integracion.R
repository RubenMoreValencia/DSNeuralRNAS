#' Preparar features STD2 para aprendizaje RNAS
#'
#' Construye una matriz de entrada y una variable objetivo a partir de un
#' data frame con señales STD2. La función alinea temporalmente las entradas
#' X_t con el objetivo y_{t+h}, evitando usar información futura como entrada.
#' \eqn{X_t} con el objetivo \eqn{y_{t+h}}, evitando usar informacion futura como entrada.#'
#' @param datos_std2 Data frame con señales STD2 y variable objetivo.
#' @param target Nombre de la variable objetivo.
#' @param features Vector de nombres de variables usadas como entradas.
#' @param horizonte Entero positivo que define el horizonte h.
#' @param quitar_na Valor lógico. Si TRUE, elimina filas incompletas.
#'
#' @return Lista con X_rnas, y_rnas, datos_alineados y configuracion.
#'
#' @examples
#' datos <- data.frame(
#'   y = 1:10,
#'   C = seq(1, 2, length.out = 10),
#'   x = rnorm(10),
#'   e = rnorm(10)
#' )
#' rnas_preparar_features_std2(datos, target = "y", features = c("C", "x", "e"))
#'
#' @export
rnas_preparar_features_std2 <- function(datos_std2,
                                        target,
                                        features,
                                        horizonte = 1L,
                                        quitar_na = TRUE) {
  if (!is.data.frame(datos_std2)) {
    stop("`datos_std2` debe ser un data frame.")
  }

  if (!is.character(target) || length(target) != 1L) {
    stop("`target` debe ser un texto con el nombre de la variable objetivo.")
  }

  if (!is.character(features) || length(features) < 1L) {
    stop("`features` debe ser un vector de nombres de variables.")
  }

  if (!target %in% names(datos_std2)) {
    stop("`target` no existe en `datos_std2`.")
  }

  if (!all(features %in% names(datos_std2))) {
    stop("Todas las variables de `features` deben existir en `datos_std2`.")
  }

  if (!is.numeric(horizonte) ||
      length(horizonte) != 1L ||
      horizonte < 1 ||
      horizonte != as.integer(horizonte)) {
    stop("`horizonte` debe ser un entero positivo.")
  }

  n <- nrow(datos_std2)

  if (n <= horizonte) {
    stop("El número de filas debe ser mayor que `horizonte`.")
  }

  idx_x <- seq_len(n - horizonte)
  idx_y <- seq(1 + horizonte, n)

  X_df <- datos_std2[idx_x, features, drop = FALSE]
  y_vec <- datos_std2[[target]][idx_y]

  datos_alineados <- data.frame(
    t_origen = idx_x,
    t_objetivo = idx_y,
    X_df,
    y_rnas = y_vec,
    stringsAsFactors = FALSE
  )

  if (isTRUE(quitar_na)) {
    datos_alineados <- stats::na.omit(datos_alineados)
  }

  X_rnas <- as.matrix(datos_alineados[, features, drop = FALSE])
  y_rnas <- as.numeric(datos_alineados$y_rnas)

  if (any(!is.finite(X_rnas)) || any(!is.finite(y_rnas))) {
    stop("Después del alineamiento, `X_rnas` e `y_rnas` deben contener valores finitos.")
  }

  res <- list(
    X_rnas = X_rnas,
    y_rnas = y_rnas,
    datos_alineados = datos_alineados,
    configuracion = list(
      target = target,
      features = features,
      horizonte = horizonte,
      quitar_na = quitar_na,
      n_original = nrow(datos_std2),
      n_alineado = nrow(datos_alineados)
    )
  )

  class(res) <- c("rnas_std2_features", class(res))
  res
}


#' Integrar señales STD2 con entrenamiento RNAS
#'
#' Prepara señales STD2 y entrena una neurona RNAS usando la matriz alineada.
#' Esta función opera como puente entre señales temporales estructurales y el
#' núcleo de aprendizaje RNAS.
#'
#' @param datos_std2 Data frame con señales STD2.
#' @param target Nombre de la variable objetivo.
#' @param features Vector de nombres de variables predictoras.
#' @param horizonte Horizonte de predicción.
#' @param theta0 Vector inicial de parámetros. Si es NULL, se inicializa en cero.
#' @param eta0 Tasa de aprendizaje inicial.
#' @param T Número de iteraciones.
#' @param activation Activación de la neurona.
#'
#' @return Objeto con datos preparados, modelo RNAS, métricas y configuración.
#'
#' @export
rnas_integrar_std2 <- function(datos_std2,
                               target,
                               features,
                               horizonte = 1L,
                               theta0 = NULL,
                               eta0 = 0.1,
                               T = 100,
                               activation = "tanh") {
  prep <- rnas_preparar_features_std2(
    datos_std2 = datos_std2,
    target = target,
    features = features,
    horizonte = horizonte
  )

  X <- prep$X_rnas
  y <- prep$y_rnas

  if (is.null(theta0)) {
    theta0 <- rep(0, ncol(X) + 1L)
  }

  if (!is.numeric(theta0) || length(theta0) != ncol(X) + 1L) {
    stop("`theta0` debe tener longitud ncol(X) + 1.")
  }

  modelo <- rnas_train_neuron_control_eta(
    X = X,
    y = y,
    theta0 = theta0,
    eta0 = eta0,
    policy = "constante",
    T = T,
    activation = activation
  )

  res <- list(
    preparacion = prep,
    modelo_rnas = modelo,
    metricas = modelo$metricas,
    configuracion = list(
      tipo_integracion = "STD2-RNAS",
      target = target,
      features = features,
      horizonte = horizonte,
      eta0 = eta0,
      T = T,
      activation = activation
    )
  )

  class(res) <- c("rnas_std2_integration", class(res))
  res
}


#' Preparar trayectorias SimuDS para aprendizaje RNAS
#'
#' Convierte trayectorias simuladas en pares de aprendizaje del tipo
#' s_t -> s_{t+h}. Conserva identificadores de escenario cuando están
#' disponibles.
#'
#' @param trayectorias_simuds Data frame con trayectorias simuladas.
#' @param estado_cols Vector de columnas que representan el estado.
#' @param escenario_col Nombre de la columna de escenario. Puede ser NULL.
#' @param tiempo_col Nombre de la columna de tiempo. Puede ser NULL.
#' @param horizonte Horizonte de transición.
#' @param quitar_na Valor lógico. Si TRUE, elimina filas incompletas.
#'
#' @return Lista con X_rnas, y_rnas, pares y configuración.
#'
#' @export
rnas_preparar_trayectorias_simuds <- function(trayectorias_simuds,
                                              estado_cols,
                                              escenario_col = NULL,
                                              tiempo_col = NULL,
                                              horizonte = 1L,
                                              quitar_na = TRUE) {
  if (!is.data.frame(trayectorias_simuds)) {
    stop("`trayectorias_simuds` debe ser un data frame.")
  }

  if (!is.character(estado_cols) || length(estado_cols) < 1L) {
    stop("`estado_cols` debe ser un vector de nombres de columnas de estado.")
  }

  if (!all(estado_cols %in% names(trayectorias_simuds))) {
    stop("Todas las columnas de `estado_cols` deben existir en `trayectorias_simuds`.")
  }

  if (!is.null(escenario_col) && !escenario_col %in% names(trayectorias_simuds)) {
    stop("`escenario_col` no existe en `trayectorias_simuds`.")
  }

  if (!is.null(tiempo_col) && !tiempo_col %in% names(trayectorias_simuds)) {
    stop("`tiempo_col` no existe en `trayectorias_simuds`.")
  }

  if (!is.numeric(horizonte) ||
      length(horizonte) != 1L ||
      horizonte < 1 ||
      horizonte != as.integer(horizonte)) {
    stop("`horizonte` debe ser un entero positivo.")
  }

  datos <- trayectorias_simuds

  if (is.null(escenario_col)) {
    datos$.escenario_rnas <- "escenario_1"
    escenario_col <- ".escenario_rnas"
  }

  if (is.null(tiempo_col)) {
    datos$.tiempo_rnas <- ave(
      seq_len(nrow(datos)),
      datos[[escenario_col]],
      FUN = seq_along
    )
    tiempo_col <- ".tiempo_rnas"
  }

  datos <- datos[order(datos[[escenario_col]], datos[[tiempo_col]]), , drop = FALSE]

  escenarios <- unique(datos[[escenario_col]])
  pares_lista <- list()

  idx <- 1L

  for (esc in escenarios) {
    sub <- datos[datos[[escenario_col]] == esc, , drop = FALSE]
    n <- nrow(sub)

    if (n <= horizonte) {
      next
    }

    for (i in seq_len(n - horizonte)) {
      j <- i + horizonte

      fila_x <- sub[i, estado_cols, drop = FALSE]
      fila_y <- sub[j, estado_cols, drop = FALSE]

      par <- data.frame(
        escenario = esc,
        t_origen = sub[[tiempo_col]][i],
        t_objetivo = sub[[tiempo_col]][j],
        fila_x,
        stringsAsFactors = FALSE
      )

      for (col in estado_cols) {
        par[[paste0(col, "_target")]] <- fila_y[[col]]
      }

      pares_lista[[idx]] <- par
      idx <- idx + 1L
    }
  }

  if (length(pares_lista) == 0) {
    stop("No se generaron pares de aprendizaje. Revise horizonte y trayectorias.")
  }

  pares <- do.call(rbind, pares_lista)
  rownames(pares) <- NULL

  if (isTRUE(quitar_na)) {
    pares <- stats::na.omit(pares)
  }

  X_rnas <- as.matrix(pares[, estado_cols, drop = FALSE])
  y_cols <- paste0(estado_cols, "_target")
  y_rnas <- as.matrix(pares[, y_cols, drop = FALSE])

  if (ncol(y_rnas) == 1L) {
    y_rnas <- as.numeric(y_rnas[, 1])
  }

  res <- list(
    X_rnas = X_rnas,
    y_rnas = y_rnas,
    pares = pares,
    configuracion = list(
      estado_cols = estado_cols,
      escenario_col = escenario_col,
      tiempo_col = tiempo_col,
      horizonte = horizonte,
      n_pares = nrow(pares)
    )
  )

  class(res) <- c("rnas_simuds_pairs", class(res))
  res
}


#' Integrar SimuDS con RNAS
#'
#' Prepara pares de transición desde trayectorias simuladas y entrena una
#' neurona RNAS cuando el objetivo es unidimensional.
#'
#' @param trayectorias_simuds Data frame con trayectorias simuladas.
#' @param estado_cols Columnas de estado.
#' @param target_estado Variable de estado que se desea aprender. Si NULL y
#' hay una sola columna de estado, se usa esa columna.
#' @param escenario_col Columna de escenario.
#' @param tiempo_col Columna de tiempo.
#' @param horizonte Horizonte de transición.
#' @param theta0 Vector inicial de parámetros.
#' @param eta0 Tasa de aprendizaje.
#' @param T Número de iteraciones.
#' @param activation Activación.
#'
#' @return Objeto con pares de transición, modelo y métricas.
#'
#' @export
rnas_integrar_simuds <- function(trayectorias_simuds,
                                 estado_cols,
                                 target_estado = NULL,
                                 escenario_col = NULL,
                                 tiempo_col = NULL,
                                 horizonte = 1L,
                                 theta0 = NULL,
                                 eta0 = 0.1,
                                 T = 100,
                                 activation = "tanh") {
  prep <- rnas_preparar_trayectorias_simuds(
    trayectorias_simuds = trayectorias_simuds,
    estado_cols = estado_cols,
    escenario_col = escenario_col,
    tiempo_col = tiempo_col,
    horizonte = horizonte
  )

  X <- prep$X_rnas

  if (is.null(target_estado)) {
    if (length(estado_cols) != 1L) {
      stop("Si hay varias columnas de estado, debe indicar `target_estado`.")
    }
    target_estado <- estado_cols[1]
  }

  if (!target_estado %in% estado_cols) {
    stop("`target_estado` debe pertenecer a `estado_cols`.")
  }

  y_col <- paste0(target_estado, "_target")
  y <- as.numeric(prep$pares[[y_col]])

  if (is.null(theta0)) {
    theta0 <- rep(0, ncol(X) + 1L)
  }

  modelo <- rnas_train_neuron_control_eta(
    X = X,
    y = y,
    theta0 = theta0,
    eta0 = eta0,
    policy = "constante",
    T = T,
    activation = activation
  )

  res <- list(
    preparacion = prep,
    modelo_rnas = modelo,
    metricas = modelo$metricas,
    configuracion = list(
      tipo_integracion = "SimuDS-RNAS",
      estado_cols = estado_cols,
      target_estado = target_estado,
      horizonte = horizonte,
      eta0 = eta0,
      T = T,
      activation = activation
    )
  )

  class(res) <- c("rnas_simuds_integration", class(res))
  res
}


#' Formular aprendizaje RNAS como problema FNL
#'
#' Construye una representación funcional del aprendizaje RNAS como problema
#' no lineal. La función devuelve evaluadores para pérdida, gradiente numérico
#' aproximado, restricciones opcionales y factibilidad.
#'
#' @param X Matriz numérica de entradas.
#' @param y Vector numérico observado.
#' @param activation Activación de la neurona.
#' @param restricciones Lista opcional de funciones que reciben theta y devuelven
#' valores numéricos. Se interpretan como g(theta) <= 0.
#'
#' @return Lista con funciones objetivo, restricciones y diagnóstico.
#'
#' @export
rnas_formular_fnl <- function(X,
                              y,
                              activation = "tanh",
                              restricciones = NULL) {
  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (nrow(X) != length(y)) {
    stop("La longitud de `y` debe coincidir con el numero de filas de `X`.")
  }

  if (!is.null(restricciones) && !is.list(restricciones)) {
    stop("`restricciones` debe ser NULL o una lista de funciones.")
  }

  if (!is.null(restricciones)) {
    es_fun <- vapply(restricciones, is.function, logical(1))
    if (!all(es_fun)) {
      stop("Todas las restricciones deben ser funciones.")
    }
  }

  objetivo <- function(theta) {
    if (!is.numeric(theta) || length(theta) != ncol(X) + 1L) {
      stop("`theta` debe tener longitud ncol(X) + 1.")
    }

    pars <- rnas_unpack_neuron_params(theta, d_input = ncol(X))

    rnas_loss_mse_neuron(
      X = X,
      y = y,
      w = pars$w,
      b = pars$b,
      activation = activation
    )
  }

  evaluar_restricciones <- function(theta) {
    if (is.null(restricciones)) {
      return(numeric(0))
    }

    vals <- vapply(restricciones, function(f) f(theta), numeric(1))
    vals
  }

  es_factible <- function(theta, tol = 1e-8) {
    vals <- evaluar_restricciones(theta)

    if (length(vals) == 0) {
      return(TRUE)
    }

    all(vals <= tol)
  }

  res <- list(
    objetivo = objetivo,
    restricciones = restricciones,
    evaluar_restricciones = evaluar_restricciones,
    es_factible = es_factible,
    configuracion = list(
      tipo_integracion = "FNL-RNAS",
      n_obs = nrow(X),
      n_features = ncol(X),
      activation = activation,
      n_restricciones = if (is.null(restricciones)) 0L else length(restricciones)
    )
  )

  class(res) <- c("rnas_fnl_formulation", class(res))
  res
}


#' Resumir integración DS Neural RNAS
#'
#' Consolida un resumen general de objetos de integración STD2, SimuDS o FNL.
#'
#' @param object Objeto de integración RNAS.
#'
#' @return Data frame con resumen interpretativo.
#'
#' @export
rnas_resumen_integracion <- function(object) {
  if (inherits(object, "rnas_std2_integration")) {
    return(data.frame(
      integracion = "STD2-RNAS",
      n_obs = object$preparacion$configuracion$n_alineado,
      n_features = length(object$configuracion$features),
      horizonte = object$configuracion$horizonte,
      loss_inicial = object$metricas$loss_inicial,
      loss_final = object$metricas$loss_final,
      reduccion_rel = object$metricas$reduccion_rel,
      stringsAsFactors = FALSE
    ))
  }

  if (inherits(object, "rnas_simuds_integration")) {
    return(data.frame(
      integracion = "SimuDS-RNAS",
      n_obs = object$preparacion$configuracion$n_pares,
      n_features = length(object$configuracion$estado_cols),
      horizonte = object$configuracion$horizonte,
      loss_inicial = object$metricas$loss_inicial,
      loss_final = object$metricas$loss_final,
      reduccion_rel = object$metricas$reduccion_rel,
      stringsAsFactors = FALSE
    ))
  }

  if (inherits(object, "rnas_fnl_formulation")) {
    return(data.frame(
      integracion = "FNL-RNAS",
      n_obs = object$configuracion$n_obs,
      n_features = object$configuracion$n_features,
      horizonte = NA_integer_,
      loss_inicial = NA_real_,
      loss_final = NA_real_,
      reduccion_rel = NA_real_,
      n_restricciones = object$configuracion$n_restricciones,
      stringsAsFactors = FALSE
    ))
  }

  stop("Tipo de objeto no reconocido para resumen de integración.")
}


#' Imprimir objeto de preparación STD2 para RNAS
#'
#' @param x Objeto `rnas_std2_features`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto x de forma invisible.
#'
#' @export
print.rnas_std2_features <- function(x, ...) {
  cat("Preparacion STD2 para RNAS\n")
  cat("--------------------------\n")
  cat("Target       :", x$configuracion$target, "\n")
  cat("Features     :", paste(x$configuracion$features, collapse = ", "), "\n")
  cat("Horizonte    :", x$configuracion$horizonte, "\n")
  cat("Obs. alineadas:", x$configuracion$n_alineado, "\n")

  invisible(x)
}


#' Imprimir integración STD2-RNAS
#'
#' @param x Objeto `rnas_std2_integration`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto x de forma invisible.
#'
#' @export
print.rnas_std2_integration <- function(x, ...) {
  cat("Integracion STD2-RNAS\n")
  cat("---------------------\n")
  cat("Target       :", x$configuracion$target, "\n")
  cat("Horizonte    :", x$configuracion$horizonte, "\n")
  cat("Loss inicial :", x$metricas$loss_inicial, "\n")
  cat("Loss final   :", x$metricas$loss_final, "\n")

  invisible(x)
}


#' Imprimir integración SimuDS-RNAS
#'
#' @param x Objeto `rnas_simuds_integration`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto x de forma invisible.
#'
#' @export
print.rnas_simuds_integration <- function(x, ...) {
  cat("Integracion SimuDS-RNAS\n")
  cat("-----------------------\n")
  cat("Target estado:", x$configuracion$target_estado, "\n")
  cat("Horizonte    :", x$configuracion$horizonte, "\n")
  cat("Loss inicial :", x$metricas$loss_inicial, "\n")
  cat("Loss final   :", x$metricas$loss_final, "\n")

  invisible(x)
}


#' Imprimir formulación FNL-RNAS
#'
#' @param x Objeto `rnas_fnl_formulation`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto x de forma invisible.
#'
#' @export
print.rnas_fnl_formulation <- function(x, ...) {
  cat("Formulacion FNL-RNAS\n")
  cat("--------------------\n")
  cat("Observaciones   :", x$configuracion$n_obs, "\n")
  cat("Features        :", x$configuracion$n_features, "\n")
  cat("Restricciones   :", x$configuracion$n_restricciones, "\n")
  cat("Activacion      :", x$configuracion$activation, "\n")

  invisible(x)
}
