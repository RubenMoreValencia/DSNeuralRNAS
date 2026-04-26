#' Calcular media movil RNAS
#'
#' Calcula una media movil simple preservando la longitud del vector original.
#' Para los primeros valores, utiliza las observaciones disponibles hasta el
#' momento, evitando producir valores NA al inicio.
#'
#' @param x Vector numerico.
#' @param ventana Entero positivo que define el tamano de la ventana.
#'
#' @return Vector numerico suavizado con la misma longitud que `x`.
#'
#' @examples
#' rnas_media_movil(c(1, 2, 3, 4, 5), ventana = 3)
#'
#' @export
rnas_media_movil <- function(x, ventana = 3L) {
  if (!is.numeric(x) || !is.vector(x)) {
    stop("`x` debe ser un vector numerico.")
  }

  if (!is.numeric(ventana) ||
      length(ventana) != 1L ||
      ventana < 1 ||
      ventana != as.integer(ventana)) {
    stop("`ventana` debe ser un entero positivo.")
  }

  n <- length(x)

  if (n == 0) {
    return(numeric(0))
  }

  out <- numeric(n)

  for (i in seq_len(n)) {
    ini <- max(1L, i - ventana + 1L)
    out[i] <- mean(x[ini:i], na.rm = TRUE)
  }

  out
}


#' Calcular senales dinamicas para regimenes RNAS
#'
#' Agrega a una trayectoria columnas dinamicas derivadas de la perdida,
#' norma del gradiente y velocidad. Las senales calculadas incluyen cambio
#' de perdida, reduccion relativa, cambio de gradiente y cambio de velocidad.
#'
#' @param trayectoria Data frame con columnas de perdida, gradiente y velocidad.
#' @param loss_col Nombre de la columna de perdida.
#' @param grad_col Nombre de la columna de norma del gradiente.
#' @param speed_col Nombre de la columna de velocidad.
#' @param ventana Entero positivo para suavizado por media movil.
#' @param eps Escalar positivo para estabilizar divisiones.
#'
#' @return Data frame con senales dinamicas agregadas.
#'
#' @examples
#' tray <- data.frame(
#'   iter = 0:3,
#'   loss = c(1, 0.8, 0.7, 0.69),
#'   grad_norm = c(0.5, 0.3, 0.2, 0.1),
#'   speed = c(0.05, 0.03, 0.02, 0.01)
#' )
#' rnas_calcular_senales_regimen(tray)
#'
#' @export
rnas_calcular_senales_regimen <- function(trayectoria,
                                          loss_col = "loss",
                                          grad_col = "grad_norm",
                                          speed_col = "speed",
                                          ventana = 3L,
                                          eps = 1e-12) {
  if (!is.data.frame(trayectoria)) {
    stop("`trayectoria` debe ser un data frame.")
  }

  req <- c(loss_col, grad_col, speed_col)

  if (!all(req %in% names(trayectoria))) {
    stop("La trayectoria debe contener las columnas indicadas en loss_col, grad_col y speed_col.")
  }

  if (!is.numeric(ventana) ||
      length(ventana) != 1L ||
      ventana < 1 ||
      ventana != as.integer(ventana)) {
    stop("`ventana` debe ser un entero positivo.")
  }

  if (!is.numeric(eps) || length(eps) != 1L || eps <= 0) {
    stop("`eps` debe ser un escalar numerico positivo.")
  }

  out <- trayectoria

  loss <- as.numeric(out[[loss_col]])
  grad <- as.numeric(out[[grad_col]])
  speed <- as.numeric(out[[speed_col]])

  if (any(!is.finite(loss)) || any(!is.finite(grad)) || any(!is.finite(speed))) {
    stop("Las columnas de perdida, gradiente y velocidad deben contener valores finitos.")
  }

  n <- nrow(out)

  delta_loss <- c(NA_real_, diff(loss))
  reduccion_relativa <- c(
    NA_real_,
    (loss[-n] - loss[-1]) / (abs(loss[-n]) + eps)
  )

  delta_grad <- c(NA_real_, diff(grad))
  delta_speed <- c(NA_real_, diff(speed))

  out$loss_suav <- rnas_media_movil(loss, ventana = ventana)
  out$grad_suav <- rnas_media_movil(grad, ventana = ventana)
  out$speed_suav <- rnas_media_movil(speed, ventana = ventana)

  out$delta_loss <- delta_loss
  out$reduccion_relativa <- reduccion_relativa
  out$delta_grad <- delta_grad
  out$delta_speed <- delta_speed

  out$delta_loss_suav <- c(NA_real_, diff(out$loss_suav))
  out$delta_grad_suav <- c(NA_real_, diff(out$grad_suav))
  out$delta_speed_suav <- c(NA_real_, diff(out$speed_suav))

  out
}


#' Clasificar regimenes dinamicos de aprendizaje RNAS
#'
#' Asigna una etiqueta de regimen dinamico a cada fila de una trayectoria con
#' senales calculadas. Las reglas consideran reduccion relativa de perdida,
#' gradiente, velocidad e incremento de perdida.
#'
#' @param senales Data frame producido por `rnas_calcular_senales_regimen()`.
#' @param tau_loss Umbral de reduccion relativa relevante.
#' @param tau_grad Umbral de gradiente activo.
#' @param eps_loss Tolerancia para cambio pequeno de perdida.
#' @param eps_grad Tolerancia para gradiente pequeno.
#' @param tau_speed Umbral para incremento abrupto de velocidad.
#' @param loss_alta Umbral de perdida para distinguir saturacion.
#' Si es `NULL`, se calcula como la mediana de la perdida.
#' @param usar_suavizado Valor logico. Si es `TRUE`, clasifica usando senales suavizadas.
#'
#' @return Data frame con columna `regimen`.
#'
#' @examples
#' tray <- data.frame(
#'   iter = 0:3,
#'   loss = c(1, 0.8, 0.7, 0.69),
#'   grad_norm = c(0.5, 0.3, 0.2, 0.1),
#'   speed = c(0.05, 0.03, 0.02, 0.01)
#' )
#' sen <- rnas_calcular_senales_regimen(tray)
#' rnas_clasificar_regimenes(sen)
#'
#' @export
rnas_clasificar_regimenes <- function(senales,
                                      tau_loss = 0.01,
                                      tau_grad = 1e-3,
                                      eps_loss = 1e-7,
                                      eps_grad = 1e-4,
                                      tau_speed = Inf,
                                      loss_alta = NULL,
                                      usar_suavizado = TRUE) {
  if (!is.data.frame(senales)) {
    stop("`senales` debe ser un data frame.")
  }

  req <- c("loss", "grad_norm", "speed", "delta_loss", "reduccion_relativa",
           "delta_speed", "loss_suav", "grad_suav", "speed_suav",
           "delta_loss_suav", "delta_speed_suav")

  if (!all(req %in% names(senales))) {
    stop("`senales` debe provenir de rnas_calcular_senales_regimen().")
  }

  umbrales <- list(
    tau_loss = tau_loss,
    tau_grad = tau_grad,
    eps_loss = eps_loss,
    eps_grad = eps_grad
  )

  for (nm in names(umbrales)) {
    val <- umbrales[[nm]]

    if (!is.numeric(val) || length(val) != 1L || val < 0) {
      stop(paste0("`", nm, "` debe ser un escalar numerico no negativo."))
    }
  }

  if (!is.numeric(tau_speed) || length(tau_speed) != 1L || tau_speed < 0) {
    stop("`tau_speed` debe ser un escalar numerico no negativo o Inf.")
  }

  if (!is.null(loss_alta) &&
      (!is.numeric(loss_alta) || length(loss_alta) != 1L || loss_alta < 0)) {
    stop("`loss_alta` debe ser NULL o un escalar numerico no negativo.")
  }

  if (!is.logical(usar_suavizado) || length(usar_suavizado) != 1L) {
    stop("`usar_suavizado` debe ser TRUE o FALSE.")
  }

  out <- senales

  loss_base <- if (isTRUE(usar_suavizado)) out$loss_suav else out$loss
  grad_base <- if (isTRUE(usar_suavizado)) out$grad_suav else out$grad_norm
  speed_base <- if (isTRUE(usar_suavizado)) out$speed_suav else out$speed
  delta_loss_base <- if (isTRUE(usar_suavizado)) out$delta_loss_suav else out$delta_loss
  delta_speed_base <- if (isTRUE(usar_suavizado)) out$delta_speed_suav else out$delta_speed

  reduccion_rel <- out$reduccion_relativa

  if (is.null(loss_alta)) {
    loss_alta <- stats::median(out$loss, na.rm = TRUE)
  }

  regimen <- character(nrow(out))

  for (i in seq_len(nrow(out))) {
    if (i == 1L || is.na(delta_loss_base[i])) {
      regimen[i] <- "Ajuste inicial"
      next
    }

    dl <- delta_loss_base[i]
    rr <- reduccion_rel[i]
    g <- grad_base[i]
    v <- speed_base[i]
    dv <- delta_speed_base[i]
    l <- loss_base[i]

    if (is.na(rr)) {
      rr <- 0
    }

    if (is.na(dv)) {
      dv <- 0
    }

    if (dl > eps_loss || dv > tau_speed) {
      regimen[i] <- "Inestabilidad"
    } else if (abs(dl) < eps_loss && g < eps_grad) {
      regimen[i] <- "Estabilizacion"
    } else if (abs(dl) < eps_loss && l > loss_alta) {
      regimen[i] <- "Saturacion"
    } else if (dl < 0 && rr > tau_loss && g > tau_grad) {
      regimen[i] <- "Descenso dirigido"
    } else if (dl < 0) {
      regimen[i] <- "Refinamiento"
    } else {
      regimen[i] <- "Transicion"
    }

    if (!is.finite(v)) {
      regimen[i] <- "Transicion"
    }
  }

  out$regimen <- regimen
  out
}


#' Segmentar regimenes dinamicos consecutivos
#'
#' Agrupa tramos consecutivos de una trayectoria que comparten la misma
#' etiqueta de regimen.
#'
#' @param trayectoria_regimen Data frame con columna `regimen`.
#' @param iter_col Nombre de la columna de iteracion.
#' @param loss_col Nombre de la columna de perdida.
#' @param grad_col Nombre de la columna de gradiente.
#' @param speed_col Nombre de la columna de velocidad.
#'
#' @return Data frame con segmentos consecutivos de regimen.
#'
#' @examples
#' df <- data.frame(
#'   iter = 0:4,
#'   loss = c(1, .8, .7, .69, .68),
#'   grad_norm = c(.5, .4, .3, .2, .1),
#'   speed = c(.05, .04, .03, .02, .01),
#'   regimen = c("A", "B", "B", "C", "C")
#' )
#' rnas_segmentar_regimenes(df)
#'
#' @export
rnas_segmentar_regimenes <- function(trayectoria_regimen,
                                     iter_col = "iter",
                                     loss_col = "loss",
                                     grad_col = "grad_norm",
                                     speed_col = "speed") {
  if (!is.data.frame(trayectoria_regimen)) {
    stop("`trayectoria_regimen` debe ser un data frame.")
  }

  req <- c(iter_col, loss_col, grad_col, speed_col, "regimen")

  if (!all(req %in% names(trayectoria_regimen))) {
    stop("La trayectoria debe contener iteracion, perdida, gradiente, velocidad y regimen.")
  }

  n <- nrow(trayectoria_regimen)

  if (n == 0) {
    return(data.frame())
  }

  regimen <- as.character(trayectoria_regimen$regimen)
  cambios <- c(TRUE, regimen[-1] != regimen[-n])
  grupo <- cumsum(cambios)

  segmentos <- split(trayectoria_regimen, grupo)

  filas <- vector("list", length(segmentos))

  for (i in seq_along(segmentos)) {
    seg <- segmentos[[i]]

    iter_ini <- seg[[iter_col]][1]
    iter_fin <- seg[[iter_col]][nrow(seg)]

    loss_ini <- seg[[loss_col]][1]
    loss_fin <- seg[[loss_col]][nrow(seg)]

    filas[[i]] <- data.frame(
      segmento = i,
      regimen = seg$regimen[1],
      iter_inicio = iter_ini,
      iter_fin = iter_fin,
      duracion = nrow(seg),
      loss_inicio = loss_ini,
      loss_fin = loss_fin,
      delta_loss = loss_fin - loss_ini,
      grad_prom = mean(seg[[grad_col]], na.rm = TRUE),
      speed_prom = mean(seg[[speed_col]], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, filas)
}


#' Resumir regimenes dinamicos de aprendizaje
#'
#' Calcula frecuencia, proporcion y estadisticos principales por regimen.
#'
#' @param trayectoria_regimen Data frame con columna `regimen`.
#' @param loss_col Nombre de la columna de perdida.
#' @param grad_col Nombre de la columna de gradiente.
#' @param speed_col Nombre de la columna de velocidad.
#'
#' @return Data frame resumen por regimen.
#'
#' @examples
#' df <- data.frame(
#'   loss = c(1, .8, .7),
#'   grad_norm = c(.5, .4, .3),
#'   speed = c(.05, .04, .03),
#'   regimen = c("A", "B", "B")
#' )
#' rnas_resumen_regimenes(df)
#'
#' @export
rnas_resumen_regimenes <- function(trayectoria_regimen,
                                   loss_col = "loss",
                                   grad_col = "grad_norm",
                                   speed_col = "speed") {
  if (!is.data.frame(trayectoria_regimen)) {
    stop("`trayectoria_regimen` debe ser un data frame.")
  }

  req <- c(loss_col, grad_col, speed_col, "regimen")

  if (!all(req %in% names(trayectoria_regimen))) {
    stop("La trayectoria debe contener perdida, gradiente, velocidad y regimen.")
  }

  n_total <- nrow(trayectoria_regimen)

  if (n_total == 0) {
    return(data.frame())
  }

  regs <- unique(trayectoria_regimen$regimen)

  filas <- vector("list", length(regs))

  for (i in seq_along(regs)) {
    reg <- regs[i]
    sub <- trayectoria_regimen[trayectoria_regimen$regimen == reg, , drop = FALSE]

    filas[[i]] <- data.frame(
      regimen = reg,
      frecuencia = nrow(sub),
      proporcion = nrow(sub) / n_total,
      loss_media = mean(sub[[loss_col]], na.rm = TRUE),
      loss_min = min(sub[[loss_col]], na.rm = TRUE),
      loss_max = max(sub[[loss_col]], na.rm = TRUE),
      grad_media = mean(sub[[grad_col]], na.rm = TRUE),
      speed_media = mean(sub[[speed_col]], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, filas)
  out[order(out$frecuencia, decreasing = TRUE), , drop = FALSE]
}


#' Analizar regimenes dinamicos de una neurona RNAS
#'
#' Ejecuta el flujo completo de analisis de regimenes sobre un objeto dinamico
#' de neurona producido por `rnas_integrar_dinamica_neuron()`.
#'
#' @param object Objeto de clase `rnas_neuron_dynamics`.
#' @param ventana Ventana para suavizado.
#' @param tau_loss Umbral de reduccion relativa relevante.
#' @param tau_grad Umbral de gradiente activo.
#' @param eps_loss Tolerancia para cambio pequeno de perdida.
#' @param eps_grad Tolerancia para gradiente pequeno.
#' @param tau_speed Umbral para incremento abrupto de velocidad.
#' @param loss_alta Umbral de perdida para saturacion.
#' @param usar_suavizado Valor logico. Si es `TRUE`, clasifica usando senales suavizadas.
#'
#' @return Objeto de clase `rnas_regimen_analysis` con trayectoria,
#' segmentos, resumen y configuracion.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta0 <- c(0.8, 0.3, 0.1)
#' dyn <- rnas_integrar_dinamica_neuron(X, y, theta0, T = 20)
#' ana <- rnas_analizar_regimenes_neuron(dyn)
#' ana$resumen
#'
#' @export
rnas_analizar_regimenes_neuron <- function(object,
                                           ventana = 5L,
                                           tau_loss = 0.01,
                                           tau_grad = 1e-3,
                                           eps_loss = 1e-8,
                                           eps_grad = 1e-4,
                                           tau_speed = Inf,
                                           loss_alta = NULL,
                                           usar_suavizado = TRUE) {
  if (!inherits(object, "rnas_neuron_dynamics")) {
    stop("`object` debe ser un objeto de clase 'rnas_neuron_dynamics'.")
  }

  senales <- rnas_calcular_senales_regimen(
    trayectoria = object$trayectoria,
    loss_col = "loss",
    grad_col = "grad_norm",
    speed_col = "speed",
    ventana = ventana
  )

  trayectoria_regimen <- rnas_clasificar_regimenes(
    senales = senales,
    tau_loss = tau_loss,
    tau_grad = tau_grad,
    eps_loss = eps_loss,
    eps_grad = eps_grad,
    tau_speed = tau_speed,
    loss_alta = loss_alta,
    usar_suavizado = usar_suavizado
  )

  segmentos <- rnas_segmentar_regimenes(trayectoria_regimen)
  resumen <- rnas_resumen_regimenes(trayectoria_regimen)

  res <- list(
    trayectoria_regimen = trayectoria_regimen,
    segmentos = segmentos,
    resumen = resumen,
    configuracion = list(
      ventana = ventana,
      tau_loss = tau_loss,
      tau_grad = tau_grad,
      eps_loss = eps_loss,
      eps_grad = eps_grad,
      tau_speed = tau_speed,
      loss_alta = loss_alta,
      usar_suavizado = usar_suavizado
    ),
    dinamica = object
  )

  class(res) <- c("rnas_regimen_analysis", class(res))
  res
}


#' Imprimir resumen de analisis de regimenes RNAS
#'
#' Metodo de impresion para objetos creados con
#' `rnas_analizar_regimenes_neuron()`.
#'
#' @param x Objeto de clase `rnas_regimen_analysis`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto `x`, de forma invisible.
#'
#' @export
print.rnas_regimen_analysis <- function(x, ...) {
  cat("Analisis de regimenes dinamicos RNAS\n")
  cat("-----------------------------------\n")
  cat("Ventana suavizado :", x$configuracion$ventana, "\n")
  cat("Usa suavizado     :", x$configuracion$usar_suavizado, "\n")
  cat("Segmentos         :", nrow(x$segmentos), "\n")
  cat("Regimenes         :", paste(unique(x$trayectoria_regimen$regimen), collapse = ", "), "\n")

  invisible(x)
}
