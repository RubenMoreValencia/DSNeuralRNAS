#' Aplicar cotas a una tasa de aprendizaje RNAS
#'
#' Controla que una tasa de aprendizaje permanezca dentro de los limites
#' definidos por eta_min y eta_max.
#'
#' @param eta Escalar numerico de tasa de aprendizaje.
#' @param eta_min Cota inferior positiva.
#' @param eta_max Cota superior positiva.
#'
#' @return Tasa de aprendizaje acotada.
#'
#' @examples
#' rnas_clip_eta(eta = 0.5, eta_min = 0.01, eta_max = 0.1)
#'
#' @export
rnas_clip_eta <- function(eta,
                          eta_min = 1e-5,
                          eta_max = 1) {
  if (!is.numeric(eta) || length(eta) != 1L || !is.finite(eta)) {
    stop("`eta` debe ser un escalar numerico finito.")
  }

  if (!is.numeric(eta_min) || length(eta_min) != 1L ||
      eta_min <= 0 || !is.finite(eta_min)) {
    stop("`eta_min` debe ser un escalar numerico positivo y finito.")
  }

  if (!is.numeric(eta_max) || length(eta_max) != 1L ||
      eta_max <= 0 || !is.finite(eta_max)) {
    stop("`eta_max` debe ser un escalar numerico positivo y finito.")
  }

  if (eta_min > eta_max) {
    stop("`eta_min` no puede ser mayor que `eta_max`.")
  }

  min(max(eta, eta_min), eta_max)
}


#' Evaluar politica de tasa de aprendizaje RNAS
#'
#' Evalua una politica de control de tasa de aprendizaje a partir de senales
#' dinamicas del entrenamiento. Las politicas disponibles son:
#' constante, temporal, mejora, regimen e inestabilidad.
#'
#' @param policy Nombre de politica o lista con campo `tipo`.
#' @param k Iteracion actual.
#' @param eta_prev Tasa de aprendizaje previa.
#' @param eta0 Tasa inicial o base.
#' @param loss Perdida actual.
#' @param loss_prev Perdida previa. Puede ser `NA` en la primera iteracion.
#' @param grad_norm Norma del gradiente actual.
#' @param speed Velocidad dinamica actual.
#' @param regimen Regimen dinamico actual.
#' @param eta_min Cota inferior de tasa.
#' @param eta_max Cota superior de tasa.
#' @param eps Escalar positivo para estabilizar divisiones.
#'
#' @return Lista con `eta`, `tipo`, `accion` y `reduccion_relativa`.
#'
#' @examples
#' rnas_eta_policy(
#'   policy = "temporal",
#'   k = 10,
#'   eta_prev = 0.1,
#'   eta0 = 0.1,
#'   loss = 0.5,
#'   loss_prev = 0.6
#' )
#'
#' @export
rnas_eta_policy <- function(policy = "constante",
                            k,
                            eta_prev,
                            eta0,
                            loss = NA_real_,
                            loss_prev = NA_real_,
                            grad_norm = NA_real_,
                            speed = NA_real_,
                            regimen = NA_character_,
                            eta_min = 1e-5,
                            eta_max = 1,
                            eps = 1e-12) {
  if (!is.numeric(k) || length(k) != 1L || k < 0 || k != as.integer(k)) {
    stop("`k` debe ser un entero no negativo.")
  }

  if (!is.numeric(eta_prev) || length(eta_prev) != 1L || eta_prev <= 0) {
    stop("`eta_prev` debe ser un escalar numerico positivo.")
  }

  if (!is.numeric(eta0) || length(eta0) != 1L || eta0 <= 0) {
    stop("`eta0` debe ser un escalar numerico positivo.")
  }

  if (!is.numeric(eps) || length(eps) != 1L || eps <= 0) {
    stop("`eps` debe ser un escalar numerico positivo.")
  }

  if (is.list(policy)) {
    tipo <- policy$tipo

    if (is.null(tipo)) {
      stop("Si `policy` es lista, debe contener el campo `tipo`.")
    }

    cfg <- policy
  } else if (is.character(policy) && length(policy) == 1L) {
    tipo <- policy
    cfg <- list(tipo = tipo)
  } else {
    stop("`policy` debe ser texto o lista con campo `tipo`.")
  }

  tipo <- tolower(tipo)

  eta_new <- eta_prev
  accion <- "mantener"

  reduccion_relativa <- NA_real_

  if (!is.na(loss) && !is.na(loss_prev)) {
    reduccion_relativa <- (loss_prev - loss) / (abs(loss_prev) + eps)
  }

  if (tipo == "constante") {
    eta_new <- eta0
    accion <- "constante"

  } else if (tipo == "temporal") {
    alpha <- cfg$alpha
    if (is.null(alpha)) alpha <- 0.01

    if (!is.numeric(alpha) || length(alpha) != 1L || alpha < 0) {
      stop("`alpha` debe ser un escalar numerico no negativo.")
    }

    eta_new <- eta0 / (1 + alpha * k)
    accion <- "decaimiento_temporal"

  } else if (tipo == "mejora") {
    tau_loss <- cfg$tau_loss
    gamma <- cfg$gamma

    if (is.null(tau_loss)) tau_loss <- 0.01
    if (is.null(gamma)) gamma <- 0.95

    if (!is.numeric(tau_loss) || length(tau_loss) != 1L || tau_loss < 0) {
      stop("`tau_loss` debe ser un escalar numerico no negativo.")
    }

    if (!is.numeric(gamma) || length(gamma) != 1L || gamma <= 0 || gamma > 1) {
      stop("`gamma` debe estar en el intervalo (0, 1].")
    }

    if (is.na(reduccion_relativa) || reduccion_relativa > tau_loss) {
      eta_new <- eta_prev
      accion <- "mantener_por_mejora"
    } else {
      eta_new <- gamma * eta_prev
      accion <- "reducir_por_baja_mejora"
    }

  } else if (tipo == "regimen") {
    gamma_ref <- cfg$gamma_refinamiento
    gamma_sat <- cfg$gamma_saturacion
    gamma_ine <- cfg$gamma_inestabilidad
    gamma_est <- cfg$gamma_estabilizacion

    if (is.null(gamma_ref)) gamma_ref <- 0.98
    if (is.null(gamma_sat)) gamma_sat <- 0.90
    if (is.null(gamma_ine)) gamma_ine <- 0.50
    if (is.null(gamma_est)) gamma_est <- 0.95

    gammas <- c(gamma_ref, gamma_sat, gamma_ine, gamma_est)

    if (any(!is.numeric(gammas)) || any(gammas <= 0) || any(gammas > 1)) {
      stop("Los factores gamma de regimen deben estar en el intervalo (0, 1].")
    }

    if (is.na(regimen)) {
      eta_new <- eta_prev
      accion <- "sin_regimen"
    } else if (regimen == "Descenso dirigido") {
      eta_new <- eta_prev
      accion <- "mantener_descenso"
    } else if (regimen == "Refinamiento") {
      eta_new <- gamma_ref * eta_prev
      accion <- "reducir_refinamiento"
    } else if (regimen == "Saturacion") {
      eta_new <- gamma_sat * eta_prev
      accion <- "reducir_saturacion"
    } else if (regimen == "Inestabilidad") {
      eta_new <- gamma_ine * eta_prev
      accion <- "reducir_inestabilidad"
    } else if (regimen == "Estabilizacion") {
      eta_new <- gamma_est * eta_prev
      accion <- "reducir_estabilizacion"
    } else {
      eta_new <- eta_prev
      accion <- "mantener_otro_regimen"
    }

  } else if (tipo == "inestabilidad") {
    gamma <- cfg$gamma

    if (is.null(gamma)) gamma <- 0.5

    if (!is.numeric(gamma) || length(gamma) != 1L || gamma <= 0 || gamma > 1) {
      stop("`gamma` debe estar en el intervalo (0, 1].")
    }

    if (!is.na(loss) && !is.na(loss_prev) && loss > loss_prev) {
      eta_new <- gamma * eta_prev
      accion <- "reducir_por_aumento_loss"
    } else {
      eta_new <- eta_prev
      accion <- "mantener_sin_inestabilidad"
    }

  } else {
    stop("Politica no reconocida. Use: constante, temporal, mejora, regimen o inestabilidad.")
  }

  eta_new <- rnas_clip_eta(
    eta = eta_new,
    eta_min = eta_min,
    eta_max = eta_max
  )

  list(
    eta = eta_new,
    tipo = tipo,
    accion = accion,
    reduccion_relativa = reduccion_relativa,
    grad_norm = grad_norm,
    speed = speed,
    regimen = regimen
  )
}


#' Entrenar neurona RNAS con control de tasa de aprendizaje
#'
#' Ejecuta entrenamiento de una neurona individual usando una tasa de
#' aprendizaje controlada por una politica. La regla de actualizacion es:
#' \eqn{\theta_{k+1} = \theta_k - \eta_k \nabla L(\theta_k)}.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta0 Vector inicial de estado con pesos y sesgo.
#' @param eta0 Tasa inicial positiva.
#' @param policy Politica de tasa. Texto o lista con campo `tipo`.
#' @param eta_min Cota inferior de tasa.
#' @param eta_max Cota superior de tasa.
#' @param T Numero entero positivo de iteraciones.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#' @param ventana_regimen Ventana usada para clasificar regimenes parciales.
#' @param tau_loss Umbral de reduccion relativa.
#' @param tau_grad Umbral de gradiente activo.
#' @param eps_loss Tolerancia de perdida.
#' @param eps_grad Tolerancia de gradiente.
#'
#' @return Objeto de clase `rnas_control_eta_train`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta0 <- c(0.8, 0.3, 0.1)
#' res <- rnas_train_neuron_control_eta(X, y, theta0, eta0 = 0.1, T = 20)
#' res$metricas
#'
#' @details
#' La politica de control puede definirse como texto o como lista.
#' Cuando se usa una lista, pueden especificarse factores multiplicativos
#' gamma para regular la tasa de aprendizaje. Estos factores modifican
#' eta segun la regla general eta[k + 1] = gamma[k] * eta[k].
#'
#' Para politica por regimen, se admiten campos como:
#' `gamma_refinamiento`, `gamma_saturacion`,
#' `gamma_inestabilidad` y `gamma_estabilizacion`.
#' Para politica por mejora relativa, se admite `gamma`.
#' @export
rnas_train_neuron_control_eta <- function(X,
                                          y,
                                          theta0,
                                          eta0 = 0.1,
                                          policy = "constante",
                                          eta_min = 1e-5,
                                          eta_max = 1,
                                          T = 100,
                                          activation = "tanh",
                                          ventana_regimen = 5L,
                                          tau_loss = 0.01,
                                          tau_grad = 1e-3,
                                          eps_loss = 1e-8,
                                          eps_grad = 1e-4) {
  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (nrow(X) != length(y)) {
    stop("La longitud de `y` debe coincidir con el numero de filas de `X`.")
  }

  if (!is.numeric(theta0) || !is.vector(theta0)) {
    stop("`theta0` debe ser un vector numerico.")
  }

  if (length(theta0) != ncol(X) + 1L) {
    stop("La longitud de `theta0` debe ser igual a ncol(X) + 1.")
  }

  if (!is.numeric(eta0) || length(eta0) != 1L || eta0 <= 0) {
    stop("`eta0` debe ser un escalar numerico positivo.")
  }

  if (!is.numeric(T) || length(T) != 1L || T < 1 || T != as.integer(T)) {
    stop("`T` debe ser un entero positivo.")
  }

  eta_actual <- rnas_clip_eta(eta0, eta_min = eta_min, eta_max = eta_max)
  theta <- as.numeric(theta0)

  trayectoria <- vector("list", T + 1L)

  loss_prev <- NA_real_
  regimen_actual <- "Ajuste inicial"

  for (k in 0:T) {
    pars <- rnas_unpack_neuron_params(theta, d_input = ncol(X))

    grad <- rnas_grad_neuron(
      X = X,
      y = y,
      w = pars$w,
      b = pars$b,
      activation = activation
    )

    grad_vec <- c(grad$grad_w, grad$grad_b)
    grad_norm <- sqrt(sum(grad_vec^2))

    speed <- eta_actual * grad_norm

    # Clasificacion provisional del regimen con la informacion ya disponible.
    if (k == 0) {
      regimen_actual <- "Ajuste inicial"
    } else {
      temp_tray <- do.call(
        rbind,
        trayectoria[!vapply(trayectoria, is.null, logical(1))]
      )

      if (!is.null(temp_tray) && nrow(temp_tray) >= 2) {
        sen <- rnas_calcular_senales_regimen(
          trayectoria = temp_tray,
          ventana = min(ventana_regimen, nrow(temp_tray))
        )

        reg <- rnas_clasificar_regimenes(
          senales = sen,
          tau_loss = tau_loss,
          tau_grad = tau_grad,
          eps_loss = eps_loss,
          eps_grad = eps_grad,
          usar_suavizado = TRUE
        )

        regimen_actual <- tail(reg$regimen, 1)
      } else {
        regimen_actual <- "Descenso dirigido"
      }
    }

    pol <- rnas_eta_policy(
      policy = policy,
      k = k,
      eta_prev = eta_actual,
      eta0 = eta0,
      loss = grad$loss,
      loss_prev = loss_prev,
      grad_norm = grad_norm,
      speed = speed,
      regimen = regimen_actual,
      eta_min = eta_min,
      eta_max = eta_max
    )

    eta_k <- pol$eta
    speed_k <- eta_k * grad_norm

    fila <- data.frame(
      iter = k,
      loss = grad$loss,
      grad_norm = grad_norm,
      eta = eta_k,
      speed = speed_k,
      regimen = regimen_actual,
      accion_eta = pol$accion,
      reduccion_relativa = pol$reduccion_relativa,
      b = pars$b,
      stringsAsFactors = FALSE
    )

    for (j in seq_along(pars$w)) {
      fila[[paste0("w", j)]] <- pars$w[j]
    }

    for (j in seq_along(theta)) {
      fila[[paste0("theta", j)]] <- theta[j]
    }

    trayectoria[[k + 1L]] <- fila

    loss_prev <- grad$loss
    eta_actual <- eta_k

    if (k < T) {
      theta <- theta - eta_k * grad_vec
    }
  }

  trayectoria_df <- do.call(rbind, trayectoria)
  rownames(trayectoria_df) <- NULL

  pars_final <- rnas_unpack_neuron_params(theta, d_input = ncol(X))

  loss_inicial <- trayectoria_df$loss[1]
  loss_final <- trayectoria_df$loss[nrow(trayectoria_df)]
  delta_loss <- loss_final - loss_inicial
  reduccion_abs <- loss_inicial - loss_final
  reduccion_rel <- if (loss_inicial != 0) reduccion_abs / abs(loss_inicial) else NA_real_

  eta_cambios <- sum(abs(diff(trayectoria_df$eta)) > .Machine$double.eps)

  y_hat_final <- rnas_neuron_forward_batch(
    X = X,
    w = pars_final$w,
    b = pars_final$b,
    activation = activation
  )

  res <- list(
    theta_final = theta,
    w_final = pars_final$w,
    b_final = pars_final$b,
    y_hat_final = y_hat_final,
    trayectoria = trayectoria_df,
    metricas = list(
      loss_inicial = loss_inicial,
      loss_final = loss_final,
      delta_loss = delta_loss,
      reduccion_abs = reduccion_abs,
      reduccion_rel = reduccion_rel,
      eta_min_obs = min(trayectoria_df$eta),
      eta_max_obs = max(trayectoria_df$eta),
      eta_media = mean(trayectoria_df$eta),
      eta_cambios = eta_cambios,
      velocidad_media = mean(trayectoria_df$speed),
      grad_norm_inicial = trayectoria_df$grad_norm[1],
      grad_norm_final = trayectoria_df$grad_norm[nrow(trayectoria_df)],
      descendente_global = isTRUE(loss_final < loss_inicial)
    ),
    configuracion = list(
      eta0 = eta0,
      policy = policy,
      eta_min = eta_min,
      eta_max = eta_max,
      T = T,
      activation = activation,
      ventana_regimen = ventana_regimen,
      tau_loss = tau_loss,
      tau_grad = tau_grad,
      eps_loss = eps_loss,
      eps_grad = eps_grad
    )
  )

  class(res) <- c("rnas_control_eta_train", class(res))
  res
}


#' Resumir entrenamiento con control de tasa RNAS
#'
#' Genera una tabla resumen de un objeto producido por
#' `rnas_train_neuron_control_eta()`.
#'
#' @param object Objeto de clase `rnas_control_eta_train`.
#' @param nombre_politica Nombre opcional de la politica.
#'
#' @return Data frame con metricas principales.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta0 <- c(0.8, 0.3, 0.1)
#' res <- rnas_train_neuron_control_eta(X, y, theta0, T = 20)
#' rnas_resumen_control_eta(res)
#'
#' @export
rnas_resumen_control_eta <- function(object,
                                     nombre_politica = NULL) {
  if (!inherits(object, "rnas_control_eta_train")) {
    stop("`object` debe ser un objeto de clase 'rnas_control_eta_train'.")
  }

  if (is.null(nombre_politica)) {
    pol <- object$configuracion$policy
    nombre_politica <- if (is.list(pol)) pol$tipo else pol
  }

  data.frame(
    politica = nombre_politica,
    activation = object$configuracion$activation,
    T_configurado = object$configuracion$T,
    loss_inicial = object$metricas$loss_inicial,
    loss_final = object$metricas$loss_final,
    delta_loss = object$metricas$delta_loss,
    reduccion_abs = object$metricas$reduccion_abs,
    reduccion_rel = object$metricas$reduccion_rel,
    eta_min_obs = object$metricas$eta_min_obs,
    eta_max_obs = object$metricas$eta_max_obs,
    eta_media = object$metricas$eta_media,
    eta_cambios = object$metricas$eta_cambios,
    velocidad_media = object$metricas$velocidad_media,
    grad_norm_inicial = object$metricas$grad_norm_inicial,
    grad_norm_final = object$metricas$grad_norm_final,
    descendente_global = object$metricas$descendente_global,
    stringsAsFactors = FALSE
  )
}


#' Comparar politicas de tasa de aprendizaje RNAS
#'
#' Ejecuta varias politicas de control de tasa de aprendizaje sobre el mismo
#' caso y devuelve una tabla comparativa de metricas.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta0 Vector inicial con pesos y sesgo.
#' @param politicas Lista nombrada de politicas. Cada elemento puede ser texto
#' o lista con campo `tipo`.
#' @param eta0 Tasa inicial.
#' @param eta_min Cota inferior de tasa.
#' @param eta_max Cota superior de tasa.
#' @param T Numero de iteraciones.
#' @param activation Activacion de la neurona.
#'
#' @return Objeto de clase `rnas_eta_policy_comparison`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta0 <- c(0.8, 0.3, 0.1)
#' politicas <- list(base = "constante", temporal = list(tipo = "temporal"))
#' comp <- rnas_comparar_politicas_eta(X, y, theta0, politicas, T = 20)
#' comp$comparacion
#'
#' @export
rnas_comparar_politicas_eta <- function(X,
                                        y,
                                        theta0,
                                        politicas,
                                        eta0 = 0.1,
                                        eta_min = 1e-5,
                                        eta_max = 1,
                                        T = 100,
                                        activation = "tanh") {
  if (!is.list(politicas) || length(politicas) == 0) {
    stop("`politicas` debe ser una lista no vacia.")
  }

  nombres <- names(politicas)

  if (is.null(nombres) || any(nombres == "")) {
    nombres <- paste0("politica_", seq_along(politicas))
  }

  modelos <- vector("list", length(politicas))
  resumenes <- vector("list", length(politicas))

  for (i in seq_along(politicas)) {
    nm <- nombres[i]

    mod <- rnas_train_neuron_control_eta(
      X = X,
      y = y,
      theta0 = theta0,
      eta0 = eta0,
      policy = politicas[[i]],
      eta_min = eta_min,
      eta_max = eta_max,
      T = T,
      activation = activation
    )

    modelos[[i]] <- mod
    resumenes[[i]] <- rnas_resumen_control_eta(
      object = mod,
      nombre_politica = nm
    )
  }

  comparacion <- do.call(rbind, resumenes)
  rownames(comparacion) <- NULL

  res <- list(
    comparacion = comparacion,
    modelos = stats::setNames(modelos, nombres),
    configuracion = list(
      eta0 = eta0,
      eta_min = eta_min,
      eta_max = eta_max,
      T = T,
      activation = activation,
      politicas = politicas
    )
  )

  class(res) <- c("rnas_eta_policy_comparison", class(res))
  res
}


#' Imprimir entrenamiento con control de tasa RNAS
#'
#' Metodo de impresion para objetos `rnas_control_eta_train`.
#'
#' @param x Objeto de clase `rnas_control_eta_train`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto `x`, de forma invisible.
#'
#' @export
print.rnas_control_eta_train <- function(x, ...) {
  pol <- x$configuracion$policy
  pol_nombre <- if (is.list(pol)) pol$tipo else pol

  cat("Entrenamiento RNAS con control de eta\n")
  cat("-------------------------------------\n")
  cat("Politica         :", pol_nombre, "\n")
  cat("Eta inicial      :", x$configuracion$eta0, "\n")
  cat("Eta media        :", x$metricas$eta_media, "\n")
  cat("Cambios de eta   :", x$metricas$eta_cambios, "\n")
  cat("Loss inicial     :", x$metricas$loss_inicial, "\n")
  cat("Loss final       :", x$metricas$loss_final, "\n")
  cat("Reduccion rel.   :", x$metricas$reduccion_rel, "\n")
  cat("Descendente      :", x$metricas$descendente_global, "\n")

  invisible(x)
}


#' Imprimir comparacion de politicas de eta RNAS
#'
#' Metodo de impresion para objetos `rnas_eta_policy_comparison`.
#'
#' @param x Objeto de clase `rnas_eta_policy_comparison`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto `x`, de forma invisible.
#'
#' @export
print.rnas_eta_policy_comparison <- function(x, ...) {
  cat("Comparacion de politicas de tasa de aprendizaje RNAS\n")
  cat("----------------------------------------------------\n")
  cat("Politicas evaluadas :", nrow(x$comparacion), "\n")
  cat("Eta inicial         :", x$configuracion$eta0, "\n")
  cat("Iteraciones         :", x$configuracion$T, "\n")

  invisible(x)
}
