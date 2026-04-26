#' Calcular costo local para una tasa candidata RNAS
#'
#' Evalua el costo local asociado a una tasa de aprendizaje candidata.
#' Para una tasa eta, se calcula una actualizacion tentativa:
#' theta_plus = theta - eta * grad,
#' luego se evalua la perdida tentativa y penalizaciones por tasa y desplazamiento.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta Vector numerico de estado actual.
#' @param grad Vector numerico de gradiente actual.
#' @param eta Tasa candidata positiva.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#' @param alpha Penalizacion por magnitud de eta.
#' @param beta Penalizacion por desplazamiento parametrico.
#'
#' @return Lista con costo total, perdida tentativa, penalizaciones y theta_plus.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta <- c(0.8, 0.3, 0.1)
#' grad <- c(-0.0007526453, -0.0101555834, -0.0114716749)
#' rnas_eta_costo_local(X, y, theta, grad, eta = 0.1)
#'
#' @export
rnas_eta_costo_local <- function(X,
                                 y,
                                 theta,
                                 grad,
                                 eta,
                                 activation = "tanh",
                                 alpha = 0,
                                 beta = 0) {
  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (nrow(X) != length(y)) {
    stop("La longitud de `y` debe coincidir con el numero de filas de `X`.")
  }

  if (!is.numeric(theta) || !is.vector(theta)) {
    stop("`theta` debe ser un vector numerico.")
  }

  if (length(theta) != ncol(X) + 1L) {
    stop("La longitud de `theta` debe ser igual a ncol(X) + 1.")
  }

  if (!is.numeric(grad) || !is.vector(grad)) {
    stop("`grad` debe ser un vector numerico.")
  }

  if (length(grad) != length(theta)) {
    stop("La longitud de `grad` debe coincidir con la longitud de `theta`.")
  }

  if (!is.numeric(eta) || length(eta) != 1L || eta <= 0 || !is.finite(eta)) {
    stop("`eta` debe ser un escalar numerico positivo y finito.")
  }

  if (!is.numeric(alpha) || length(alpha) != 1L || alpha < 0) {
    stop("`alpha` debe ser un escalar numerico no negativo.")
  }

  if (!is.numeric(beta) || length(beta) != 1L || beta < 0) {
    stop("`beta` debe ser un escalar numerico no negativo.")
  }

  theta_plus <- theta - eta * grad
  pars_plus <- rnas_unpack_neuron_params(theta_plus, d_input = ncol(X))

  loss_plus <- rnas_loss_mse_neuron(
    X = X,
    y = y,
    w = pars_plus$w,
    b = pars_plus$b,
    activation = activation
  )

  desplazamiento <- sqrt(sum((theta_plus - theta)^2))
  pen_eta <- alpha * eta^2
  pen_desplazamiento <- beta * desplazamiento^2

  costo <- loss_plus + pen_eta + pen_desplazamiento

  list(
    costo = as.numeric(costo),
    loss_plus = as.numeric(loss_plus),
    pen_eta = as.numeric(pen_eta),
    pen_desplazamiento = as.numeric(pen_desplazamiento),
    desplazamiento = as.numeric(desplazamiento),
    eta = as.numeric(eta),
    theta_plus = theta_plus
  )
}


#' Seleccionar tasa optima local por busqueda discreta RNAS
#'
#' Evalua una grilla de tasas candidatas y selecciona la que minimiza
#' el costo local.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta Vector numerico de estado actual.
#' @param grad Vector numerico de gradiente actual.
#' @param eta_grid Vector numerico de tasas candidatas.
#' @param activation Nombre de activacion.
#' @param alpha Penalizacion por magnitud de eta.
#' @param beta Penalizacion por desplazamiento parametrico.
#'
#' @return Lista con eta seleccionada, costo minimo y tabla de costos.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta <- c(0.8, 0.3, 0.1)
#' grad <- c(-0.0007526453, -0.0101555834, -0.0114716749)
#' rnas_eta_opt_local(X, y, theta, grad, eta_grid = c(0.01, 0.05, 0.1))
#'
#' @export
rnas_eta_opt_local <- function(X,
                               y,
                               theta,
                               grad,
                               eta_grid,
                               activation = "tanh",
                               alpha = 0,
                               beta = 0) {
  if (!is.numeric(eta_grid) || !is.vector(eta_grid) ||
      length(eta_grid) < 1L || any(eta_grid <= 0) || any(!is.finite(eta_grid))) {
    stop("`eta_grid` debe ser un vector numerico positivo y finito.")
  }

  costos <- vector("list", length(eta_grid))

  for (i in seq_along(eta_grid)) {
    res <- rnas_eta_costo_local(
      X = X,
      y = y,
      theta = theta,
      grad = grad,
      eta = eta_grid[i],
      activation = activation,
      alpha = alpha,
      beta = beta
    )

    costos[[i]] <- data.frame(
      eta = eta_grid[i],
      costo = res$costo,
      loss_plus = res$loss_plus,
      pen_eta = res$pen_eta,
      pen_desplazamiento = res$pen_desplazamiento,
      desplazamiento = res$desplazamiento,
      stringsAsFactors = FALSE
    )
  }

  costos_df <- do.call(rbind, costos)
  idx <- which.min(costos_df$costo)

  list(
    eta_opt = costos_df$eta[idx],
    costo_min = costos_df$costo[idx],
    loss_plus = costos_df$loss_plus[idx],
    costos = costos_df
  )
}


#' Evaluar politica geometrica de tasa de aprendizaje RNAS
#'
#' Calcula una tasa de aprendizaje ajustada por informacion geometrica.
#' Los modos disponibles son `curvatura`, `lambda_max` y `hibrida`.
#'
#' @param eta0 Tasa base positiva.
#' @param kappa Curvatura direccional.
#' @param lambda_max Autovalor maximo del Hessiano.
#' @param modo Tipo de politica: `"curvatura"`, `"lambda_max"` o `"hibrida"`.
#' @param alpha Sensibilidad geometrica.
#' @param eta_reg Tasa proveniente de una politica por regimen, usada en modo hibrido.
#' @param eta_min Cota inferior.
#' @param eta_max Cota superior.
#'
#' @return Lista con eta, modo, factor geometrico y senal usada.
#'
#' @examples
#' rnas_eta_policy_geo(eta0 = 0.1, kappa = 1.2, modo = "curvatura")
#'
#' @export
rnas_eta_policy_geo <- function(eta0,
                                kappa = NA_real_,
                                lambda_max = NA_real_,
                                modo = "curvatura",
                                alpha = 1,
                                eta_reg = NULL,
                                eta_min = 1e-5,
                                eta_max = 1) {
  if (!is.numeric(eta0) || length(eta0) != 1L || eta0 <= 0) {
    stop("`eta0` debe ser un escalar numerico positivo.")
  }

  if (!is.character(modo) || length(modo) != 1L) {
    stop("`modo` debe ser un texto.")
  }

  if (!is.numeric(alpha) || length(alpha) != 1L || alpha < 0) {
    stop("`alpha` debe ser un escalar numerico no negativo.")
  }

  modo <- tolower(modo)

  if (modo == "curvatura") {
    if (!is.numeric(kappa) || length(kappa) != 1L || !is.finite(kappa)) {
      stop("Para modo 'curvatura', `kappa` debe ser un escalar numerico finito.")
    }

    senal <- abs(kappa)
    eta_raw <- eta0 / (1 + alpha * senal)

  } else if (modo == "lambda_max") {
    if (!is.numeric(lambda_max) || length(lambda_max) != 1L || !is.finite(lambda_max)) {
      stop("Para modo 'lambda_max', `lambda_max` debe ser un escalar numerico finito.")
    }

    senal <- max(lambda_max, 0)
    eta_raw <- eta0 / (1 + alpha * senal)

  } else if (modo == "hibrida") {
    if (!is.numeric(kappa) || length(kappa) != 1L || !is.finite(kappa)) {
      stop("Para modo 'hibrida', `kappa` debe ser un escalar numerico finito.")
    }

    if (is.null(eta_reg)) {
      eta_reg <- eta0
    }

    if (!is.numeric(eta_reg) || length(eta_reg) != 1L || eta_reg <= 0) {
      stop("`eta_reg` debe ser un escalar numerico positivo.")
    }

    senal <- abs(kappa)
    eta_raw <- eta_reg / (1 + alpha * senal)

  } else {
    stop("`modo` debe ser 'curvatura', 'lambda_max' o 'hibrida'.")
  }

  eta <- rnas_clip_eta(
    eta = eta_raw,
    eta_min = eta_min,
    eta_max = eta_max
  )

  list(
    eta = eta,
    modo = modo,
    factor_geo = eta / eta0,
    senal = senal,
    eta_raw = eta_raw
  )
}


#' Entrenar neurona RNAS con meta-control geometrico
#'
#' Entrena una neurona individual usando una politica de meta-control
#' geometrico. Puede usar seleccion optima local sobre una grilla de tasas,
#' politica por curvatura o politica por autovalor maximo.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta0 Vector inicial de estado con pesos y sesgo.
#' @param eta0 Tasa base positiva.
#' @param metodo Metodo de control: `"opt_local"`, `"curvatura"` o `"lambda_max"`.
#' @param eta_grid Grilla de tasas candidatas para metodo `opt_local`.
#' @param eta_min Cota inferior de tasa.
#' @param eta_max Cota superior de tasa.
#' @param T Numero entero positivo de iteraciones.
#' @param activation Nombre de activacion.
#' @param alpha Penalizacion o sensibilidad geometrica.
#' @param beta Penalizacion por desplazamiento para costo local.
#' @param h Paso de diferencias finitas para Hessiano.
#' @param evaluar_geo_cada Frecuencia para calcular geometria completa.
#'
#' @return Objeto de clase `rnas_meta_geo_train`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta0 <- c(0.8, 0.3, 0.1)
#' res <- rnas_train_neuron_meta_geo(X, y, theta0, T = 10)
#' res$metricas
#'
#' @export
rnas_train_neuron_meta_geo <- function(X,
                                       y,
                                       theta0,
                                       eta0 = 0.1,
                                       metodo = "opt_local",
                                       eta_grid = seq(0.01, 0.1, length.out = 10),
                                       eta_min = 0.01,
                                       eta_max = 0.1,
                                       T = 100,
                                       activation = "tanh",
                                       alpha = 0,
                                       beta = 0,
                                       h = 1e-4,
                                       evaluar_geo_cada = 1L) {
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

  if (!is.numeric(T) || length(T) != 1L || T < 1 || T != as.integer(T)) {
    stop("`T` debe ser un entero positivo.")
  }

  if (!is.numeric(evaluar_geo_cada) ||
      length(evaluar_geo_cada) != 1L ||
      evaluar_geo_cada < 1 ||
      evaluar_geo_cada != as.integer(evaluar_geo_cada)) {
    stop("`evaluar_geo_cada` debe ser un entero positivo.")
  }

  metodo <- tolower(metodo)

  if (!metodo %in% c("opt_local", "curvatura", "lambda_max")) {
    stop("`metodo` debe ser 'opt_local', 'curvatura' o 'lambda_max'.")
  }

  theta <- as.numeric(theta0)
  eta_actual <- rnas_clip_eta(eta0, eta_min = eta_min, eta_max = eta_max)

  trayectoria <- vector("list", T + 1L)
  geo_cache <- NULL

  for (k in 0:T) {
    pars <- rnas_unpack_neuron_params(theta, d_input = ncol(X))

    grad_res <- rnas_grad_neuron(
      X = X,
      y = y,
      w = pars$w,
      b = pars$b,
      activation = activation
    )

    grad_vec <- c(grad_res$grad_w, grad_res$grad_b)
    grad_norm <- sqrt(sum(grad_vec^2))

    if (is.null(geo_cache) || k %% evaluar_geo_cada == 0) {
      geo_cache <- rnas_resumen_geometria_neuron(
        X = X,
        y = y,
        theta = theta,
        activation = activation,
        h = h
      )
    }

    kappa <- geo_cache$curvatura
    lambda_max <- geo_cache$lambda_max

    accion_eta <- NA_character_
    costo_local <- NA_real_

    if (metodo == "opt_local") {
      opt <- rnas_eta_opt_local(
        X = X,
        y = y,
        theta = theta,
        grad = grad_vec,
        eta_grid = eta_grid,
        activation = activation,
        alpha = alpha,
        beta = beta
      )

      eta_k <- rnas_clip_eta(
        eta = opt$eta_opt,
        eta_min = eta_min,
        eta_max = eta_max
      )

      costo_local <- opt$costo_min
      accion_eta <- "seleccion_optima_local"

    } else if (metodo == "curvatura") {
      pol <- rnas_eta_policy_geo(
        eta0 = eta0,
        kappa = kappa,
        modo = "curvatura",
        alpha = alpha,
        eta_min = eta_min,
        eta_max = eta_max
      )

      eta_k <- pol$eta
      accion_eta <- "ajuste_por_curvatura"

    } else if (metodo == "lambda_max") {
      pol <- rnas_eta_policy_geo(
        eta0 = eta0,
        lambda_max = lambda_max,
        modo = "lambda_max",
        alpha = alpha,
        eta_min = eta_min,
        eta_max = eta_max
      )

      eta_k <- pol$eta
      accion_eta <- "ajuste_por_lambda_max"
    }

    speed <- eta_k * grad_norm

    fila <- data.frame(
      iter = k,
      loss = grad_res$loss,
      grad_norm = grad_norm,
      eta = eta_k,
      speed = speed,
      kappa = kappa,
      lambda_min = geo_cache$lambda_min,
      lambda_max = lambda_max,
      costo_local = costo_local,
      accion_eta = accion_eta,
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

  reduccion_abs <- loss_inicial - loss_final
  reduccion_rel <- if (loss_inicial != 0) reduccion_abs / abs(loss_inicial) else NA_real_

  costo_acum <- sum(trayectoria_df$costo_local, na.rm = TRUE)

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
      delta_loss = loss_final - loss_inicial,
      reduccion_abs = reduccion_abs,
      reduccion_rel = reduccion_rel,
      eta_min_obs = min(trayectoria_df$eta),
      eta_max_obs = max(trayectoria_df$eta),
      eta_media = mean(trayectoria_df$eta),
      eta_cambios = sum(abs(diff(trayectoria_df$eta)) > .Machine$double.eps),
      velocidad_media = mean(trayectoria_df$speed),
      grad_norm_inicial = trayectoria_df$grad_norm[1],
      grad_norm_final = trayectoria_df$grad_norm[nrow(trayectoria_df)],
      kappa_inicial = trayectoria_df$kappa[1],
      kappa_final = trayectoria_df$kappa[nrow(trayectoria_df)],
      lambda_max_inicial = trayectoria_df$lambda_max[1],
      lambda_max_final = trayectoria_df$lambda_max[nrow(trayectoria_df)],
      costo_acumulado = costo_acum,
      descendente_global = isTRUE(loss_final < loss_inicial)
    ),
    configuracion = list(
      eta0 = eta0,
      metodo = metodo,
      eta_grid = eta_grid,
      eta_min = eta_min,
      eta_max = eta_max,
      T = T,
      activation = activation,
      alpha = alpha,
      beta = beta,
      h = h,
      evaluar_geo_cada = evaluar_geo_cada
    )
  )

  class(res) <- c("rnas_meta_geo_train", class(res))
  res
}


#' Resumir entrenamiento con meta-control geometrico RNAS
#'
#' Genera una tabla resumen de un objeto producido por
#' `rnas_train_neuron_meta_geo()`.
#'
#' @param object Objeto de clase `rnas_meta_geo_train`.
#' @param nombre_politica Nombre opcional de la politica.
#'
#' @return Data frame con metricas principales.
#'
#' @export
rnas_resumen_meta_geo <- function(object,
                                  nombre_politica = NULL) {
  if (!inherits(object, "rnas_meta_geo_train")) {
    stop("`object` debe ser un objeto de clase 'rnas_meta_geo_train'.")
  }

  if (is.null(nombre_politica)) {
    nombre_politica <- object$configuracion$metodo
  }

  data.frame(
    politica = nombre_politica,
    metodo = object$configuracion$metodo,
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
    kappa_inicial = object$metricas$kappa_inicial,
    kappa_final = object$metricas$kappa_final,
    lambda_max_inicial = object$metricas$lambda_max_inicial,
    lambda_max_final = object$metricas$lambda_max_final,
    costo_acumulado = object$metricas$costo_acumulado,
    descendente_global = object$metricas$descendente_global,
    stringsAsFactors = FALSE
  )
}


#' Comparar politicas de meta-control geometrico RNAS
#'
#' Ejecuta varias politicas de meta-control geometrico sobre el mismo caso
#' y devuelve una tabla comparativa.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta0 Vector inicial de estado.
#' @param politicas Lista nombrada de configuraciones. Cada elemento debe
#' contener al menos `metodo`.
#' @param eta0 Tasa base.
#' @param eta_min Cota inferior.
#' @param eta_max Cota superior.
#' @param T Numero de iteraciones.
#' @param activation Activacion.
#'
#' @return Objeto de clase `rnas_meta_policy_comparison`.
#'
#' @export
rnas_comparar_meta_politicas <- function(X,
                                         y,
                                         theta0,
                                         politicas,
                                         eta0 = 0.1,
                                         eta_min = 0.01,
                                         eta_max = 0.1,
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
    cfg <- politicas[[i]]

    if (!is.list(cfg) || is.null(cfg$metodo)) {
      stop("Cada politica debe ser una lista con el campo `metodo`.")
    }

    metodo <- cfg$metodo
    alpha <- if (is.null(cfg$alpha)) 0 else cfg$alpha
    beta <- if (is.null(cfg$beta)) 0 else cfg$beta
    h <- if (is.null(cfg$h)) 1e-4 else cfg$h
    eta_grid <- if (is.null(cfg$eta_grid)) seq(eta_min, eta_max, length.out = 10) else cfg$eta_grid
    evaluar_geo_cada <- if (is.null(cfg$evaluar_geo_cada)) 1L else cfg$evaluar_geo_cada

    mod <- rnas_train_neuron_meta_geo(
      X = X,
      y = y,
      theta0 = theta0,
      eta0 = eta0,
      metodo = metodo,
      eta_grid = eta_grid,
      eta_min = eta_min,
      eta_max = eta_max,
      T = T,
      activation = activation,
      alpha = alpha,
      beta = beta,
      h = h,
      evaluar_geo_cada = evaluar_geo_cada
    )

    modelos[[i]] <- mod
    resumenes[[i]] <- rnas_resumen_meta_geo(
      object = mod,
      nombre_politica = nombres[i]
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

  class(res) <- c("rnas_meta_policy_comparison", class(res))
  res
}


#' Imprimir entrenamiento con meta-control geometrico RNAS
#'
#' @param x Objeto de clase `rnas_meta_geo_train`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto `x`, de forma invisible.
#'
#' @export
print.rnas_meta_geo_train <- function(x, ...) {
  cat("Entrenamiento RNAS con meta-control geometrico\n")
  cat("------------------------------------------------\n")
  cat("Metodo           :", x$configuracion$metodo, "\n")
  cat("Eta media        :", x$metricas$eta_media, "\n")
  cat("Cambios de eta   :", x$metricas$eta_cambios, "\n")
  cat("Loss inicial     :", x$metricas$loss_inicial, "\n")
  cat("Loss final       :", x$metricas$loss_final, "\n")
  cat("Reduccion rel.   :", x$metricas$reduccion_rel, "\n")
  cat("Costo acumulado  :", x$metricas$costo_acumulado, "\n")
  cat("Descendente      :", x$metricas$descendente_global, "\n")

  invisible(x)
}


#' Imprimir comparacion de politicas de meta-control RNAS
#'
#' @param x Objeto de clase `rnas_meta_policy_comparison`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto `x`, de forma invisible.
#'
#' @export
print.rnas_meta_policy_comparison <- function(x, ...) {
  cat("Comparacion de politicas de meta-control geometrico RNAS\n")
  cat("---------------------------------------------------------\n")
  cat("Politicas evaluadas :", nrow(x$comparacion), "\n")
  cat("Eta inicial         :", x$configuracion$eta0, "\n")
  cat("Iteraciones         :", x$configuracion$T, "\n")

  invisible(x)
}
