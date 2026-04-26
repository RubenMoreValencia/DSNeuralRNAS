#' Empaquetar parametros de una neurona RNAS
#'
#' Agrupa pesos y sesgo en un unico vector de estado theta.
#' Si w tiene longitud d, entonces theta tiene longitud d + 1.
#'
#' @param w Vector numerico de pesos.
#' @param b Sesgo numerico escalar.
#'
#' @return Vector numerico theta = c(w, b).
#'
#' @examples
#' theta <- rnas_pack_neuron_params(w = c(0.8, 0.3), b = 0.1)
#' theta
#'
#' @export
rnas_pack_neuron_params <- function(w, b) {
  if (!is.numeric(w) || !is.vector(w)) {
    stop("`w` debe ser un vector numerico.")
  }

  if (!is.numeric(b) || length(b) != 1L) {
    stop("`b` debe ser un escalar numerico.")
  }

  c(as.numeric(w), as.numeric(b))
}


#' Desempaquetar parametros de una neurona RNAS
#'
#' Recupera pesos y sesgo desde un vector de estado theta.
#'
#' @param theta Vector numerico de estado.
#' @param d_input Entero positivo con la dimension de entrada.
#'
#' @return Lista con `w` y `b`.
#'
#' @examples
#' theta <- c(0.8, 0.3, 0.1)
#' rnas_unpack_neuron_params(theta, d_input = 2)
#'
#' @export
rnas_unpack_neuron_params <- function(theta, d_input) {
  if (!is.numeric(theta) || !is.vector(theta)) {
    stop("`theta` debe ser un vector numerico.")
  }

  if (!is.numeric(d_input) || length(d_input) != 1L ||
      d_input < 1 || d_input != as.integer(d_input)) {
    stop("`d_input` debe ser un entero positivo.")
  }

  if (length(theta) != d_input + 1L) {
    stop("La longitud de `theta` debe ser igual a d_input + 1.")
  }

  list(
    w = as.numeric(theta[seq_len(d_input)]),
    b = as.numeric(theta[d_input + 1L])
  )
}


#' Evaluar tasa de aprendizaje dinamica RNAS
#'
#' Evalua una tasa de aprendizaje que puede ser escalar numerico o funcion.
#' Si es funcion, debe aceptar al menos el argumento `t`.
#'
#' @param eta Escalar numerico positivo o funcion.
#' @param t Tiempo actual.
#' @param theta Estado actual.
#' @param loss Perdida actual.
#' @param grad_norm Norma del gradiente actual.
#'
#' @return Escalar numerico positivo.
#'
#' @examples
#' rnas_eval_eta(0.1, t = 0)
#' rnas_eval_eta(function(t, ...) 0.1 / (1 + t), t = 2)
#'
#' @export
rnas_eval_eta <- function(eta,
                          t,
                          theta = NULL,
                          loss = NULL,
                          grad_norm = NULL) {
  if (is.function(eta)) {
    eta_val <- eta(
      t = t,
      theta = theta,
      loss = loss,
      grad_norm = grad_norm
    )
  } else {
    eta_val <- eta
  }

  if (!is.numeric(eta_val) || length(eta_val) != 1L || eta_val <= 0) {
    stop("La tasa de aprendizaje evaluada debe ser un escalar numerico positivo.")
  }

  as.numeric(eta_val)
}


#' Campo de gradiente para una neurona RNAS
#'
#' Calcula la derivada temporal del estado theta bajo la dinamica:
#' dtheta/dt = - eta * grad L(theta).
#'
#' @param theta Vector de estado que contiene pesos y sesgo.
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param eta Escalar positivo o funcion de tasa de aprendizaje.
#' @param t Tiempo actual.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#'
#' @return Lista con `theta_dot`, `loss`, `grad`, `grad_norm`, `speed` y `eta`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' theta <- c(0.8, 0.3, 0.1)
#' rnas_campo_gradiente_neuron(theta, X, y, eta = 0.1)
#'
#' @export
rnas_campo_gradiente_neuron <- function(theta,
                                        X,
                                        y,
                                        eta = 0.1,
                                        t = 0,
                                        activation = "tanh") {
  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (nrow(X) != length(y)) {
    stop("La longitud de `y` debe coincidir con el numero de filas de `X`.")
  }

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

  eta_val <- rnas_eval_eta(
    eta = eta,
    t = t,
    theta = theta,
    loss = grad$loss,
    grad_norm = grad_norm
  )

  theta_dot <- -eta_val * grad_vec
  speed <- sqrt(sum(theta_dot^2))

  list(
    theta_dot = theta_dot,
    loss = grad$loss,
    grad = grad_vec,
    grad_norm = grad_norm,
    speed = speed,
    eta = eta_val,
    w = pars$w,
    b = pars$b,
    activation = activation
  )
}


#' Integrar dinamica continua de una neurona RNAS
#'
#' Integra mediante Euler explicito la dinamica continua aproximada:
#' Actualiza el estado desde theta[k] hacia theta[k + 1] usando dt y theta_dot.
#' donde theta_dot_k = - eta * grad L(theta_k).
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta0 Vector inicial de estado. Debe contener pesos y sesgo.
#' @param eta Escalar positivo o funcion de tasa de aprendizaje.
#' @param dt Paso temporal positivo.
#' @param T Numero entero positivo de pasos de integracion.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#' @param tol_loss Tolerancia opcional para cambio absoluto de perdida.
#' @param tol_grad Tolerancia opcional para norma del gradiente.
#' @param registrar_cada Frecuencia de registro de la trayectoria.
#'
#' @return Objeto de clase `rnas_neuron_dynamics`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta0 <- rnas_pack_neuron_params(c(0.8, 0.3), 0.1)
#' res <- rnas_integrar_dinamica_neuron(X, y, theta0, eta = 0.1, dt = 1, T = 50)
#' res$metricas
#'
#' @export
rnas_integrar_dinamica_neuron <- function(X,
                                          y,
                                          theta0,
                                          eta = 0.1,
                                          dt = 1,
                                          T = 100,
                                          activation = "tanh",
                                          tol_loss = NULL,
                                          tol_grad = NULL,
                                          registrar_cada = 1L) {
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

  if (!is.numeric(dt) || length(dt) != 1L || dt <= 0) {
    stop("`dt` debe ser un escalar numerico positivo.")
  }

  if (!is.numeric(T) || length(T) != 1L || T < 1 || T != as.integer(T)) {
    stop("`T` debe ser un entero positivo.")
  }

  if (!is.null(tol_loss) &&
      (!is.numeric(tol_loss) || length(tol_loss) != 1L || tol_loss <= 0)) {
    stop("`tol_loss` debe ser NULL o un escalar numerico positivo.")
  }

  if (!is.null(tol_grad) &&
      (!is.numeric(tol_grad) || length(tol_grad) != 1L || tol_grad <= 0)) {
    stop("`tol_grad` debe ser NULL o un escalar numerico positivo.")
  }

  if (!is.numeric(registrar_cada) ||
      length(registrar_cada) != 1L ||
      registrar_cada < 1 ||
      registrar_cada != as.integer(registrar_cada)) {
    stop("`registrar_cada` debe ser un entero positivo.")
  }

  theta <- as.numeric(theta0)

  trayectoria <- vector("list", T + 1L)
  parada <- "max_iter"
  iter_final <- T
  loss_anterior <- NA_real_

  for (k in 0:T) {
    t_actual <- k * dt

    campo <- rnas_campo_gradiente_neuron(
      theta = theta,
      X = X,
      y = y,
      eta = eta,
      t = t_actual,
      activation = activation
    )

    pars <- rnas_unpack_neuron_params(theta, d_input = ncol(X))

    if (k %% registrar_cada == 0 || k == T) {
      fila <- data.frame(
        iter = k,
        time = t_actual,
        loss = campo$loss,
        grad_norm = campo$grad_norm,
        speed = campo$speed,
        eta = campo$eta,
        b = pars$b,
        stringsAsFactors = FALSE
      )

      for (j in seq_along(pars$w)) {
        fila[[paste0("w", j)]] <- pars$w[j]
      }

      for (j in seq_along(theta)) {
        fila[[paste0("theta", j)]] <- theta[j]
        fila[[paste0("theta_dot", j)]] <- campo$theta_dot[j]
      }

      trayectoria[[k + 1L]] <- fila
    }

    if (!is.null(tol_grad) && campo$grad_norm < tol_grad) {
      parada <- "tol_grad"
      iter_final <- k
      break
    }

    if (!is.null(tol_loss) && !is.na(loss_anterior)) {
      if (abs(loss_anterior - campo$loss) < tol_loss) {
        parada <- "tol_loss"
        iter_final <- k
        break
      }
    }

    loss_anterior <- campo$loss

    if (k < T) {
      theta <- theta + dt * campo$theta_dot
    }
  }

  trayectoria_df <- do.call(
    rbind,
    trayectoria[!vapply(trayectoria, is.null, logical(1))]
  )

  rownames(trayectoria_df) <- NULL

  pars_final <- rnas_unpack_neuron_params(theta, d_input = ncol(X))

  loss_inicial <- trayectoria_df$loss[1]
  loss_final <- trayectoria_df$loss[nrow(trayectoria_df)]
  delta_loss <- loss_final - loss_inicial
  reduccion_abs <- loss_inicial - loss_final
  reduccion_rel <- if (loss_inicial != 0) reduccion_abs / abs(loss_inicial) else NA_real_
  velocidad_media <- mean(trayectoria_df$speed, na.rm = TRUE)

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
      velocidad_media = velocidad_media,
      grad_norm_inicial = trayectoria_df$grad_norm[1],
      grad_norm_final = trayectoria_df$grad_norm[nrow(trayectoria_df)],
      speed_inicial = trayectoria_df$speed[1],
      speed_final = trayectoria_df$speed[nrow(trayectoria_df)],
      iter_final = iter_final,
      parada = parada,
      descendente_global = isTRUE(loss_final < loss_inicial)
    ),
    configuracion = list(
      eta = eta,
      dt = dt,
      T = T,
      activation = activation,
      tol_loss = tol_loss,
      tol_grad = tol_grad,
      registrar_cada = registrar_cada
    )
  )

  class(res) <- c("rnas_neuron_dynamics", class(res))
  res
}


#' Resumir dinamica continua de una neurona RNAS
#'
#' Genera una tabla resumen de la trayectoria dinamica producida por
#' `rnas_integrar_dinamica_neuron()`.
#'
#' @param object Objeto de clase `rnas_neuron_dynamics`.
#'
#' @return Data frame con metricas principales de la dinamica.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta0 <- rnas_pack_neuron_params(c(0.8, 0.3), 0.1)
#' res <- rnas_integrar_dinamica_neuron(X, y, theta0, eta = 0.1, dt = 1, T = 50)
#' rnas_resumen_dinamica_neuron(res)
#'
#' @export
rnas_resumen_dinamica_neuron <- function(object) {
  if (!inherits(object, "rnas_neuron_dynamics")) {
    stop("`object` debe ser un objeto de clase 'rnas_neuron_dynamics'.")
  }

  data.frame(
    activation = object$configuracion$activation,
    eta_tipo = if (is.function(object$configuracion$eta)) "funcion" else "constante",
    dt = object$configuracion$dt,
    T_configurado = object$configuracion$T,
    iter_final = object$metricas$iter_final,
    parada = object$metricas$parada,
    loss_inicial = object$metricas$loss_inicial,
    loss_final = object$metricas$loss_final,
    delta_loss = object$metricas$delta_loss,
    reduccion_abs = object$metricas$reduccion_abs,
    reduccion_rel = object$metricas$reduccion_rel,
    grad_norm_inicial = object$metricas$grad_norm_inicial,
    grad_norm_final = object$metricas$grad_norm_final,
    speed_inicial = object$metricas$speed_inicial,
    speed_final = object$metricas$speed_final,
    velocidad_media = object$metricas$velocidad_media,
    descendente_global = object$metricas$descendente_global,
    stringsAsFactors = FALSE
  )
}


#' Predecir desde la dinamica de una neurona RNAS
#'
#' Genera predicciones usando el estado final de una trayectoria dinamica.
#'
#' @param object Objeto de clase `rnas_neuron_dynamics`.
#' @param X Matriz numerica de entradas.
#'
#' @return Vector numerico de predicciones.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' theta0 <- rnas_pack_neuron_params(c(0.8, 0.3), 0.1)
#' res <- rnas_integrar_dinamica_neuron(X, y, theta0, eta = 0.1, dt = 1, T = 10)
#' rnas_predict_dinamica_neuron(res, X)
#'
#' @export
rnas_predict_dinamica_neuron <- function(object, X) {
  if (!inherits(object, "rnas_neuron_dynamics")) {
    stop("`object` debe ser un objeto de clase 'rnas_neuron_dynamics'.")
  }

  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (ncol(X) != length(object$w_final)) {
    stop("El numero de columnas de `X` debe coincidir con la longitud de `w_final`.")
  }

  rnas_neuron_forward_batch(
    X = X,
    w = object$w_final,
    b = object$b_final,
    activation = object$configuracion$activation
  )
}


#' Imprimir resumen de dinamica continua de neurona RNAS
#'
#' Metodo de impresion para objetos creados con
#' `rnas_integrar_dinamica_neuron()`.
#'
#' @param x Objeto de clase `rnas_neuron_dynamics`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto `x`, de forma invisible.
#'
#' @export
print.rnas_neuron_dynamics <- function(x, ...) {
  cat("Dinamica continua aproximada de neurona RNAS\n")
  cat("------------------------------------------------\n")
  cat("Activacion       :", x$configuracion$activation, "\n")
  cat("dt               :", x$configuracion$dt, "\n")
  cat("Iteracion final  :", x$metricas$iter_final, "\n")
  cat("Parada           :", x$metricas$parada, "\n")
  cat("Loss inicial     :", x$metricas$loss_inicial, "\n")
  cat("Loss final       :", x$metricas$loss_final, "\n")
  cat("Delta loss       :", x$metricas$delta_loss, "\n")
  cat("Velocidad media  :", x$metricas$velocidad_media, "\n")
  cat("Descendente      :", x$metricas$descendente_global, "\n")

  invisible(x)
}
