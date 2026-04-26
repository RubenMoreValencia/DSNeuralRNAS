#' Actualizar parámetros de una neurona RNAS
#'
#' Actualiza pesos y sesgo usando la regla de descenso por gradiente:
#' w_new = w - eta * grad_w
#' b_new = b - eta * grad_b
#'
#' @param w Vector numérico de pesos actuales.
#' @param b Sesgo numérico escalar actual.
#' @param grad_w Vector numérico de gradientes respecto a los pesos.
#' @param grad_b Gradiente numérico respecto al sesgo.
#' @param eta Tasa de aprendizaje positiva.
#'
#' @return Lista con `w_new` y `b_new`.
#'
#' @examples
#' w <- c(0.8, 0.3)
#' b <- 0.1
#' grad_w <- c(-0.01, 0.02)
#' grad_b <- -0.03
#' rnas_update_params_neuron(w, b, grad_w, grad_b, eta = 0.1)
#'
#' @export
rnas_update_params_neuron <- function(w, b, grad_w, grad_b, eta) {
  if (!is.numeric(w) || !is.vector(w)) {
    stop("`w` debe ser un vector numerico.")
  }

  if (!is.numeric(b) || length(b) != 1L) {
    stop("`b` debe ser un escalar numerico.")
  }

  if (!is.numeric(grad_w) || !is.vector(grad_w)) {
    stop("`grad_w` debe ser un vector numerico.")
  }

  if (!is.numeric(grad_b) || length(grad_b) != 1L) {
    stop("`grad_b` debe ser un escalar numerico.")
  }

  if (length(w) != length(grad_w)) {
    stop("`w` y `grad_w` deben tener la misma longitud.")
  }

  if (!is.numeric(eta) || length(eta) != 1L || eta <= 0) {
    stop("`eta` debe ser un escalar numerico positivo.")
  }

  list(
    w_new = w - eta * grad_w,
    b_new = as.numeric(b - eta * grad_b)
  )
}


#' Entrenar una neurona RNAS por descenso discreto del gradiente
#'
#' Ejecuta el ciclo de entrenamiento discreto de una neurona individual.
#' En cada iteración calcula pérdida, gradiente, norma del gradiente y
#' actualiza pesos y sesgo usando una tasa de aprendizaje constante.
#'
#' @param X Matriz numérica de entradas.
#' @param y Vector numérico de valores observados.
#' @param w0 Vector numérico de pesos iniciales.
#' @param b0 Sesgo inicial numérico escalar.
#' @param eta Tasa de aprendizaje positiva.
#' @param T Número máximo de iteraciones.
#' @param activation Nombre de activación. Puede ser `"sigmoid"` o `"tanh"`.
#' @param tol_loss Tolerancia opcional para cambio absoluto de pérdida.
#' Use `NULL` para desactivar este criterio.
#' @param tol_grad Tolerancia opcional para norma del gradiente.
#' Use `NULL` para desactivar este criterio.
#' @param registrar_cada Entero positivo. Frecuencia de registro de la trayectoria.
#'
#' @return Objeto de clase `rnas_neuron_train` con parámetros finales,
#' trayectoria, métricas y configuración.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' res <- rnas_train_neuron(X, y, w0 = c(0.8, 0.3), b0 = 0.1,
#'                          eta = 0.1, T = 50)
#' res$metricas
#'
#' @export
rnas_train_neuron <- function(X,
                              y,
                              w0,
                              b0,
                              eta = 0.1,
                              T = 100,
                              activation = "tanh",
                              tol_loss = NULL,
                              tol_grad = NULL,
                              registrar_cada = 1L) {
  rnas_validar_datos_supervisados(X, y, w0, b0)

  if (!is.numeric(eta) || length(eta) != 1L || eta <= 0) {
    stop("`eta` debe ser un escalar numerico positivo.")
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

  w <- as.numeric(w0)
  b <- as.numeric(b0)

  trayectoria <- vector("list", T + 1L)

  parada <- "max_iter"
  iter_final <- T
  loss_anterior <- NA_real_

  for (k in 0:T) {
    grad <- rnas_grad_neuron(
      X = X,
      y = y,
      w = w,
      b = b,
      activation = activation
    )

    grad_norm <- sqrt(sum(grad$grad_w^2) + grad$grad_b^2)

    if (k %% registrar_cada == 0 || k == T) {
      trayectoria[[k + 1L]] <- data.frame(
        iter = k,
        loss = grad$loss,
        grad_norm = grad_norm,
        b = b,
        stringsAsFactors = FALSE
      )

      for (j in seq_along(w)) {
        trayectoria[[k + 1L]][[paste0("w", j)]] <- w[j]
      }
    }

    if (!is.null(tol_grad) && grad_norm < tol_grad) {
      parada <- "tol_grad"
      iter_final <- k
      break
    }

    if (!is.null(tol_loss) && !is.na(loss_anterior)) {
      if (abs(loss_anterior - grad$loss) < tol_loss) {
        parada <- "tol_loss"
        iter_final <- k
        break
      }
    }

    loss_anterior <- grad$loss

    if (k < T) {
      upd <- rnas_update_params_neuron(
        w = w,
        b = b,
        grad_w = grad$grad_w,
        grad_b = grad$grad_b,
        eta = eta
      )

      w <- upd$w_new
      b <- upd$b_new
    }
  }

  trayectoria_df <- do.call(
    rbind,
    trayectoria[!vapply(trayectoria, is.null, logical(1))]
  )

  rownames(trayectoria_df) <- NULL

  loss_inicial <- trayectoria_df$loss[1]
  loss_final <- trayectoria_df$loss[nrow(trayectoria_df)]
  delta_loss <- loss_final - loss_inicial
  reduccion_abs <- loss_inicial - loss_final
  reduccion_rel <- if (loss_inicial != 0) reduccion_abs / abs(loss_inicial) else NA_real_

  y_hat_final <- rnas_neuron_forward_batch(
    X = X,
    w = w,
    b = b,
    activation = activation
  )

  res <- list(
    w_final = w,
    b_final = b,
    y_hat_final = y_hat_final,
    trayectoria = trayectoria_df,
    metricas = list(
      loss_inicial = loss_inicial,
      loss_final = loss_final,
      delta_loss = delta_loss,
      reduccion_abs = reduccion_abs,
      reduccion_rel = reduccion_rel,
      iter_final = iter_final,
      parada = parada,
      descendente_global = isTRUE(loss_final < loss_inicial)
    ),
    configuracion = list(
      eta = eta,
      T = T,
      activation = activation,
      tol_loss = tol_loss,
      tol_grad = tol_grad,
      registrar_cada = registrar_cada
    )
  )

  class(res) <- c("rnas_neuron_train", class(res))
  res
}


#' Predecir con una neurona RNAS entrenada
#'
#' Genera predicciones usando un objeto entrenado por `rnas_train_neuron()`
#' o usando pesos y sesgo proporcionados directamente.
#'
#' @param object Objeto `rnas_neuron_train` o `NULL`.
#' @param X Matriz numérica de entradas.
#' @param w Vector de pesos. Se usa si `object = NULL`.
#' @param b Sesgo numérico. Se usa si `object = NULL`.
#' @param activation Activación. Se usa si `object = NULL`.
#'
#' @return Vector numérico de predicciones.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' rnas_predict_neuron(NULL, X, w = c(0.8, 0.3), b = 0.1)
#'
#' @export
rnas_predict_neuron <- function(object = NULL,
                                X,
                                w = NULL,
                                b = NULL,
                                activation = "tanh") {
  if (!is.null(object)) {
    if (!inherits(object, "rnas_neuron_train")) {
      stop("`object` debe ser NULL o un objeto de clase 'rnas_neuron_train'.")
    }

    w <- object$w_final
    b <- object$b_final
    activation <- object$configuracion$activation
  }

  if (is.null(w) || is.null(b)) {
    stop("Debe proporcionar `object` o los argumentos `w` y `b`.")
  }

  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (ncol(X) != length(w)) {
    stop("El numero de columnas de `X` debe coincidir con la longitud de `w`.")
  }

  rnas_neuron_forward_batch(
    X = X,
    w = w,
    b = b,
    activation = activation
  )
}


#' Resumir entrenamiento de una neurona RNAS
#'
#' Genera una tabla resumen con métricas principales de un entrenamiento
#' realizado con `rnas_train_neuron()`.
#'
#' @param object Objeto de clase `rnas_neuron_train`.
#'
#' @return Data frame con pérdida inicial, pérdida final, reducción,
#' iteración final, criterio de parada y configuración principal.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' res <- rnas_train_neuron(X, y, w0 = c(0.8, 0.3), b0 = 0.1,
#'                          eta = 0.1, T = 50)
#' rnas_resumen_entrenamiento_neuron(res)
#'
#' @export
rnas_resumen_entrenamiento_neuron <- function(object) {
  if (!inherits(object, "rnas_neuron_train")) {
    stop("`object` debe ser un objeto de clase 'rnas_neuron_train'.")
  }

  data.frame(
    activation = object$configuracion$activation,
    eta = object$configuracion$eta,
    T_configurado = object$configuracion$T,
    iter_final = object$metricas$iter_final,
    parada = object$metricas$parada,
    loss_inicial = object$metricas$loss_inicial,
    loss_final = object$metricas$loss_final,
    delta_loss = object$metricas$delta_loss,
    reduccion_abs = object$metricas$reduccion_abs,
    reduccion_rel = object$metricas$reduccion_rel,
    descendente_global = object$metricas$descendente_global,
    stringsAsFactors = FALSE
  )
}


#' Imprimir resumen de entrenamiento de neurona RNAS
#'
#' Método de impresión para objetos creados con `rnas_train_neuron()`.
#'
#' @param x Objeto de clase `rnas_neuron_train`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto `x`, de forma invisible.
#'
#' @export
print.rnas_neuron_train <- function(x, ...) {
  cat("Entrenamiento de neurona RNAS\n")
  cat("--------------------------------\n")
  cat("Activacion       :", x$configuracion$activation, "\n")
  cat("Eta              :", x$configuracion$eta, "\n")
  cat("Iteracion final  :", x$metricas$iter_final, "\n")
  cat("Parada           :", x$metricas$parada, "\n")
  cat("Loss inicial     :", x$metricas$loss_inicial, "\n")
  cat("Loss final       :", x$metricas$loss_final, "\n")
  cat("Delta loss       :", x$metricas$delta_loss, "\n")
  cat("Descendente      :", x$metricas$descendente_global, "\n")

  invisible(x)
}
