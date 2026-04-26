#' Validar datos supervisados para una neurona RNAS
#'
#' Verifica que las entradas, salidas observadas, pesos y sesgo tengan
#' estructura compatible para calcular pérdida y gradiente en una neurona.
#'
#' @param X Matriz numérica de entradas.
#' @param y Vector numérico de valores observados.
#' @param w Vector numérico de pesos.
#' @param b Sesgo numérico escalar.
#'
#' @return `TRUE` de forma invisible si las dimensiones son válidas.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' w <- c(0.8, 0.3)
#' b <- 0.1
#' rnas_validar_datos_supervisados(X, y, w, b)
#'
#' @export
rnas_validar_datos_supervisados <- function(X, y, w, b) {
  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (!is.numeric(w) || !is.vector(w)) {
    stop("`w` debe ser un vector numerico.")
  }

  if (!is.numeric(b) || length(b) != 1L) {
    stop("`b` debe ser un escalar numerico.")
  }

  if (nrow(X) != length(y)) {
    stop("El numero de filas de `X` debe coincidir con la longitud de `y`.")
  }

  if (ncol(X) != length(w)) {
    stop("El numero de columnas de `X` debe coincidir con la longitud de `w`.")
  }

  invisible(TRUE)
}


#' Calcular pérdida MSE para una neurona RNAS
#'
#' Calcula la pérdida cuadrática media entre los valores observados `y`
#' y las predicciones generadas por una neurona individual.
#'
#' La pérdida se define como:
#' L = mean((y_hat - y)^2).
#'
#' @param X Matriz numérica de entradas.
#' @param y Vector numérico de valores observados.
#' @param w Vector numérico de pesos.
#' @param b Sesgo numérico escalar.
#' @param activation Función de activación o nombre de activación.
#' Puede ser `"sigmoid"` o `"tanh"`.
#' @param devolver_detalle Valor lógico. Si es `TRUE`, retorna predicciones,
#' errores y pérdida.
#'
#' @return Si `devolver_detalle = FALSE`, retorna la pérdida MSE como escalar.
#' Si `devolver_detalle = TRUE`, retorna una lista con `loss`, `y_hat`,
#' `error` y `activation`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' w <- c(0.8, 0.3)
#' b <- 0.1
#' rnas_loss_mse_neuron(X, y, w, b, activation = "tanh")
#'
#' @export
rnas_loss_mse_neuron <- function(X,
                                 y,
                                 w,
                                 b,
                                 activation = "tanh",
                                 devolver_detalle = FALSE) {
  rnas_validar_datos_supervisados(X, y, w, b)

  pred <- rnas_neuron_forward_batch(
    X = X,
    w = w,
    b = b,
    activation = activation,
    devolver_z = TRUE
  )

  error <- pred$y_hat - y
  loss <- mean(error^2)

  if (isTRUE(devolver_detalle)) {
    return(list(
      loss = loss,
      y_hat = pred$y_hat,
      error = error,
      z = pred$z,
      activation = pred$activation
    ))
  }

  loss
}


#' Calcular gradiente analítico para una neurona RNAS
#'
#' Calcula el gradiente analítico de la pérdida MSE respecto a los pesos
#' y al sesgo de una neurona individual.
#'
#' @param X Matriz numérica de entradas.
#' @param y Vector numérico de valores observados.
#' @param w Vector numérico de pesos.
#' @param b Sesgo numérico escalar.
#' @param activation Nombre de activación. Puede ser `"sigmoid"` o `"tanh"`.
#'
#' @return Lista con:
#' \describe{
#'   \item{grad_w}{Vector de gradientes respecto a los pesos.}
#'   \item{grad_b}{Gradiente respecto al sesgo.}
#'   \item{loss}{Valor de pérdida MSE.}
#'   \item{y_hat}{Predicciones de la neurona.}
#'   \item{error}{Errores de predicción.}
#'   \item{z}{Potenciales internos.}
#' }
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' w <- c(0.8, 0.3)
#' b <- 0.1
#' rnas_grad_neuron(X, y, w, b, activation = "tanh")
#'
#' @export
rnas_grad_neuron <- function(X,
                             y,
                             w,
                             b,
                             activation = "tanh") {
  rnas_validar_datos_supervisados(X, y, w, b)

  if (!is.character(activation)) {
    stop("`activation` debe indicarse como nombre: 'sigmoid' o 'tanh'.")
  }

  act <- rnas_get_activation(activation)

  z <- as.numeric(X %*% w + b)
  y_hat <- act$activation(z)
  error <- y_hat - y
  dphi <- act$derivative(z)

  delta <- error * dphi

  n <- nrow(X)

  grad_w <- as.numeric((2 / n) * crossprod(X, delta))
  grad_b <- as.numeric((2 / n) * sum(delta))
  loss <- mean(error^2)

  list(
    grad_w = grad_w,
    grad_b = grad_b,
    loss = loss,
    y_hat = y_hat,
    error = error,
    z = z,
    activation = act$name
  )
}


#' Calcular gradiente numérico para una neurona RNAS
#'
#' Aproxima el gradiente de la pérdida MSE mediante diferencias finitas
#' centradas. Esta función se usa para verificar el gradiente analítico.
#'
#' @param X Matriz numérica de entradas.
#' @param y Vector numérico de valores observados.
#' @param w Vector numérico de pesos.
#' @param b Sesgo numérico escalar.
#' @param activation Nombre de activación. Puede ser `"sigmoid"` o `"tanh"`.
#' @param h Paso positivo para diferencias finitas.
#'
#' @return Lista con `grad_w`, `grad_b` y `h`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' w <- c(0.8, 0.3)
#' b <- 0.1
#' rnas_grad_num_neuron(X, y, w, b, activation = "tanh")
#'
#' @export
rnas_grad_num_neuron <- function(X,
                                 y,
                                 w,
                                 b,
                                 activation = "tanh",
                                 h = 1e-6) {
  rnas_validar_datos_supervisados(X, y, w, b)

  if (!is.numeric(h) || length(h) != 1L || h <= 0) {
    stop("`h` debe ser un escalar numerico positivo.")
  }

  grad_w <- numeric(length(w))

  for (j in seq_along(w)) {
    w_plus <- w
    w_minus <- w

    w_plus[j] <- w_plus[j] + h
    w_minus[j] <- w_minus[j] - h

    loss_plus <- rnas_loss_mse_neuron(
      X = X,
      y = y,
      w = w_plus,
      b = b,
      activation = activation
    )

    loss_minus <- rnas_loss_mse_neuron(
      X = X,
      y = y,
      w = w_minus,
      b = b,
      activation = activation
    )

    grad_w[j] <- (loss_plus - loss_minus) / (2 * h)
  }

  loss_b_plus <- rnas_loss_mse_neuron(
    X = X,
    y = y,
    w = w,
    b = b + h,
    activation = activation
  )

  loss_b_minus <- rnas_loss_mse_neuron(
    X = X,
    y = y,
    w = w,
    b = b - h,
    activation = activation
  )

  grad_b <- (loss_b_plus - loss_b_minus) / (2 * h)

  list(
    grad_w = grad_w,
    grad_b = grad_b,
    h = h
  )
}


#' Verificar gradiente de una neurona RNAS
#'
#' Compara el gradiente analítico con el gradiente numérico calculado
#' por diferencias finitas centradas.
#'
#' @param X Matriz numérica de entradas.
#' @param y Vector numérico de valores observados.
#' @param w Vector numérico de pesos.
#' @param b Sesgo numérico escalar.
#' @param activation Nombre de activación. Puede ser `"sigmoid"` o `"tanh"`.
#' @param h Paso positivo para diferencias finitas.
#' @param tol Tolerancia para considerar correcta la verificación.
#'
#' @return Lista con gradiente analítico, gradiente numérico, norma de diferencia,
#' error relativo y resultado lógico de verificación.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' w <- c(0.8, 0.3)
#' b <- 0.1
#' rnas_grad_check_neuron(X, y, w, b, activation = "tanh")
#'
#' @export
rnas_grad_check_neuron <- function(X,
                                   y,
                                   w,
                                   b,
                                   activation = "tanh",
                                   h = 1e-6,
                                   tol = 1e-5) {
  grad_a <- rnas_grad_neuron(
    X = X,
    y = y,
    w = w,
    b = b,
    activation = activation
  )

  grad_n <- rnas_grad_num_neuron(
    X = X,
    y = y,
    w = w,
    b = b,
    activation = activation,
    h = h
  )

  vec_a <- c(grad_a$grad_w, grad_a$grad_b)
  vec_n <- c(grad_n$grad_w, grad_n$grad_b)

  diff_abs <- sqrt(sum((vec_a - vec_n)^2))
  error_rel <- diff_abs / (sqrt(sum(vec_a^2)) + sqrt(sum(vec_n^2)) + .Machine$double.eps)

  list(
    grad_analitico = list(
      grad_w = grad_a$grad_w,
      grad_b = grad_a$grad_b
    ),
    grad_numerico = list(
      grad_w = grad_n$grad_w,
      grad_b = grad_n$grad_b
    ),
    diff_abs = diff_abs,
    error_rel = error_rel,
    verificado = isTRUE(error_rel < tol),
    tol = tol,
    loss = grad_a$loss,
    activation = activation
  )
}
