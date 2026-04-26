#' Validar dimensiones para una neurona RNAS
#'
#' Verifica que la entrada `x` y los pesos `w` sean numericos y tengan
#' dimensiones compatibles para evaluar una neurona individual.
#'
#' @param x Vector numerico de entrada.
#' @param w Vector numerico de pesos.
#' @param b Sesgo numerico escalar.
#'
#' @return `TRUE` de forma invisible si las dimensiones son validas.
#'
#' @examples
#' rnas_validar_dimensiones_neurona(c(1, 2), c(0.5, -0.2), 0.1)
#'
#' @export
rnas_validar_dimensiones_neurona <- function(x, w, b) {
  if (!is.numeric(x) || !is.vector(x)) {
    stop("`x` debe ser un vector numerico.")
  }

  if (!is.numeric(w) || !is.vector(w)) {
    stop("`w` debe ser un vector numerico.")
  }

  if (!is.numeric(b) || length(b) != 1L) {
    stop("`b` debe ser un escalar numerico.")
  }

  if (length(x) != length(w)) {
    stop("`x` y `w` deben tener la misma longitud.")
  }

  invisible(TRUE)
}


#' Evaluar una neurona individual RNAS
#'
#' Calcula el potencial interno y la salida estimada de una neurona individual.
#' La neurona se define como:
#' y_hat = phi(w^T x + b).
#'
#' @param x Vector numerico de entrada.
#' @param w Vector numerico de pesos.
#' @param b Sesgo numerico escalar.
#' @param activation Funcion de activacion o nombre de activacion.
#' Puede ser una funcion R o una cadena: `"sigmoid"` o `"tanh"`.
#' @param devolver_z Valor logico. Si es `TRUE`, devuelve tambien el potencial interno `z`.
#'
#' @return Si `devolver_z = FALSE`, retorna la salida estimada `y_hat`.
#' Si `devolver_z = TRUE`, retorna una lista con `z`, `y_hat` y `activation`.
#'
#' @examples
#' x <- c(0.5, -0.2)
#' w <- c(0.8, 0.3)
#' b <- 0.1
#'
#' rnas_neuron_forward(x, w, b, activation = "tanh")
#' rnas_neuron_forward(x, w, b, activation = "tanh", devolver_z = TRUE)
#'
#' @export
rnas_neuron_forward <- function(x,
                                w,
                                b,
                                activation = "tanh",
                                devolver_z = FALSE) {
  rnas_validar_dimensiones_neurona(x, w, b)

  if (is.character(activation)) {
    act <- rnas_get_activation(activation)
    activation_fun <- act$activation
    activation_name <- act$name
  } else if (is.function(activation)) {
    activation_fun <- activation
    activation_name <- "custom"
  } else {
    stop("`activation` debe ser una funcion o un nombre valido: 'sigmoid' o 'tanh'.")
  }

  z <- sum(w * x) + b
  y_hat <- activation_fun(z)

  if (isTRUE(devolver_z)) {
    return(list(
      z = z,
      y_hat = y_hat,
      activation = activation_name
    ))
  }

  y_hat
}


#' Evaluar una neurona RNAS para un conjunto de observaciones
#'
#' Evalua una neurona individual sobre una matriz de entradas `X`.
#' Cada fila de `X` representa una observacion y cada columna una variable.
#'
#' @param X Matriz numerica de entradas.
#' @param w Vector numerico de pesos.
#' @param b Sesgo numerico escalar.
#' @param activation Funcion de activacion o nombre de activacion.
#' Puede ser una funcion R o una cadena: `"sigmoid"` o `"tanh"`.
#' @param devolver_z Valor logico. Si es `TRUE`, devuelve tambien los potenciales internos.
#'
#' @return Si `devolver_z = FALSE`, retorna un vector de salidas estimadas.
#' Si `devolver_z = TRUE`, retorna una lista con `z`, `y_hat` y `activation`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8), ncol = 2, byrow = TRUE)
#' w <- c(0.8, 0.3)
#' b <- 0.1
#'
#' rnas_neuron_forward_batch(X, w, b, activation = "tanh")
#'
#' @export
rnas_neuron_forward_batch <- function(X,
                                      w,
                                      b,
                                      activation = "tanh",
                                      devolver_z = FALSE) {
  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (!is.numeric(w) || !is.vector(w)) {
    stop("`w` debe ser un vector numerico.")
  }

  if (!is.numeric(b) || length(b) != 1L) {
    stop("`b` debe ser un escalar numerico.")
  }

  if (ncol(X) != length(w)) {
    stop("El numero de columnas de `X` debe coincidir con la longitud de `w`.")
  }

  if (is.character(activation)) {
    act <- rnas_get_activation(activation)
    activation_fun <- act$activation
    activation_name <- act$name
  } else if (is.function(activation)) {
    activation_fun <- activation
    activation_name <- "custom"
  } else {
    stop("`activation` debe ser una funcion o un nombre valido: 'sigmoid' o 'tanh'.")
  }

  z <- as.numeric(X %*% w + b)
  y_hat <- activation_fun(z)

  if (isTRUE(devolver_z)) {
    return(list(
      z = z,
      y_hat = y_hat,
      activation = activation_name
    ))
  }

  y_hat
}
