#' Activacion sigmoide RNAS
#'
#' Evalua la funcion sigmoide sobre un valor escalar, vector o matriz numerica.
#' La funcion sigmoide se define como:
#' sigma(z) = 1 / (1 + exp(-z)).
#'
#' @param z Valor numerico, vector numerico o matriz numerica.
#'
#' @return Objeto numerico del mismo tipo dimensional que `z`, con valores entre 0 y 1.
#'
#' @examples
#' rnas_sigmoid(0)
#' rnas_sigmoid(c(-2, 0, 2))
#'
#' @export
rnas_sigmoid <- function(z) {
  if (!is.numeric(z)) {
    stop("`z` debe ser numerico.")
  }

  1 / (1 + exp(-z))
}


#' Derivada de la activacion sigmoide RNAS
#'
#' Evalua la derivada de la funcion sigmoide. La derivada se calcula como:
#' sigma'(z) = sigma(z) * (1 - sigma(z)).
#'
#' @param z Valor numerico, vector numerico o matriz numerica.
#'
#' @return Objeto numerico del mismo tipo dimensional que `z`, con los valores de la derivada.
#'
#' @examples
#' rnas_dsigmoid(0)
#' rnas_dsigmoid(c(-2, 0, 2))
#'
#' @export
rnas_dsigmoid <- function(z) {
  if (!is.numeric(z)) {
    stop("`z` debe ser numerico.")
  }

  s <- rnas_sigmoid(z)
  s * (1 - s)
}


#' Activacion tangente hiperbolica RNAS
#'
#' Evalua la funcion tangente hiperbolica sobre un valor escalar,
#' vector o matriz numerica.
#'
#' @param z Valor numerico, vector numerico o matriz numerica.
#'
#' @return Objeto numerico del mismo tipo dimensional que `z`, con valores entre -1 y 1.
#'
#' @examples
#' rnas_tanh(0)
#' rnas_tanh(c(-2, 0, 2))
#'
#' @export
rnas_tanh <- function(z) {
  if (!is.numeric(z)) {
    stop("`z` debe ser numerico.")
  }

  tanh(z)
}


#' Derivada de la activacion tangente hiperbolica RNAS
#'
#' Evalua la derivada de la tangente hiperbolica. La derivada se calcula como:
#' tanh'(z) = 1 - tanh(z)^2.
#'
#' @param z Valor numerico, vector numerico o matriz numerica.
#'
#' @return Objeto numerico del mismo tipo dimensional que `z`, con los valores de la derivada.
#'
#' @examples
#' rnas_dtanh(0)
#' rnas_dtanh(c(-2, 0, 2))
#'
#' @export
rnas_dtanh <- function(z) {
  if (!is.numeric(z)) {
    stop("`z` debe ser numerico.")
  }

  1 - tanh(z)^2
}


#' Obtener activacion RNAS por nombre
#'
#' Devuelve una lista con la funcion de activacion y su derivada asociada.
#' Esta funcion estandariza el uso de activaciones dentro del paquete.
#'
#' @param nombre Cadena de texto con el nombre de la activacion.
#' Puede ser `"sigmoid"` o `"tanh"`.
#'
#' @return Lista con dos elementos:
#' \describe{
#'   \item{activation}{Funcion de activacion.}
#'   \item{derivative}{Derivada de la activacion.}
#' }
#'
#' @examples
#' act <- rnas_get_activation("tanh")
#' act$activation(0)
#' act$derivative(0)
#'
#' @export
rnas_get_activation <- function(nombre = c("sigmoid", "tanh")) {
  nombre <- match.arg(nombre)

  if (nombre == "sigmoid") {
    return(list(
      activation = rnas_sigmoid,
      derivative = rnas_dsigmoid,
      name = "sigmoid"
    ))
  }

  if (nombre == "tanh") {
    return(list(
      activation = rnas_tanh,
      derivative = rnas_dtanh,
      name = "tanh"
    ))
  }
}
