#' Inicializar un MLP simple RNAS
#'
#' Inicializa los parametros de un perceptron multicapa simple con una capa
#' oculta. La arquitectura considerada es:
#' y_hat = v^T phi(Wx + b1) + b2.
#'
#' @param d_input Entero positivo. Numero de variables de entrada.
#' @param d_hidden Entero positivo. Numero de neuronas ocultas.
#' @param init_sd Desviacion estandar para inicializar pesos aleatorios.
#' @param seed Semilla opcional para reproducibilidad.
#'
#' @return Lista con parametros `W`, `b1`, `v` y `b2`.
#'
#' @examples
#' params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)
#' str(params)
#'
#' @export
rnas_init_mlp <- function(d_input,
                          d_hidden,
                          init_sd = 0.1,
                          seed = NULL) {
  if (!is.numeric(d_input) || length(d_input) != 1L ||
      d_input < 1 || d_input != as.integer(d_input)) {
    stop("`d_input` debe ser un entero positivo.")
  }

  if (!is.numeric(d_hidden) || length(d_hidden) != 1L ||
      d_hidden < 1 || d_hidden != as.integer(d_hidden)) {
    stop("`d_hidden` debe ser un entero positivo.")
  }

  if (!is.numeric(init_sd) || length(init_sd) != 1L || init_sd <= 0) {
    stop("`init_sd` debe ser un escalar numerico positivo.")
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  W <- matrix(
    rnorm(d_hidden * d_input, mean = 0, sd = init_sd),
    nrow = d_hidden,
    ncol = d_input
  )

  b1 <- rep(0, d_hidden)
  v <- rnorm(d_hidden, mean = 0, sd = init_sd)
  b2 <- 0

  list(
    W = W,
    b1 = b1,
    v = v,
    b2 = b2
  )
}


#' Validar parametros de un MLP simple RNAS
#'
#' Verifica que la lista de parametros del MLP tenga estructura y dimensiones
#' compatibles con la matriz de entradas.
#'
#' @param params Lista con `W`, `b1`, `v` y `b2`.
#' @param X Matriz numerica de entradas.
#'
#' @return `TRUE` de forma invisible si la estructura es valida.
#'
#' @examples
#' X <- matrix(rnorm(8), ncol = 2)
#' params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)
#' rnas_validar_params_mlp(params, X)
#'
#' @export
rnas_validar_params_mlp <- function(params, X) {
  if (!is.list(params)) {
    stop("`params` debe ser una lista.")
  }

  req <- c("W", "b1", "v", "b2")

  if (!all(req %in% names(params))) {
    stop("`params` debe contener los elementos: W, b1, v y b2.")
  }

  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  W <- params$W
  b1 <- params$b1
  v <- params$v
  b2 <- params$b2

  if (!is.numeric(W) || !is.matrix(W)) {
    stop("`params$W` debe ser una matriz numerica.")
  }

  if (!is.numeric(b1) || !is.vector(b1)) {
    stop("`params$b1` debe ser un vector numerico.")
  }

  if (!is.numeric(v) || !is.vector(v)) {
    stop("`params$v` debe ser un vector numerico.")
  }

  if (!is.numeric(b2) || length(b2) != 1L) {
    stop("`params$b2` debe ser un escalar numerico.")
  }

  d_hidden <- nrow(W)
  d_input <- ncol(W)

  if (ncol(X) != d_input) {
    stop("El numero de columnas de `X` debe coincidir con ncol(params$W).")
  }

  if (length(b1) != d_hidden) {
    stop("La longitud de `b1` debe coincidir con nrow(W).")
  }

  if (length(v) != d_hidden) {
    stop("La longitud de `v` debe coincidir con nrow(W).")
  }

  invisible(TRUE)
}


#' Propagacion hacia adelante en un MLP simple RNAS
#'
#' Calcula potenciales ocultos, activaciones ocultas y predicciones de un
#' perceptron multicapa simple con una capa oculta.
#'
#' @param X Matriz numerica de entradas.
#' @param params Lista con parametros `W`, `b1`, `v` y `b2`.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#' @param devolver_cache Valor logico. Si es `TRUE`, devuelve tambien `Z1` y `H`.
#'
#' @return Si `devolver_cache = FALSE`, retorna vector de predicciones.
#' Si `TRUE`, retorna lista con `Z1`, `H`, `y_hat` y `activation`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)
#' rnas_mlp_forward(X, params, activation = "tanh")
#'
#' @export
rnas_mlp_forward <- function(X,
                             params,
                             activation = "tanh",
                             devolver_cache = FALSE) {
  rnas_validar_params_mlp(params, X)

  if (!is.character(activation)) {
    stop("`activation` debe indicarse como nombre: 'sigmoid' o 'tanh'.")
  }

  act <- rnas_get_activation(activation)

  W <- params$W
  b1 <- params$b1
  v <- params$v
  b2 <- params$b2

  Z1 <- sweep(X %*% t(W), 2, b1, "+")
  H <- act$activation(Z1)
  y_hat <- as.numeric(H %*% v + b2)

  if (isTRUE(devolver_cache)) {
    return(list(
      Z1 = Z1,
      H = H,
      y_hat = y_hat,
      activation = act$name
    ))
  }

  y_hat
}


#' Calcular perdida MSE para un MLP simple RNAS
#'
#' Calcula la perdida cuadratica media entre valores observados y predicciones
#' generadas por un MLP simple.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param params Lista con parametros `W`, `b1`, `v` y `b2`.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#' @param devolver_detalle Valor logico. Si es `TRUE`, retorna predicciones y errores.
#'
#' @return Escalar de perdida o lista con detalle.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)
#' rnas_mlp_loss(X, y, params)
#'
#' @export
rnas_mlp_loss <- function(X,
                          y,
                          params,
                          activation = "tanh",
                          devolver_detalle = FALSE) {
  rnas_validar_params_mlp(params, X)

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (length(y) != nrow(X)) {
    stop("La longitud de `y` debe coincidir con el numero de filas de `X`.")
  }

  forward <- rnas_mlp_forward(
    X = X,
    params = params,
    activation = activation,
    devolver_cache = TRUE
  )

  error <- forward$y_hat - y
  loss <- mean(error^2)

  if (isTRUE(devolver_detalle)) {
    return(list(
      loss = loss,
      y_hat = forward$y_hat,
      error = error,
      Z1 = forward$Z1,
      H = forward$H,
      activation = forward$activation
    ))
  }

  loss
}


#' Retropropagacion para un MLP simple RNAS
#'
#' Calcula los gradientes analiticos de la perdida MSE respecto a todos los
#' parametros del MLP simple.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param params Lista con parametros `W`, `b1`, `v` y `b2`.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#'
#' @return Lista con gradientes, perdida, predicciones y errores.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)
#' rnas_mlp_backward(X, y, params)
#'
#' @export
rnas_mlp_backward <- function(X,
                              y,
                              params,
                              activation = "tanh") {
  rnas_validar_params_mlp(params, X)

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (length(y) != nrow(X)) {
    stop("La longitud de `y` debe coincidir con el numero de filas de `X`.")
  }

  if (!is.character(activation)) {
    stop("`activation` debe indicarse como nombre: 'sigmoid' o 'tanh'.")
  }

  act <- rnas_get_activation(activation)

  W <- params$W
  v <- params$v

  forward <- rnas_mlp_forward(
    X = X,
    params = params,
    activation = activation,
    devolver_cache = TRUE
  )

  Z1 <- forward$Z1
  H <- forward$H
  y_hat <- forward$y_hat

  n <- nrow(X)
  error <- y_hat - y

  grad_v <- as.numeric((2 / n) * crossprod(H, error))
  grad_b2 <- as.numeric((2 / n) * sum(error))

  dphi <- act$derivative(Z1)

  delta_hidden <- sweep(dphi, 2, v, "*")
  delta_hidden <- sweep(delta_hidden, 1, error, "*")

  grad_W <- (2 / n) * t(delta_hidden) %*% X
  grad_b1 <- as.numeric((2 / n) * colSums(delta_hidden))

  grad_norm <- sqrt(
    sum(grad_W^2) +
      sum(grad_b1^2) +
      sum(grad_v^2) +
      grad_b2^2
  )

  list(
    grad_W = grad_W,
    grad_b1 = grad_b1,
    grad_v = grad_v,
    grad_b2 = grad_b2,
    grad_norm = grad_norm,
    loss = mean(error^2),
    y_hat = y_hat,
    error = error,
    Z1 = Z1,
    H = H,
    activation = activation
  )
}


#' Actualizar parametros de un MLP simple RNAS
#'
#' Actualiza los parametros del MLP mediante descenso por gradiente.
#'
#' @param params Lista con parametros actuales.
#' @param grads Lista con gradientes calculados por `rnas_mlp_backward()`.
#' @param eta Tasa de aprendizaje positiva.
#'
#' @return Lista de parametros actualizados.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8)
#' params <- rnas_init_mlp(2, 3, seed = 123)
#' grads <- rnas_mlp_backward(X, y, params)
#' rnas_update_params_mlp(params, grads, eta = 0.1)
#'
#' @export
rnas_update_params_mlp <- function(params, grads, eta) {
  req_params <- c("W", "b1", "v", "b2")
  req_grads <- c("grad_W", "grad_b1", "grad_v", "grad_b2")

  if (!is.list(params) || !all(req_params %in% names(params))) {
    stop("`params` debe contener W, b1, v y b2.")
  }

  if (!is.list(grads) || !all(req_grads %in% names(grads))) {
    stop("`grads` debe contener grad_W, grad_b1, grad_v y grad_b2.")
  }

  if (!is.numeric(eta) || length(eta) != 1L || eta <= 0) {
    stop("`eta` debe ser un escalar numerico positivo.")
  }

  if (!all(dim(params$W) == dim(grads$grad_W))) {
    stop("`grad_W` debe tener la misma dimension que `W`.")
  }

  if (length(params$b1) != length(grads$grad_b1)) {
    stop("`grad_b1` debe tener la misma longitud que `b1`.")
  }

  if (length(params$v) != length(grads$grad_v)) {
    stop("`grad_v` debe tener la misma longitud que `v`.")
  }

  if (length(grads$grad_b2) != 1L) {
    stop("`grad_b2` debe ser escalar.")
  }

  list(
    W = params$W - eta * grads$grad_W,
    b1 = params$b1 - eta * grads$grad_b1,
    v = params$v - eta * grads$grad_v,
    b2 = as.numeric(params$b2 - eta * grads$grad_b2)
  )
}


#' Entrenar un MLP simple RNAS
#'
#' Ejecuta el entrenamiento discreto de un perceptron multicapa simple con una
#' capa oculta y salida escalar.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param d_hidden Numero de neuronas ocultas.
#' @param params0 Lista opcional de parametros iniciales. Si es `NULL`,
#' se inicializan internamente.
#' @param eta Tasa de aprendizaje positiva.
#' @param T Numero maximo de iteraciones.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#' @param init_sd Desviacion estandar usada si `params0 = NULL`.
#' @param seed Semilla opcional para reproducibilidad.
#' @param tol_loss Tolerancia opcional para cambio absoluto de perdida.
#' @param tol_grad Tolerancia opcional para norma del gradiente.
#' @param registrar_cada Frecuencia de registro de la trayectoria.
#'
#' @return Objeto de clase `rnas_mlp_train`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' res <- rnas_train_mlp(X, y, d_hidden = 3, eta = 0.1, T = 50, seed = 123)
#' res$metricas
#'
#' @export
rnas_train_mlp <- function(X,
                           y,
                           d_hidden = 3,
                           params0 = NULL,
                           eta = 0.1,
                           T = 100,
                           activation = "tanh",
                           init_sd = 0.1,
                           seed = NULL,
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

  if (is.null(params0)) {
    params <- rnas_init_mlp(
      d_input = ncol(X),
      d_hidden = d_hidden,
      init_sd = init_sd,
      seed = seed
    )
  } else {
    params <- params0
    rnas_validar_params_mlp(params, X)
  }

  trayectoria <- vector("list", T + 1L)

  parada <- "max_iter"
  iter_final <- T
  loss_anterior <- NA_real_

  for (k in 0:T) {
    grads <- rnas_mlp_backward(
      X = X,
      y = y,
      params = params,
      activation = activation
    )

    if (k %% registrar_cada == 0 || k == T) {
      trayectoria[[k + 1L]] <- data.frame(
        iter = k,
        loss = grads$loss,
        grad_norm = grads$grad_norm,
        W_norm = sqrt(sum(params$W^2)),
        b1_norm = sqrt(sum(params$b1^2)),
        v_norm = sqrt(sum(params$v^2)),
        b2 = params$b2,
        stringsAsFactors = FALSE
      )
    }

    if (!is.null(tol_grad) && grads$grad_norm < tol_grad) {
      parada <- "tol_grad"
      iter_final <- k
      break
    }

    if (!is.null(tol_loss) && !is.na(loss_anterior)) {
      if (abs(loss_anterior - grads$loss) < tol_loss) {
        parada <- "tol_loss"
        iter_final <- k
        break
      }
    }

    loss_anterior <- grads$loss

    if (k < T) {
      params <- rnas_update_params_mlp(
        params = params,
        grads = grads,
        eta = eta
      )
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

  y_hat_final <- rnas_mlp_forward(
    X = X,
    params = params,
    activation = activation
  )

  res <- list(
    params_final = params,
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
      d_input = ncol(X),
      d_hidden = nrow(params$W),
      eta = eta,
      T = T,
      activation = activation,
      init_sd = init_sd,
      seed = seed,
      tol_loss = tol_loss,
      tol_grad = tol_grad,
      registrar_cada = registrar_cada
    )
  )

  class(res) <- c("rnas_mlp_train", class(res))
  res
}


#' Predecir con un MLP simple RNAS entrenado
#'
#' Genera predicciones usando un objeto entrenado por `rnas_train_mlp()`
#' o usando parametros proporcionados directamente.
#'
#' @param object Objeto `rnas_mlp_train` o `NULL`.
#' @param X Matriz numerica de entradas.
#' @param params Lista de parametros. Se usa si `object = NULL`.
#' @param activation Activacion. Se usa si `object = NULL`.
#'
#' @return Vector numerico de predicciones.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3), ncol = 2, byrow = TRUE)
#' params <- rnas_init_mlp(2, 3, seed = 123)
#' rnas_predict_mlp(NULL, X, params = params)
#'
#' @export
rnas_predict_mlp <- function(object = NULL,
                             X,
                             params = NULL,
                             activation = "tanh") {
  if (!is.null(object)) {
    if (!inherits(object, "rnas_mlp_train")) {
      stop("`object` debe ser NULL o un objeto de clase 'rnas_mlp_train'.")
    }

    params <- object$params_final
    activation <- object$configuracion$activation
  }

  if (is.null(params)) {
    stop("Debe proporcionar `object` o `params`.")
  }

  rnas_mlp_forward(
    X = X,
    params = params,
    activation = activation
  )
}


#' Resumir entrenamiento de un MLP simple RNAS
#'
#' Genera una tabla resumen con las metricas principales de un entrenamiento
#' realizado con `rnas_train_mlp()`.
#'
#' @param object Objeto de clase `rnas_mlp_train`.
#'
#' @return Data frame con resumen del entrenamiento.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' res <- rnas_train_mlp(X, y, d_hidden = 3, eta = 0.1, T = 50, seed = 123)
#' rnas_resumen_entrenamiento_mlp(res)
#'
#' @export
rnas_resumen_entrenamiento_mlp <- function(object) {
  if (!inherits(object, "rnas_mlp_train")) {
    stop("`object` debe ser un objeto de clase 'rnas_mlp_train'.")
  }

  data.frame(
    activation = object$configuracion$activation,
    d_input = object$configuracion$d_input,
    d_hidden = object$configuracion$d_hidden,
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


#' Imprimir resumen de entrenamiento de MLP RNAS
#'
#' Metodo de impresion para objetos creados con `rnas_train_mlp()`.
#'
#' @param x Objeto de clase `rnas_mlp_train`.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return El objeto `x`, de forma invisible.
#'
#' @export
print.rnas_mlp_train <- function(x, ...) {
  cat("Entrenamiento de MLP simple RNAS\n")
  cat("---------------------------------\n")
  cat("Activacion       :", x$configuracion$activation, "\n")
  cat("Dimension entrada:", x$configuracion$d_input, "\n")
  cat("Dimension oculta :", x$configuracion$d_hidden, "\n")
  cat("Eta              :", x$configuracion$eta, "\n")
  cat("Iteracion final  :", x$metricas$iter_final, "\n")
  cat("Parada           :", x$metricas$parada, "\n")
  cat("Loss inicial     :", x$metricas$loss_inicial, "\n")
  cat("Loss final       :", x$metricas$loss_final, "\n")
  cat("Delta loss       :", x$metricas$delta_loss, "\n")
  cat("Descendente      :", x$metricas$descendente_global, "\n")

  invisible(x)
}
