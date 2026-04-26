#' Construir malla de perdida para una neurona RNAS
#'
#' Construye una malla de valores de perdida variando dos pesos de una neurona
#' y manteniendo fijo el sesgo. Esta funcion permite representar el paisaje
#' de perdida en dos dimensiones.
#'
#' @param X Matriz numerica de entradas. Debe tener al menos dos columnas.
#' @param y Vector numerico de valores observados.
#' @param w1_seq Vector numerico con valores para el primer peso.
#' @param w2_seq Vector numerico con valores para el segundo peso.
#' @param b Sesgo numerico fijo.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#'
#' @return Data frame con columnas `w1`, `w2` y `loss`.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' grid <- rnas_loss_grid_neuron(X, y,
#'                               w1_seq = seq(0.7, 0.9, length.out = 5),
#'                               w2_seq = seq(0.2, 0.4, length.out = 5),
#'                               b = 0.1)
#' head(grid)
#'
#' @export
rnas_loss_grid_neuron <- function(X,
                                  y,
                                  w1_seq,
                                  w2_seq,
                                  b,
                                  activation = "tanh") {
  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (ncol(X) < 2) {
    stop("`X` debe tener al menos dos columnas para construir la malla.")
  }

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (nrow(X) != length(y)) {
    stop("La longitud de `y` debe coincidir con el numero de filas de `X`.")
  }

  if (!is.numeric(w1_seq) || length(w1_seq) < 2) {
    stop("`w1_seq` debe ser un vector numerico con al menos dos valores.")
  }

  if (!is.numeric(w2_seq) || length(w2_seq) < 2) {
    stop("`w2_seq` debe ser un vector numerico con al menos dos valores.")
  }

  if (!is.numeric(b) || length(b) != 1L) {
    stop("`b` debe ser un escalar numerico.")
  }

  grid <- expand.grid(
    w1 = w1_seq,
    w2 = w2_seq,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  loss_vals <- numeric(nrow(grid))

  for (i in seq_len(nrow(grid))) {
    w <- c(grid$w1[i], grid$w2[i])

    loss_vals[i] <- rnas_loss_mse_neuron(
      X = X[, 1:2, drop = FALSE],
      y = y,
      w = w,
      b = b,
      activation = activation
    )
  }

  grid$loss <- loss_vals
  grid
}


#' Calcular Hessiano numerico para una neurona RNAS
#'
#' Aproxima numericamente la matriz Hessiana de la perdida MSE respecto
#' al vector de estado theta = c(w, b), usando diferencias finitas centrales.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta Vector numerico de estado con pesos y sesgo.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#' @param h Paso positivo para diferencias finitas.
#'
#' @return Matriz Hessiana aproximada.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta <- c(0.8, 0.3, 0.1)
#' H <- rnas_hessian_num_neuron(X, y, theta)
#' H
#'
#' @export
rnas_hessian_num_neuron <- function(X,
                                    y,
                                    theta,
                                    activation = "tanh",
                                    h = 1e-4) {
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

  if (!is.numeric(h) || length(h) != 1L || h <= 0) {
    stop("`h` debe ser un escalar numerico positivo.")
  }

  p <- length(theta)
  H <- matrix(0, nrow = p, ncol = p)

  loss_theta <- function(th) {
    pars <- rnas_unpack_neuron_params(th, d_input = ncol(X))

    rnas_loss_mse_neuron(
      X = X,
      y = y,
      w = pars$w,
      b = pars$b,
      activation = activation
    )
  }

  for (i in seq_len(p)) {
    for (j in seq_len(p)) {
      e_i <- rep(0, p)
      e_j <- rep(0, p)

      e_i[i] <- h
      e_j[j] <- h

      f_pp <- loss_theta(theta + e_i + e_j)
      f_pm <- loss_theta(theta + e_i - e_j)
      f_mp <- loss_theta(theta - e_i + e_j)
      f_mm <- loss_theta(theta - e_i - e_j)

      H[i, j] <- (f_pp - f_pm - f_mp + f_mm) / (4 * h^2)
    }
  }

  H <- (H + t(H)) / 2

  colnames(H) <- rownames(H) <- paste0("theta", seq_len(p))
  H
}


#' Calcular autovalores e indicadores del Hessiano
#'
#' Calcula autovalores de una matriz Hessiana y resume indicadores
#' geometricos basicos.
#'
#' @param H Matriz Hessiana numerica cuadrada.
#' @param tol Tolerancia para clasificar autovalores cercanos a cero.
#'
#' @return Lista con autovalores, minimo, maximo, numero de positivos,
#' negativos, cercanos a cero y clasificacion local.
#'
#' @examples
#' H <- diag(c(1, 2, 3))
#' rnas_autovalores_hessian(H)
#'
#' @export
rnas_autovalores_hessian <- function(H, tol = 1e-8) {
  if (!is.numeric(H) || !is.matrix(H)) {
    stop("`H` debe ser una matriz numerica.")
  }

  if (nrow(H) != ncol(H)) {
    stop("`H` debe ser una matriz cuadrada.")
  }

  if (!is.numeric(tol) || length(tol) != 1L || tol <= 0) {
    stop("`tol` debe ser un escalar numerico positivo.")
  }

  vals <- eigen(H, symmetric = TRUE, only.values = TRUE)$values

  n_pos <- sum(vals > tol)
  n_neg <- sum(vals < -tol)
  n_zero <- sum(abs(vals) <= tol)

  clasificacion <- if (n_pos == length(vals)) {
    "curvatura_positiva"
  } else if (n_neg == length(vals)) {
    "curvatura_negativa"
  } else if (n_pos > 0 && n_neg > 0) {
    "silla"
  } else {
    "plana_o_semidefinida"
  }

  list(
    eigenvalues = vals,
    lambda_min = min(vals),
    lambda_max = max(vals),
    n_pos = n_pos,
    n_neg = n_neg,
    n_zero = n_zero,
    clasificacion = clasificacion
  )
}


#' Calcular curvatura direccional
#'
#' Calcula la curvatura direccional v^T H v, normalizando la direccion
#' si no tiene norma unitaria.
#'
#' @param H Matriz Hessiana numerica cuadrada.
#' @param v Vector numerico de direccion.
#'
#' @return Escalar numerico con la curvatura direccional.
#'
#' @examples
#' H <- diag(c(1, 2))
#' rnas_curvatura_direccional(H, v = c(1, 1))
#'
#' @export
rnas_curvatura_direccional <- function(H, v) {
  if (!is.numeric(H) || !is.matrix(H)) {
    stop("`H` debe ser una matriz numerica.")
  }

  if (nrow(H) != ncol(H)) {
    stop("`H` debe ser una matriz cuadrada.")
  }

  if (!is.numeric(v) || !is.vector(v)) {
    stop("`v` debe ser un vector numerico.")
  }

  if (length(v) != ncol(H)) {
    stop("La longitud de `v` debe coincidir con la dimension de `H`.")
  }

  norm_v <- sqrt(sum(v^2))

  if (norm_v == 0) {
    stop("`v` no puede ser el vector cero.")
  }

  v_unit <- v / norm_v

  as.numeric(t(v_unit) %*% H %*% v_unit)
}


#' Resumen geometrico local para una neurona RNAS
#'
#' Calcula perdida, gradiente, Hessiano numerico, autovalores y curvatura
#' direccional en un estado theta.
#'
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param theta Vector numerico de estado con pesos y sesgo.
#' @param activation Nombre de activacion. Puede ser `"tanh"` o `"sigmoid"`.
#' @param h Paso positivo para diferencias finitas.
#' @param direccion Direccion opcional para curvatura. Si es `NULL`, se usa
#' la direccion del gradiente cuando su norma es positiva.
#'
#' @return Lista con indicadores geometricos locales.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta <- c(0.8, 0.3, 0.1)
#' rnas_resumen_geometria_neuron(X, y, theta)
#'
#' @export
rnas_resumen_geometria_neuron <- function(X,
                                          y,
                                          theta,
                                          activation = "tanh",
                                          h = 1e-4,
                                          direccion = NULL) {
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

  H <- rnas_hessian_num_neuron(
    X = X,
    y = y,
    theta = theta,
    activation = activation,
    h = h
  )

  eig <- rnas_autovalores_hessian(H)

  if (is.null(direccion)) {
    if (grad_norm > 0) {
      direccion <- grad_vec
      direccion_tipo <- "gradiente"
    } else {
      direccion <- rep(1, length(theta))
      direccion_tipo <- "unitaria_general"
    }
  } else {
    direccion_tipo <- "usuario"
  }

  curvatura <- rnas_curvatura_direccional(H, direccion)

  list(
    theta = theta,
    w = pars$w,
    b = pars$b,
    loss = grad$loss,
    grad = grad_vec,
    grad_norm = grad_norm,
    H = H,
    eigenvalues = eig$eigenvalues,
    lambda_min = eig$lambda_min,
    lambda_max = eig$lambda_max,
    n_pos = eig$n_pos,
    n_neg = eig$n_neg,
    n_zero = eig$n_zero,
    clasificacion = eig$clasificacion,
    curvatura = curvatura,
    direccion_tipo = direccion_tipo,
    activation = activation,
    h = h
  )
}


#' Evaluar geometria local sobre una trayectoria de neurona RNAS
#'
#' Evalua indicadores geometricos locales en estados seleccionados de una
#' trayectoria dinamica producida por `rnas_integrar_dinamica_neuron()`.
#'
#' @param object Objeto de clase `rnas_neuron_dynamics`.
#' @param X Matriz numerica de entradas.
#' @param y Vector numerico de valores observados.
#' @param iteraciones Vector de iteraciones a evaluar. Si es `NULL`, usa
#' algunas iteraciones representativas disponibles en la trayectoria.
#' @param h Paso positivo para diferencias finitas.
#'
#' @return Data frame con indicadores geometricos por iteracion.
#'
#' @examples
#' X <- matrix(c(0.5, -0.2,
#'               1.0,  0.3,
#'              -0.7,  0.8,
#'               0.0,  0.0), ncol = 2, byrow = TRUE)
#' y <- c(0.4, 0.8, -0.2, 0.1)
#' theta0 <- c(0.8, 0.3, 0.1)
#' dyn <- rnas_integrar_dinamica_neuron(X, y, theta0, T = 10)
#' rnas_geometria_trayectoria_neuron(dyn, X, y, iteraciones = c(0, 5, 10))
#'
#' @export
rnas_geometria_trayectoria_neuron <- function(object,
                                              X,
                                              y,
                                              iteraciones = NULL,
                                              h = 1e-4) {
  if (!inherits(object, "rnas_neuron_dynamics")) {
    stop("`object` debe ser un objeto de clase 'rnas_neuron_dynamics'.")
  }

  if (!is.numeric(X) || !is.matrix(X)) {
    stop("`X` debe ser una matriz numerica.")
  }

  if (!is.numeric(y) || !is.vector(y)) {
    stop("`y` debe ser un vector numerico.")
  }

  if (nrow(X) != length(y)) {
    stop("La longitud de `y` debe coincidir con el numero de filas de `X`.")
  }

  trayectoria <- object$trayectoria

  if (is.null(iteraciones)) {
    iteraciones <- unique(c(
      min(trayectoria$iter),
      stats::median(trayectoria$iter),
      max(trayectoria$iter)
    ))
    iteraciones <- as.integer(round(iteraciones))
  }

  iteraciones <- iteraciones[iteraciones %in% trayectoria$iter]

  if (length(iteraciones) == 0) {
    stop("No hay iteraciones validas para evaluar en la trayectoria.")
  }

  filas <- vector("list", length(iteraciones))

  for (i in seq_along(iteraciones)) {
    k <- iteraciones[i]

    fila_tray <- trayectoria[trayectoria$iter == k, , drop = FALSE][1, ]

    theta_cols <- grep("^theta[0-9]+$", names(fila_tray), value = TRUE)
    theta_cols <- theta_cols[order(as.integer(gsub("theta", "", theta_cols)))]

    theta <- as.numeric(fila_tray[1, theta_cols])

    geo <- rnas_resumen_geometria_neuron(
      X = X,
      y = y,
      theta = theta,
      activation = object$configuracion$activation,
      h = h
    )

    filas[[i]] <- data.frame(
      iter = k,
      time = fila_tray$time,
      loss = geo$loss,
      grad_norm = geo$grad_norm,
      lambda_min = geo$lambda_min,
      lambda_max = geo$lambda_max,
      n_pos = geo$n_pos,
      n_neg = geo$n_neg,
      n_zero = geo$n_zero,
      clasificacion = geo$clasificacion,
      curvatura = geo$curvatura,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, filas)
}
