test_that("rnas_pack_neuron_params agrupa pesos y sesgo", {
  w <- c(0.8, 0.3)
  b <- 0.1

  theta <- rnas_pack_neuron_params(w, b)

  expect_true(is.numeric(theta))
  expect_equal(theta, c(0.8, 0.3, 0.1))
  expect_length(theta, length(w) + 1)
})


test_that("rnas_unpack_neuron_params recupera pesos y sesgo", {
  theta <- c(0.8, 0.3, 0.1)

  pars <- rnas_unpack_neuron_params(theta, d_input = 2)

  expect_true(is.list(pars))
  expect_equal(pars$w, c(0.8, 0.3))
  expect_equal(pars$b, 0.1)
})


test_that("rnas_unpack_neuron_params rechaza longitud incorrecta", {
  theta <- c(0.8, 0.3, 0.1)

  expect_error(rnas_unpack_neuron_params(theta, d_input = 3))
})


test_that("rnas_eval_eta acepta eta constante", {
  eta <- rnas_eval_eta(0.1, t = 0)

  expect_equal(eta, 0.1)
})


test_that("rnas_eval_eta acepta eta funcional", {
  eta_fun <- function(t, ...) 0.1 / (1 + t)

  eta0 <- rnas_eval_eta(eta_fun, t = 0)
  eta1 <- rnas_eval_eta(eta_fun, t = 1)

  expect_equal(eta0, 0.1)
  expect_equal(eta1, 0.05)
})


test_that("rnas_eval_eta rechaza eta no positiva", {
  expect_error(rnas_eval_eta(0, t = 0))
})


test_that("rnas_campo_gradiente_neuron devuelve campo compatible", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2)
  theta <- c(0.8, 0.3, 0.1)

  campo <- rnas_campo_gradiente_neuron(
    theta = theta,
    X = X,
    y = y,
    eta = 0.1,
    t = 0,
    activation = "tanh"
  )

  expect_true(is.list(campo))
  expect_length(campo$theta_dot, length(theta))
  expect_length(campo$grad, length(theta))
  expect_true(campo$loss >= 0)
  expect_true(campo$grad_norm >= 0)
  expect_true(campo$speed >= 0)
  expect_equal(campo$eta, 0.1)
})


test_that("rnas_campo_gradiente_neuron cumple theta_dot = -eta * grad", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2)
  theta <- c(0.8, 0.3, 0.1)
  eta <- 0.1

  campo <- rnas_campo_gradiente_neuron(
    theta = theta,
    X = X,
    y = y,
    eta = eta,
    t = 0,
    activation = "tanh"
  )

  expect_equal(campo$theta_dot, -eta * campo$grad)
})


test_that("rnas_integrar_dinamica_neuron retorna objeto estructurado", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta0 <- c(0.8, 0.3, 0.1)

  res <- rnas_integrar_dinamica_neuron(
    X = X,
    y = y,
    theta0 = theta0,
    eta = 0.1,
    dt = 1,
    T = 20,
    activation = "tanh"
  )

  expect_s3_class(res, "rnas_neuron_dynamics")
  expect_true(all(c("theta_final", "w_final", "b_final", "trayectoria", "metricas") %in% names(res)))
  expect_true(is.data.frame(res$trayectoria))
  expect_true(res$metricas$loss_final >= 0)
})


test_that("rnas_integrar_dinamica_neuron reduce perdida en caso controlado", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta0 <- c(0.8, 0.3, 0.1)

  res <- rnas_integrar_dinamica_neuron(
    X = X,
    y = y,
    theta0 = theta0,
    eta = 0.1,
    dt = 1,
    T = 50,
    activation = "tanh"
  )

  expect_true(res$metricas$loss_final < res$metricas$loss_inicial)
  expect_true(res$metricas$descendente_global)
})


test_that("rnas_integrar_dinamica_neuron registra T mas 1 estados", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta0 <- c(0.8, 0.3, 0.1)

  res <- rnas_integrar_dinamica_neuron(
    X = X,
    y = y,
    theta0 = theta0,
    eta = 0.1,
    dt = 1,
    T = 10,
    activation = "tanh"
  )

  expect_equal(nrow(res$trayectoria), 11)
  expect_equal(res$trayectoria$iter[1], 0)
  expect_equal(tail(res$trayectoria$iter, 1), 10)
})


test_that("rnas_integrar_dinamica_neuron acepta eta funcional", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta0 <- c(0.8, 0.3, 0.1)

  eta_fun <- function(t, ...) 0.1 / (1 + 0.01 * t)

  res <- rnas_integrar_dinamica_neuron(
    X = X,
    y = y,
    theta0 = theta0,
    eta = eta_fun,
    dt = 1,
    T = 10,
    activation = "tanh"
  )

  expect_s3_class(res, "rnas_neuron_dynamics")
  expect_true(all(res$trayectoria$eta > 0))
})


test_that("rnas_resumen_dinamica_neuron retorna data frame", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta0 <- c(0.8, 0.3, 0.1)

  res <- rnas_integrar_dinamica_neuron(
    X = X,
    y = y,
    theta0 = theta0,
    eta = 0.1,
    dt = 1,
    T = 20,
    activation = "tanh"
  )

  resumen <- rnas_resumen_dinamica_neuron(res)

  expect_true(is.data.frame(resumen))
  expect_equal(nrow(resumen), 1)
  expect_true(all(c("loss_inicial", "loss_final", "velocidad_media") %in% names(resumen)))
})


test_that("rnas_predict_dinamica_neuron predice desde objeto dinamico", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta0 <- c(0.8, 0.3, 0.1)

  res <- rnas_integrar_dinamica_neuron(
    X = X,
    y = y,
    theta0 = theta0,
    eta = 0.1,
    dt = 1,
    T = 20,
    activation = "tanh"
  )

  pred <- rnas_predict_dinamica_neuron(res, X)

  expect_true(is.numeric(pred))
  expect_length(pred, nrow(X))
})


test_that("rnas_integrar_dinamica_neuron valida dt positivo", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8)
  theta0 <- c(0.8, 0.3, 0.1)

  expect_error(
    rnas_integrar_dinamica_neuron(
      X = X,
      y = y,
      theta0 = theta0,
      eta = 0.1,
      dt = 0,
      T = 10
    )
  )
})
