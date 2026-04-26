test_that("rnas_update_params_neuron actualiza correctamente", {
  w <- c(0.8, 0.3)
  b <- 0.1
  grad_w <- c(-0.01, 0.02)
  grad_b <- -0.03
  eta <- 0.1

  upd <- rnas_update_params_neuron(w, b, grad_w, grad_b, eta)

  expect_equal(upd$w_new, c(0.801, 0.298))
  expect_equal(upd$b_new, 0.103)
})


test_that("rnas_update_params_neuron valida eta positivo", {
  w <- c(0.8, 0.3)
  b <- 0.1
  grad_w <- c(-0.01, 0.02)
  grad_b <- -0.03

  expect_error(rnas_update_params_neuron(w, b, grad_w, grad_b, eta = 0))
})


test_that("rnas_train_neuron retorna objeto estructurado", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_neuron(
    X = X,
    y = y,
    w0 = c(0.8, 0.3),
    b0 = 0.1,
    eta = 0.1,
    T = 20,
    activation = "tanh"
  )

  expect_s3_class(res, "rnas_neuron_train")
  expect_true(all(c("w_final", "b_final", "trayectoria", "metricas", "configuracion") %in% names(res)))
  expect_true(is.data.frame(res$trayectoria))
  expect_true(res$metricas$loss_final >= 0)
})


test_that("rnas_train_neuron reduce perdida en caso controlado", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_neuron(
    X = X,
    y = y,
    w0 = c(0.8, 0.3),
    b0 = 0.1,
    eta = 0.1,
    T = 50,
    activation = "tanh"
  )

  expect_true(res$metricas$loss_final < res$metricas$loss_inicial)
  expect_true(res$metricas$descendente_global)
})


test_that("rnas_train_neuron registra trayectoria compatible", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_neuron(
    X = X,
    y = y,
    w0 = c(0.8, 0.3),
    b0 = 0.1,
    eta = 0.1,
    T = 10,
    activation = "tanh"
  )

  expect_true(all(c("iter", "loss", "grad_norm", "b", "w1", "w2") %in% names(res$trayectoria)))
  expect_equal(res$trayectoria$iter[1], 0)
  expect_equal(tail(res$trayectoria$iter, 1), 10)
})


test_that("rnas_predict_neuron predice desde objeto entrenado", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_neuron(
    X = X,
    y = y,
    w0 = c(0.8, 0.3),
    b0 = 0.1,
    eta = 0.1,
    T = 20,
    activation = "tanh"
  )

  pred <- rnas_predict_neuron(res, X)

  expect_true(is.numeric(pred))
  expect_length(pred, nrow(X))
})


test_that("rnas_predict_neuron predice con parametros directos", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3),
    ncol = 2,
    byrow = TRUE
  )

  pred <- rnas_predict_neuron(
    object = NULL,
    X = X,
    w = c(0.8, 0.3),
    b = 0.1,
    activation = "tanh"
  )

  expect_true(is.numeric(pred))
  expect_length(pred, nrow(X))
})


test_that("rnas_resumen_entrenamiento_neuron retorna data frame", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_neuron(
    X = X,
    y = y,
    w0 = c(0.8, 0.3),
    b0 = 0.1,
    eta = 0.1,
    T = 20,
    activation = "tanh"
  )

  resumen <- rnas_resumen_entrenamiento_neuron(res)

  expect_true(is.data.frame(resumen))
  expect_equal(nrow(resumen), 1)
  expect_true(all(c("loss_inicial", "loss_final", "delta_loss") %in% names(resumen)))
})


test_that("rnas_train_neuron valida T entero positivo", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8)

  expect_error(
    rnas_train_neuron(
      X = X,
      y = y,
      w0 = c(0.8, 0.3),
      b0 = 0.1,
      eta = 0.1,
      T = 0
    )
  )
})


test_that("rnas_train_neuron acepta criterio tol_grad", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_neuron(
    X = X,
    y = y,
    w0 = c(0.8, 0.3),
    b0 = 0.1,
    eta = 0.1,
    T = 50,
    activation = "tanh",
    tol_grad = 1e-10
  )

  expect_s3_class(res, "rnas_neuron_train")
})
