test_that("rnas_init_mlp crea parametros con dimensiones correctas", {
  params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)

  expect_true(is.list(params))
  expect_equal(dim(params$W), c(3, 2))
  expect_length(params$b1, 3)
  expect_length(params$v, 3)
  expect_length(params$b2, 1)
})


test_that("rnas_validar_params_mlp acepta parametros correctos", {
  X <- matrix(rnorm(8), ncol = 2)
  params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)

  expect_true(rnas_validar_params_mlp(params, X))
})


test_that("rnas_validar_params_mlp rechaza dimensiones incorrectas", {
  X <- matrix(rnorm(8), ncol = 2)
  params <- rnas_init_mlp(d_input = 3, d_hidden = 3, seed = 123)

  expect_error(rnas_validar_params_mlp(params, X))
})


test_that("rnas_mlp_forward devuelve predicciones y cache compatibles", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)

  pred <- rnas_mlp_forward(X, params, activation = "tanh")

  expect_true(is.numeric(pred))
  expect_length(pred, nrow(X))

  cache <- rnas_mlp_forward(X, params, activation = "tanh", devolver_cache = TRUE)

  expect_true(is.list(cache))
  expect_equal(dim(cache$Z1), c(nrow(X), 3))
  expect_equal(dim(cache$H), c(nrow(X), 3))
  expect_length(cache$y_hat, nrow(X))
})


test_that("rnas_mlp_loss devuelve perdida no negativa", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2)
  params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)

  loss <- rnas_mlp_loss(X, y, params, activation = "tanh")

  expect_true(is.numeric(loss))
  expect_length(loss, 1)
  expect_true(loss >= 0)
})


test_that("rnas_mlp_loss con detalle retorna estructura completa", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2)
  params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)

  res <- rnas_mlp_loss(
    X = X,
    y = y,
    params = params,
    activation = "tanh",
    devolver_detalle = TRUE
  )

  expect_true(is.list(res))
  expect_true(all(c("loss", "y_hat", "error", "Z1", "H", "activation") %in% names(res)))
  expect_length(res$y_hat, nrow(X))
  expect_equal(dim(res$H), c(nrow(X), 3))
})


test_that("rnas_mlp_backward devuelve gradientes con dimensiones correctas", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2)
  params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)

  grad <- rnas_mlp_backward(X, y, params, activation = "tanh")

  expect_equal(dim(grad$grad_W), dim(params$W))
  expect_length(grad$grad_b1, length(params$b1))
  expect_length(grad$grad_v, length(params$v))
  expect_length(grad$grad_b2, 1)
  expect_true(grad$loss >= 0)
  expect_true(grad$grad_norm >= 0)
})


test_that("rnas_update_params_mlp actualiza parametros conservando dimensiones", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2)
  params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)
  grad <- rnas_mlp_backward(X, y, params, activation = "tanh")

  upd <- rnas_update_params_mlp(params, grad, eta = 0.1)

  expect_equal(dim(upd$W), dim(params$W))
  expect_length(upd$b1, length(params$b1))
  expect_length(upd$v, length(params$v))
  expect_length(upd$b2, 1)
})


test_that("rnas_train_mlp retorna objeto estructurado", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_mlp(
    X = X,
    y = y,
    d_hidden = 3,
    eta = 0.1,
    T = 20,
    activation = "tanh",
    seed = 123
  )

  expect_s3_class(res, "rnas_mlp_train")
  expect_true(all(c("params_final", "trayectoria", "metricas", "configuracion") %in% names(res)))
  expect_true(is.data.frame(res$trayectoria))
  expect_true(res$metricas$loss_final >= 0)
})


test_that("rnas_train_mlp reduce perdida en caso controlado", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_mlp(
    X = X,
    y = y,
    d_hidden = 3,
    eta = 0.1,
    T = 100,
    activation = "tanh",
    seed = 123
  )

  expect_true(res$metricas$loss_final < res$metricas$loss_inicial)
  expect_true(res$metricas$descendente_global)
})


test_that("rnas_predict_mlp predice desde objeto entrenado", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_mlp(
    X = X,
    y = y,
    d_hidden = 3,
    eta = 0.1,
    T = 20,
    activation = "tanh",
    seed = 123
  )

  pred <- rnas_predict_mlp(res, X)

  expect_true(is.numeric(pred))
  expect_length(pred, nrow(X))
})


test_that("rnas_predict_mlp predice con parametros directos", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3),
    ncol = 2,
    byrow = TRUE
  )

  params <- rnas_init_mlp(d_input = 2, d_hidden = 3, seed = 123)

  pred <- rnas_predict_mlp(
    object = NULL,
    X = X,
    params = params,
    activation = "tanh"
  )

  expect_true(is.numeric(pred))
  expect_length(pred, nrow(X))
})


test_that("rnas_resumen_entrenamiento_mlp retorna data frame", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_mlp(
    X = X,
    y = y,
    d_hidden = 3,
    eta = 0.1,
    T = 20,
    activation = "tanh",
    seed = 123
  )

  resumen <- rnas_resumen_entrenamiento_mlp(res)

  expect_true(is.data.frame(resumen))
  expect_equal(nrow(resumen), 1)
  expect_true(all(c("loss_inicial", "loss_final", "delta_loss") %in% names(resumen)))
})


test_that("rnas_train_mlp valida T entero positivo", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8)

  expect_error(
    rnas_train_mlp(
      X = X,
      y = y,
      d_hidden = 3,
      eta = 0.1,
      T = 0,
      seed = 123
    )
  )
})


test_that("rnas_train_mlp acepta criterio tol_grad", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  res <- rnas_train_mlp(
    X = X,
    y = y,
    d_hidden = 3,
    eta = 0.1,
    T = 50,
    activation = "tanh",
    seed = 123,
    tol_grad = 1e-10
  )

  expect_s3_class(res, "rnas_mlp_train")
})
