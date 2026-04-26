test_that("rnas_validar_dimensiones_neurona acepta dimensiones correctas", {
  x <- c(1, 2)
  w <- c(0.5, -0.2)
  b <- 0.1

  expect_true(rnas_validar_dimensiones_neurona(x, w, b))
})


test_that("rnas_validar_dimensiones_neurona rechaza dimensiones incorrectas", {
  x <- c(1, 2, 3)
  w <- c(0.5, -0.2)
  b <- 0.1

  expect_error(rnas_validar_dimensiones_neurona(x, w, b))
})


test_that("rnas_neuron_forward devuelve una salida numerica", {
  x <- c(0.5, -0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  y_hat <- rnas_neuron_forward(x, w, b, activation = "tanh")

  expect_true(is.numeric(y_hat))
  expect_length(y_hat, 1)
})


test_that("rnas_neuron_forward con devolver_z retorna lista estructurada", {
  x <- c(0.5, -0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  res <- rnas_neuron_forward(
    x = x,
    w = w,
    b = b,
    activation = "tanh",
    devolver_z = TRUE
  )

  expect_true(is.list(res))
  expect_true(all(c("z", "y_hat", "activation") %in% names(res)))
  expect_equal(res$activation, "tanh")
})


test_that("rnas_neuron_forward_batch devuelve vector de predicciones", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  w <- c(0.8, 0.3)
  b <- 0.1

  y_hat <- rnas_neuron_forward_batch(X, w, b, activation = "tanh")

  expect_true(is.numeric(y_hat))
  expect_length(y_hat, nrow(X))
})


test_that("rnas_neuron_forward_batch valida dimensiones", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3),
    ncol = 2,
    byrow = TRUE
  )

  w <- c(0.8, 0.3, 0.1)
  b <- 0.1

  expect_error(rnas_neuron_forward_batch(X, w, b, activation = "tanh"))
})
