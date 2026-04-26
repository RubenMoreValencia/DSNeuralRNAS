test_that("rnas_validar_datos_supervisados acepta datos correctos", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8)
  w <- c(0.8, 0.3)
  b <- 0.1

  expect_true(rnas_validar_datos_supervisados(X, y, w, b))
})


test_that("rnas_validar_datos_supervisados rechaza dimensiones incorrectas", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8, 0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  expect_error(rnas_validar_datos_supervisados(X, y, w, b))
})


test_that("rnas_loss_mse_neuron devuelve perdida no negativa", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3,
                -0.7,  0.8), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8, -0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  loss <- rnas_loss_mse_neuron(X, y, w, b, activation = "tanh")

  expect_true(is.numeric(loss))
  expect_length(loss, 1)
  expect_true(loss >= 0)
})


test_that("rnas_loss_mse_neuron con detalle retorna estructura completa", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3,
                -0.7,  0.8), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8, -0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  res <- rnas_loss_mse_neuron(
    X = X,
    y = y,
    w = w,
    b = b,
    activation = "tanh",
    devolver_detalle = TRUE
  )

  expect_true(is.list(res))
  expect_true(all(c("loss", "y_hat", "error", "z", "activation") %in% names(res)))
  expect_length(res$y_hat, nrow(X))
  expect_length(res$error, nrow(X))
  expect_true(res$loss >= 0)
})


test_that("rnas_grad_neuron devuelve gradientes con dimensiones correctas", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3,
                -0.7,  0.8), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8, -0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  grad <- rnas_grad_neuron(X, y, w, b, activation = "tanh")

  expect_true(is.list(grad))
  expect_length(grad$grad_w, length(w))
  expect_length(grad$grad_b, 1)
  expect_true(is.numeric(grad$loss))
  expect_true(grad$loss >= 0)
})


test_that("rnas_grad_num_neuron devuelve gradientes numericos", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3,
                -0.7,  0.8), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8, -0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  grad_num <- rnas_grad_num_neuron(X, y, w, b, activation = "tanh")

  expect_true(is.list(grad_num))
  expect_length(grad_num$grad_w, length(w))
  expect_length(grad_num$grad_b, 1)
})


test_that("rnas_grad_check_neuron verifica gradiente tanh", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3,
                -0.7,  0.8), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8, -0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  check <- rnas_grad_check_neuron(
    X = X,
    y = y,
    w = w,
    b = b,
    activation = "tanh",
    h = 1e-6,
    tol = 1e-5
  )

  expect_true(is.list(check))
  expect_true(check$verificado)
  expect_true(check$error_rel < 1e-5)
})


test_that("rnas_grad_check_neuron verifica gradiente sigmoid", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3,
                -0.7,  0.8), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8, -0.2)
  w <- c(0.8, 0.3)
  b <- 0.1

  check <- rnas_grad_check_neuron(
    X = X,
    y = y,
    w = w,
    b = b,
    activation = "sigmoid",
    h = 1e-6,
    tol = 1e-5
  )

  expect_true(is.list(check))
  expect_true(check$verificado)
  expect_true(check$error_rel < 1e-5)
})


test_that("rnas_grad_num_neuron rechaza h no positivo", {
  X <- matrix(c(0.5, -0.2,
                1.0,  0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8)
  w <- c(0.8, 0.3)
  b <- 0.1

  expect_error(rnas_grad_num_neuron(X, y, w, b, h = 0))
})
