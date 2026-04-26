test_that("rnas_loss_grid_neuron construye malla con dimensiones correctas", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  w1_seq <- seq(0.7, 0.9, length.out = 5)
  w2_seq <- seq(0.2, 0.4, length.out = 4)

  grid <- rnas_loss_grid_neuron(
    X = X,
    y = y,
    w1_seq = w1_seq,
    w2_seq = w2_seq,
    b = 0.1,
    activation = "tanh"
  )

  expect_true(is.data.frame(grid))
  expect_equal(nrow(grid), length(w1_seq) * length(w2_seq))
  expect_true(all(c("w1", "w2", "loss") %in% names(grid)))
  expect_true(all(grid$loss >= 0))
})


test_that("rnas_loss_grid_neuron rechaza X con menos de dos columnas", {
  X <- matrix(c(0.5, 1.0, -0.7), ncol = 1)
  y <- c(0.4, 0.8, -0.2)

  expect_error(
    rnas_loss_grid_neuron(
      X = X,
      y = y,
      w1_seq = seq(0.7, 0.9, length.out = 5),
      w2_seq = seq(0.2, 0.4, length.out = 5),
      b = 0.1
    )
  )
})


test_that("rnas_hessian_num_neuron devuelve matriz cuadrada y simetrica", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta <- c(0.8, 0.3, 0.1)

  H <- rnas_hessian_num_neuron(
    X = X,
    y = y,
    theta = theta,
    activation = "tanh",
    h = 1e-4
  )

  expect_true(is.matrix(H))
  expect_equal(dim(H), c(length(theta), length(theta)))
  expect_equal(H, t(H), tolerance = 1e-8)
})


test_that("rnas_hessian_num_neuron valida h positivo", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8)
  theta <- c(0.8, 0.3, 0.1)

  expect_error(
    rnas_hessian_num_neuron(
      X = X,
      y = y,
      theta = theta,
      h = 0
    )
  )
})


test_that("rnas_autovalores_hessian calcula indicadores", {
  H <- diag(c(1, 2, 3))

  eig <- rnas_autovalores_hessian(H)

  expect_true(is.list(eig))
  expect_length(eig$eigenvalues, 3)
  expect_equal(eig$lambda_min, 1)
  expect_equal(eig$lambda_max, 3)
  expect_equal(eig$n_pos, 3)
  expect_equal(eig$n_neg, 0)
  expect_equal(eig$clasificacion, "curvatura_positiva")
})


test_that("rnas_autovalores_hessian detecta punto tipo silla", {
  H <- diag(c(1, -2, 0.5))

  eig <- rnas_autovalores_hessian(H)

  expect_equal(eig$n_pos, 2)
  expect_equal(eig$n_neg, 1)
  expect_equal(eig$clasificacion, "silla")
})


test_that("rnas_autovalores_hessian rechaza matriz no cuadrada", {
  H <- matrix(1:6, nrow = 2)

  expect_error(rnas_autovalores_hessian(H))
})


test_that("rnas_curvatura_direccional calcula escalar", {
  H <- diag(c(1, 2))
  curv <- rnas_curvatura_direccional(H, v = c(1, 1))

  expect_true(is.numeric(curv))
  expect_length(curv, 1)
  expect_true(curv > 0)
})


test_that("rnas_curvatura_direccional rechaza vector cero", {
  H <- diag(c(1, 2))

  expect_error(rnas_curvatura_direccional(H, v = c(0, 0)))
})


test_that("rnas_resumen_geometria_neuron retorna estructura completa", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta <- c(0.8, 0.3, 0.1)

  geo <- rnas_resumen_geometria_neuron(
    X = X,
    y = y,
    theta = theta,
    activation = "tanh",
    h = 1e-4
  )

  expect_true(is.list(geo))
  expect_true(all(c(
    "theta", "loss", "grad", "grad_norm", "H",
    "eigenvalues", "lambda_min", "lambda_max",
    "clasificacion", "curvatura"
  ) %in% names(geo)))

  expect_length(geo$theta, length(theta))
  expect_length(geo$grad, length(theta))
  expect_equal(dim(geo$H), c(length(theta), length(theta)))
  expect_length(geo$eigenvalues, length(theta))
  expect_true(geo$loss >= 0)
})


test_that("rnas_resumen_geometria_neuron acepta direccion de usuario", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)
  theta <- c(0.8, 0.3, 0.1)

  geo <- rnas_resumen_geometria_neuron(
    X = X,
    y = y,
    theta = theta,
    activation = "tanh",
    direccion = c(1, 0, 0)
  )

  expect_equal(geo$direccion_tipo, "usuario")
  expect_true(is.numeric(geo$curvatura))
})


test_that("rnas_geometria_trayectoria_neuron evalua iteraciones seleccionadas", {
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

  dyn <- rnas_integrar_dinamica_neuron(
    X = X,
    y = y,
    theta0 = theta0,
    eta = 0.1,
    dt = 1,
    T = 10,
    activation = "tanh"
  )

  geo_tray <- rnas_geometria_trayectoria_neuron(
    object = dyn,
    X = X,
    y = y,
    iteraciones = c(0, 5, 10),
    h = 1e-4
  )

  expect_true(is.data.frame(geo_tray))
  expect_equal(nrow(geo_tray), 3)
  expect_true(all(c(
    "iter", "loss", "grad_norm", "lambda_min",
    "lambda_max", "clasificacion", "curvatura"
  ) %in% names(geo_tray)))
})


test_that("rnas_geometria_trayectoria_neuron rechaza objeto no dinamico", {
  expect_error(
    rnas_geometria_trayectoria_neuron(
      object = list(),
      X = matrix(rnorm(8), ncol = 2),
      y = rnorm(4)
    )
  )
})
