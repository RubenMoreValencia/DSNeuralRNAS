test_that("rnas_eta_costo_local retorna costo finito", {
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

  grad <- rnas_grad_neuron(
    X = X,
    y = y,
    w = c(0.8, 0.3),
    b = 0.1,
    activation = "tanh"
  )

  grad_vec <- c(grad$grad_w, grad$grad_b)

  res <- rnas_eta_costo_local(
    X = X,
    y = y,
    theta = theta,
    grad = grad_vec,
    eta = 0.1,
    activation = "tanh",
    alpha = 0,
    beta = 0
  )

  expect_true(is.list(res))
  expect_true(is.finite(res$costo))
  expect_true(res$costo >= 0)
  expect_length(res$theta_plus, length(theta))
})


test_that("rnas_eta_costo_local valida eta positiva", {
  X <- matrix(c(0.5, -0.2, 1.0, 0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8)
  theta <- c(0.8, 0.3, 0.1)
  grad <- c(0.1, 0.2, 0.3)

  expect_error(
    rnas_eta_costo_local(X, y, theta, grad, eta = 0)
  )
})


test_that("rnas_eta_opt_local selecciona eta desde grilla", {
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

  grad <- rnas_grad_neuron(
    X = X,
    y = y,
    w = c(0.8, 0.3),
    b = 0.1,
    activation = "tanh"
  )

  grad_vec <- c(grad$grad_w, grad$grad_b)
  eta_grid <- c(0.01, 0.05, 0.1)

  opt <- rnas_eta_opt_local(
    X = X,
    y = y,
    theta = theta,
    grad = grad_vec,
    eta_grid = eta_grid,
    activation = "tanh"
  )

  expect_true(opt$eta_opt %in% eta_grid)
  expect_true(is.data.frame(opt$costos))
  expect_equal(nrow(opt$costos), length(eta_grid))
})


test_that("rnas_eta_opt_local rechaza grilla no positiva", {
  X <- matrix(c(0.5, -0.2, 1.0, 0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8)
  theta <- c(0.8, 0.3, 0.1)
  grad <- c(0.1, 0.2, 0.3)

  expect_error(
    rnas_eta_opt_local(X, y, theta, grad, eta_grid = c(0, 0.1))
  )
})


test_that("rnas_eta_policy_geo reduce eta con curvatura positiva", {
  pol_baja <- rnas_eta_policy_geo(
    eta0 = 0.1,
    kappa = 0.1,
    modo = "curvatura",
    alpha = 1,
    eta_min = 0.01,
    eta_max = 0.1
  )

  pol_alta <- rnas_eta_policy_geo(
    eta0 = 0.1,
    kappa = 2,
    modo = "curvatura",
    alpha = 1,
    eta_min = 0.01,
    eta_max = 0.1
  )

  expect_true(pol_alta$eta <= pol_baja$eta)
})


test_that("rnas_eta_policy_geo funciona con lambda_max", {
  pol <- rnas_eta_policy_geo(
    eta0 = 0.1,
    lambda_max = 1.5,
    modo = "lambda_max",
    alpha = 1,
    eta_min = 0.01,
    eta_max = 0.1
  )

  expect_true(is.numeric(pol$eta))
  expect_true(pol$eta >= 0.01)
  expect_true(pol$eta <= 0.1)
})


test_that("rnas_eta_policy_geo respeta cotas", {
  pol <- rnas_eta_policy_geo(
    eta0 = 0.1,
    kappa = 100,
    modo = "curvatura",
    alpha = 10,
    eta_min = 0.02,
    eta_max = 0.1
  )

  expect_equal(pol$eta, 0.02)
})


test_that("rnas_train_neuron_meta_geo retorna objeto estructurado", {
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

  res <- rnas_train_neuron_meta_geo(
    X = X,
    y = y,
    theta0 = theta0,
    eta0 = 0.1,
    metodo = "opt_local",
    eta_grid = c(0.01, 0.05, 0.1),
    T = 10,
    activation = "tanh"
  )

  expect_s3_class(res, "rnas_meta_geo_train")
  expect_true(all(c("theta_final", "trayectoria", "metricas", "configuracion") %in% names(res)))
  expect_equal(nrow(res$trayectoria), 11)
})


test_that("rnas_train_neuron_meta_geo reduce perdida con opt_local", {
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

  res <- rnas_train_neuron_meta_geo(
    X = X,
    y = y,
    theta0 = theta0,
    metodo = "opt_local",
    eta_grid = c(0.01, 0.05, 0.1),
    T = 20,
    activation = "tanh"
  )

  expect_true(res$metricas$loss_final < res$metricas$loss_inicial)
  expect_true(res$metricas$descendente_global)
})


test_that("rnas_train_neuron_meta_geo funciona con metodo curvatura", {
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

  res <- rnas_train_neuron_meta_geo(
    X = X,
    y = y,
    theta0 = theta0,
    metodo = "curvatura",
    alpha = 1,
    T = 10
  )

  expect_s3_class(res, "rnas_meta_geo_train")
  expect_true(all(res$trayectoria$eta >= 0.01))
  expect_true(all(res$trayectoria$eta <= 0.1))
})


test_that("rnas_train_neuron_meta_geo funciona con metodo lambda_max", {
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

  res <- rnas_train_neuron_meta_geo(
    X = X,
    y = y,
    theta0 = theta0,
    metodo = "lambda_max",
    alpha = 1,
    T = 10
  )

  expect_s3_class(res, "rnas_meta_geo_train")
  expect_true(all(res$trayectoria$eta >= 0.01))
  expect_true(all(res$trayectoria$eta <= 0.1))
})


test_that("rnas_resumen_meta_geo retorna data frame", {
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

  res <- rnas_train_neuron_meta_geo(X, y, theta0, T = 5)
  tab <- rnas_resumen_meta_geo(res)

  expect_true(is.data.frame(tab))
  expect_equal(nrow(tab), 1)
  expect_true(all(c("loss_inicial", "loss_final", "costo_acumulado") %in% names(tab)))
})


test_that("rnas_comparar_meta_politicas compara multiples politicas", {
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

  politicas <- list(
    opt_local = list(metodo = "opt_local", eta_grid = c(0.01, 0.05, 0.1)),
    curvatura = list(metodo = "curvatura", alpha = 1),
    lambda = list(metodo = "lambda_max", alpha = 1)
  )

  comp <- rnas_comparar_meta_politicas(
    X = X,
    y = y,
    theta0 = theta0,
    politicas = politicas,
    T = 5
  )

  expect_s3_class(comp, "rnas_meta_policy_comparison")
  expect_true(is.data.frame(comp$comparacion))
  expect_equal(nrow(comp$comparacion), 3)
})


test_that("rnas_comparar_meta_politicas rechaza lista vacia", {
  expect_error(
    rnas_comparar_meta_politicas(
      X = matrix(rnorm(8), ncol = 2),
      y = rnorm(4),
      theta0 = c(0.8, 0.3, 0.1),
      politicas = list()
    )
  )
})


test_that("print.rnas_meta_geo_train imprime resumen", {
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

  res <- rnas_train_neuron_meta_geo(X, y, theta0, T = 5)

  expect_output(print(res), "meta-control geometrico")
})


test_that("print.rnas_meta_policy_comparison imprime resumen", {
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

  politicas <- list(
    opt_local = list(metodo = "opt_local", eta_grid = c(0.01, 0.05, 0.1))
  )

  comp <- rnas_comparar_meta_politicas(X, y, theta0, politicas, T = 5)

  expect_output(print(comp), "meta-control geometrico")
})
