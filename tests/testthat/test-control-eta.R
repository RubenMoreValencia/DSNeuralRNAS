test_that("rnas_clip_eta aplica cotas correctamente", {
  expect_equal(rnas_clip_eta(0.5, eta_min = 0.01, eta_max = 0.1), 0.1)
  expect_equal(rnas_clip_eta(0.001, eta_min = 0.01, eta_max = 0.1), 0.01)
  expect_equal(rnas_clip_eta(0.05, eta_min = 0.01, eta_max = 0.1), 0.05)
})


test_that("rnas_clip_eta valida cotas", {
  expect_error(rnas_clip_eta(0.1, eta_min = 0.2, eta_max = 0.1))
  expect_error(rnas_clip_eta(0.1, eta_min = 0, eta_max = 0.1))
})


test_that("rnas_eta_policy constante mantiene eta0", {
  pol <- rnas_eta_policy(
    policy = "constante",
    k = 5,
    eta_prev = 0.05,
    eta0 = 0.1,
    loss = 0.5,
    loss_prev = 0.6
  )

  expect_equal(pol$eta, 0.1)
  expect_equal(pol$tipo, "constante")
})


test_that("rnas_eta_policy temporal reduce con k", {
  pol <- rnas_eta_policy(
    policy = list(tipo = "temporal", alpha = 0.1),
    k = 10,
    eta_prev = 0.1,
    eta0 = 0.1,
    loss = 0.5,
    loss_prev = 0.6
  )

  expect_equal(pol$eta, 0.1 / (1 + 0.1 * 10))
  expect_equal(pol$accion, "decaimiento_temporal")
})


test_that("rnas_eta_policy mejora reduce si mejora relativa es baja", {
  pol <- rnas_eta_policy(
    policy = list(tipo = "mejora", tau_loss = 0.1, gamma = 0.5),
    k = 2,
    eta_prev = 0.1,
    eta0 = 0.1,
    loss = 0.95,
    loss_prev = 1.0
  )

  expect_equal(pol$eta, 0.05)
  expect_equal(pol$accion, "reducir_por_baja_mejora")
})


test_that("rnas_eta_policy regimen reduce en refinamiento", {
  pol <- rnas_eta_policy(
    policy = list(tipo = "regimen", gamma_refinamiento = 0.9),
    k = 5,
    eta_prev = 0.1,
    eta0 = 0.1,
    loss = 0.5,
    loss_prev = 0.6,
    regimen = "Refinamiento"
  )

  expect_equal(pol$eta, 0.09)
  expect_equal(pol$accion, "reducir_refinamiento")
})


test_that("rnas_eta_policy inestabilidad reduce si aumenta perdida", {
  pol <- rnas_eta_policy(
    policy = list(tipo = "inestabilidad", gamma = 0.5),
    k = 3,
    eta_prev = 0.1,
    eta0 = 0.1,
    loss = 1.2,
    loss_prev = 1.0
  )

  expect_equal(pol$eta, 0.05)
  expect_equal(pol$accion, "reducir_por_aumento_loss")
})


test_that("rnas_eta_policy rechaza politica desconocida", {
  expect_error(
    rnas_eta_policy(
      policy = "desconocida",
      k = 1,
      eta_prev = 0.1,
      eta0 = 0.1
    )
  )
})


test_that("rnas_train_neuron_control_eta retorna objeto estructurado", {
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

  res <- rnas_train_neuron_control_eta(
    X = X,
    y = y,
    theta0 = theta0,
    eta0 = 0.1,
    policy = "constante",
    T = 20,
    activation = "tanh"
  )

  expect_s3_class(res, "rnas_control_eta_train")
  expect_true(all(c("theta_final", "trayectoria", "metricas", "configuracion") %in% names(res)))
  expect_true(is.data.frame(res$trayectoria))
  expect_equal(nrow(res$trayectoria), 21)
})


test_that("rnas_train_neuron_control_eta reduce perdida con politica constante", {
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

  res <- rnas_train_neuron_control_eta(
    X = X,
    y = y,
    theta0 = theta0,
    eta0 = 0.1,
    policy = "constante",
    T = 50,
    activation = "tanh"
  )

  expect_true(res$metricas$loss_final < res$metricas$loss_inicial)
  expect_true(res$metricas$descendente_global)
})


test_that("rnas_train_neuron_control_eta respeta cotas de eta", {
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

  res <- rnas_train_neuron_control_eta(
    X = X,
    y = y,
    theta0 = theta0,
    eta0 = 0.1,
    policy = list(tipo = "temporal", alpha = 1),
    eta_min = 0.02,
    eta_max = 0.08,
    T = 20,
    activation = "tanh"
  )

  expect_true(all(res$trayectoria$eta >= 0.02))
  expect_true(all(res$trayectoria$eta <= 0.08))
})


test_that("rnas_resumen_control_eta retorna data frame", {
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

  res <- rnas_train_neuron_control_eta(X, y, theta0, T = 20)
  tab <- rnas_resumen_control_eta(res)

  expect_true(is.data.frame(tab))
  expect_equal(nrow(tab), 1)
  expect_true(all(c("loss_inicial", "loss_final", "eta_media") %in% names(tab)))
})


test_that("rnas_comparar_politicas_eta compara multiples politicas", {
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
    constante = "constante",
    temporal = list(tipo = "temporal", alpha = 0.01),
    mejora = list(tipo = "mejora", tau_loss = 0.01, gamma = 0.95)
  )

  comp <- rnas_comparar_politicas_eta(
    X = X,
    y = y,
    theta0 = theta0,
    politicas = politicas,
    eta0 = 0.1,
    T = 20,
    activation = "tanh"
  )

  expect_s3_class(comp, "rnas_eta_policy_comparison")
  expect_true(is.data.frame(comp$comparacion))
  expect_equal(nrow(comp$comparacion), 3)
  expect_true(all(c("politica", "loss_final", "eta_media") %in% names(comp$comparacion)))
})


test_that("rnas_comparar_politicas_eta rechaza lista vacia", {
  expect_error(
    rnas_comparar_politicas_eta(
      X = matrix(rnorm(8), ncol = 2),
      y = rnorm(4),
      theta0 = c(0.8, 0.3, 0.1),
      politicas = list()
    )
  )
})


test_that("print.rnas_control_eta_train imprime resumen", {
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

  res <- rnas_train_neuron_control_eta(X, y, theta0, T = 5)

  expect_output(print(res), "Entrenamiento RNAS con control de eta")
})


test_that("print.rnas_eta_policy_comparison imprime resumen", {
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
    constante = "constante",
    temporal = list(tipo = "temporal", alpha = 0.01)
  )

  comp <- rnas_comparar_politicas_eta(X, y, theta0, politicas, T = 5)

  expect_output(print(comp), "Comparacion de politicas")
})

test_that("politica por regimen aplica factores gamma", {
  X <- matrix(
    c(
      0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0
    ),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  theta0 <- c(0.8, 0.3, 0.1)

  res <- rnas_train_neuron_control_eta(
    X = X,
    y = y,
    theta0 = theta0,
    eta0 = 0.1,
    policy = list(
      tipo = "regimen",
      gamma_refinamiento = 0.98,
      gamma_saturacion = 0.90,
      gamma_inestabilidad = 0.50,
      gamma_estabilizacion = 0.95
    ),
    eta_min = 0.001,
    eta_max = 0.1,
    T = 20,
    activation = "tanh"
  )

  expect_s3_class(res, "rnas_control_eta_train")
  expect_true(is.data.frame(res$trayectoria))
  expect_true("eta" %in% names(res$trayectoria))
  expect_true(all(res$trayectoria$eta >= 0.001))
  expect_true(all(res$trayectoria$eta <= 0.1))
  expect_true(all(is.finite(res$trayectoria$eta)))
})
