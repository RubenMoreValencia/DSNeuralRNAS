test_that("rnas_preparar_features_std2 alinea X_t con y_t+h", {
  datos <- data.frame(
    y = 1:10,
    C = seq(0.1, 1, length.out = 10),
    x = seq(1, 2, length.out = 10),
    e = seq(-0.5, 0.5, length.out = 10)
  )

  prep <- rnas_preparar_features_std2(
    datos_std2 = datos,
    target = "y",
    features = c("C", "x", "e"),
    horizonte = 1
  )

  expect_s3_class(prep, "rnas_std2_features")
  expect_equal(nrow(prep$X_rnas), 9)
  expect_equal(length(prep$y_rnas), 9)
  expect_equal(prep$y_rnas[1], datos$y[2])
  expect_equal(prep$datos_alineados$t_origen[1], 1)
  expect_equal(prep$datos_alineados$t_objetivo[1], 2)
})


test_that("rnas_preparar_features_std2 valida columnas", {
  datos <- data.frame(y = 1:5, C = 1:5)

  expect_error(
    rnas_preparar_features_std2(
      datos_std2 = datos,
      target = "z",
      features = "C"
    )
  )

  expect_error(
    rnas_preparar_features_std2(
      datos_std2 = datos,
      target = "y",
      features = "no_existe"
    )
  )
})


test_that("rnas_integrar_std2 entrena modelo puente", {
  datos <- data.frame(
    y = seq(-0.5, 0.5, length.out = 30),
    C = seq(-0.3, 0.3, length.out = 30),
    x = sin(seq(0, 2, length.out = 30)),
    e = cos(seq(0, 2, length.out = 30)) / 10
  )

  res <- rnas_integrar_std2(
    datos_std2 = datos,
    target = "y",
    features = c("C", "x", "e"),
    horizonte = 1,
    eta0 = 0.05,
    T = 10,
    activation = "tanh"
  )

  expect_s3_class(res, "rnas_std2_integration")
  expect_true(is.list(res$modelo_rnas))
  expect_true(is.data.frame(res$modelo_rnas$trayectoria))
  expect_equal(res$configuracion$tipo_integracion, "STD2-RNAS")
})


test_that("rnas_preparar_trayectorias_simuds genera pares por escenario", {
  tray <- data.frame(
    escenario = rep(c("A", "B"), each = 5),
    t = rep(1:5, times = 2),
    s = c(1:5, 2:6)
  )

  prep <- rnas_preparar_trayectorias_simuds(
    trayectorias_simuds = tray,
    estado_cols = "s",
    escenario_col = "escenario",
    tiempo_col = "t",
    horizonte = 1
  )

  expect_s3_class(prep, "rnas_simuds_pairs")
  expect_equal(nrow(prep$pares), 8)
  expect_equal(ncol(prep$X_rnas), 1)
  expect_equal(length(prep$y_rnas), 8)
  expect_true("s_target" %in% names(prep$pares))
})


test_that("rnas_preparar_trayectorias_simuds valida columnas", {
  tray <- data.frame(t = 1:5, s = 1:5)

  expect_error(
    rnas_preparar_trayectorias_simuds(
      trayectorias_simuds = tray,
      estado_cols = "x"
    )
  )
})


test_that("rnas_integrar_simuds entrena modelo sobre transiciones", {
  tray <- data.frame(
    escenario = rep(c("A", "B"), each = 15),
    t = rep(1:15, times = 2),
    s = c(seq(-0.5, 0.5, length.out = 15),
          seq(-0.4, 0.6, length.out = 15))
  )

  res <- rnas_integrar_simuds(
    trayectorias_simuds = tray,
    estado_cols = "s",
    target_estado = "s",
    escenario_col = "escenario",
    tiempo_col = "t",
    horizonte = 1,
    eta0 = 0.05,
    T = 10,
    activation = "tanh"
  )

  expect_s3_class(res, "rnas_simuds_integration")
  expect_true(is.data.frame(res$modelo_rnas$trayectoria))
  expect_equal(res$configuracion$tipo_integracion, "SimuDS-RNAS")
})


test_that("rnas_integrar_simuds exige target si hay varios estados", {
  tray <- data.frame(
    t = 1:5,
    s1 = 1:5,
    s2 = 2:6
  )

  expect_error(
    rnas_integrar_simuds(
      trayectorias_simuds = tray,
      estado_cols = c("s1", "s2")
    )
  )
})


test_that("rnas_formular_fnl construye objetivo evaluable", {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2)

  fnl <- rnas_formular_fnl(
    X = X,
    y = y,
    activation = "tanh"
  )

  theta <- c(0.8, 0.3, 0.1)
  val <- fnl$objetivo(theta)

  expect_s3_class(fnl, "rnas_fnl_formulation")
  expect_true(is.numeric(val))
  expect_true(val >= 0)
  expect_true(fnl$es_factible(theta))
})


test_that("rnas_formular_fnl evalua restricciones", {
  X <- matrix(c(0.5, -0.2, 1.0, 0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8)

  restricciones <- list(
    function(theta) sum(theta^2) - 10,
    function(theta) theta[1] - 2
  )

  fnl <- rnas_formular_fnl(
    X = X,
    y = y,
    restricciones = restricciones
  )

  theta <- c(0.8, 0.3, 0.1)

  vals <- fnl$evaluar_restricciones(theta)

  expect_length(vals, 2)
  expect_true(fnl$es_factible(theta))
})


test_that("rnas_formular_fnl detecta restriccion no funcion", {
  X <- matrix(c(0.5, -0.2, 1.0, 0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8)

  expect_error(
    rnas_formular_fnl(
      X = X,
      y = y,
      restricciones = list(1)
    )
  )
})


test_that("rnas_resumen_integracion resume STD2, SimuDS y FNL", {
  datos <- data.frame(
    y = seq(-0.5, 0.5, length.out = 20),
    C = seq(-0.3, 0.3, length.out = 20),
    x = sin(seq(0, 2, length.out = 20))
  )

  std2 <- rnas_integrar_std2(
    datos_std2 = datos,
    target = "y",
    features = c("C", "x"),
    T = 5
  )

  res_std2 <- rnas_resumen_integracion(std2)

  expect_true(is.data.frame(res_std2))
  expect_equal(res_std2$integracion, "STD2-RNAS")

  X <- matrix(c(0.5, -0.2, 1.0, 0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8)

  fnl <- rnas_formular_fnl(X, y)
  res_fnl <- rnas_resumen_integracion(fnl)

  expect_equal(res_fnl$integracion, "FNL-RNAS")
})


test_that("rnas_resumen_integracion rechaza objeto desconocido", {
  expect_error(rnas_resumen_integracion(list()))
})


test_that("metodos print de integracion producen salida", {
  datos <- data.frame(
    y = seq(-0.5, 0.5, length.out = 20),
    C = seq(-0.3, 0.3, length.out = 20),
    x = sin(seq(0, 2, length.out = 20))
  )

  prep <- rnas_preparar_features_std2(
    datos_std2 = datos,
    target = "y",
    features = c("C", "x")
  )

  expect_output(print(prep), "Preparacion STD2")

  std2 <- rnas_integrar_std2(
    datos_std2 = datos,
    target = "y",
    features = c("C", "x"),
    T = 5
  )

  expect_output(print(std2), "Integracion STD2-RNAS")

  X <- matrix(c(0.5, -0.2, 1.0, 0.3), ncol = 2, byrow = TRUE)
  y <- c(0.4, 0.8)

  fnl <- rnas_formular_fnl(X, y)

  expect_output(print(fnl), "Formulacion FNL-RNAS")
})
