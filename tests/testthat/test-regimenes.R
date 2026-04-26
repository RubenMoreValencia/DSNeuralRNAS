test_that("rnas_media_movil conserva longitud", {
  x <- c(1, 2, 3, 4, 5)
  y <- rnas_media_movil(x, ventana = 3)

  expect_true(is.numeric(y))
  expect_length(y, length(x))
  expect_equal(y[1], 1)
  expect_equal(y[3], mean(c(1, 2, 3)))
  expect_equal(y[5], mean(c(3, 4, 5)))
})


test_that("rnas_media_movil valida ventana positiva", {
  expect_error(rnas_media_movil(c(1, 2, 3), ventana = 0))
})


test_that("rnas_calcular_senales_regimen agrega columnas dinamicas", {
  tray <- data.frame(
    iter = 0:4,
    loss = c(1, 0.8, 0.7, 0.69, 0.68),
    grad_norm = c(0.5, 0.4, 0.3, 0.2, 0.1),
    speed = c(0.05, 0.04, 0.03, 0.02, 0.01)
  )

  sen <- rnas_calcular_senales_regimen(tray, ventana = 2)

  expect_true(is.data.frame(sen))
  expect_true(all(c(
    "loss_suav", "grad_suav", "speed_suav",
    "delta_loss", "reduccion_relativa",
    "delta_grad", "delta_speed"
  ) %in% names(sen)))

  expect_equal(nrow(sen), nrow(tray))
  expect_true(is.na(sen$delta_loss[1]))
  expect_equal(sen$delta_loss[2], -0.2)
})


test_that("rnas_calcular_senales_regimen valida columnas requeridas", {
  tray <- data.frame(loss = c(1, 0.9))

  expect_error(rnas_calcular_senales_regimen(tray))
})


test_that("rnas_clasificar_regimenes asigna regimen a cada fila", {
  tray <- data.frame(
    iter = 0:4,
    loss = c(1, 0.8, 0.6, 0.59, 0.589),
    grad_norm = c(0.5, 0.4, 0.3, 0.1, 0.01),
    speed = c(0.05, 0.04, 0.03, 0.01, 0.001)
  )

  sen <- rnas_calcular_senales_regimen(tray, ventana = 1)
  reg <- rnas_clasificar_regimenes(
    sen,
    tau_loss = 0.01,
    tau_grad = 1e-3,
    eps_loss = 1e-8,
    eps_grad = 1e-4,
    usar_suavizado = FALSE
  )

  expect_true("regimen" %in% names(reg))
  expect_equal(length(reg$regimen), nrow(tray))
  expect_false(any(is.na(reg$regimen)))
  expect_equal(reg$regimen[1], "Ajuste inicial")
})


test_that("rnas_clasificar_regimenes detecta inestabilidad por aumento de perdida", {
  tray <- data.frame(
    iter = 0:3,
    loss = c(1, 0.8, 0.85, 0.7),
    grad_norm = c(0.5, 0.4, 0.3, 0.2),
    speed = c(0.05, 0.04, 0.03, 0.02)
  )

  sen <- rnas_calcular_senales_regimen(tray, ventana = 1)
  reg <- rnas_clasificar_regimenes(sen, usar_suavizado = FALSE)

  expect_true("Inestabilidad" %in% reg$regimen)
})


test_that("rnas_segmentar_regimenes agrupa tramos consecutivos", {
  df <- data.frame(
    iter = 0:5,
    loss = c(1, 0.8, 0.7, 0.69, 0.68, 0.67),
    grad_norm = c(0.5, 0.4, 0.3, 0.2, 0.1, 0.09),
    speed = c(0.05, 0.04, 0.03, 0.02, 0.01, 0.009),
    regimen = c("A", "B", "B", "C", "C", "C")
  )

  seg <- rnas_segmentar_regimenes(df)

  expect_true(is.data.frame(seg))
  expect_equal(nrow(seg), 3)
  expect_equal(seg$duracion, c(1, 2, 3))
  expect_equal(sum(seg$duracion), nrow(df))
})


test_that("rnas_segmentar_regimenes valida columna regimen", {
  df <- data.frame(
    iter = 0:2,
    loss = c(1, 0.8, 0.7),
    grad_norm = c(0.5, 0.4, 0.3),
    speed = c(0.05, 0.04, 0.03)
  )

  expect_error(rnas_segmentar_regimenes(df))
})


test_that("rnas_resumen_regimenes calcula frecuencias y proporciones", {
  df <- data.frame(
    loss = c(1, 0.8, 0.7, 0.6),
    grad_norm = c(0.5, 0.4, 0.3, 0.2),
    speed = c(0.05, 0.04, 0.03, 0.02),
    regimen = c("A", "B", "B", "A")
  )

  res <- rnas_resumen_regimenes(df)

  expect_true(is.data.frame(res))
  expect_equal(sum(res$frecuencia), nrow(df))
  expect_equal(sum(res$proporcion), 1)
  expect_true(all(c("loss_media", "grad_media", "speed_media") %in% names(res)))
})


test_that("rnas_analizar_regimenes_neuron retorna objeto completo", {
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
    T = 20,
    activation = "tanh"
  )

  ana <- rnas_analizar_regimenes_neuron(
    object = dyn,
    ventana = 3,
    tau_loss = 0.01,
    tau_grad = 1e-3,
    eps_loss = 1e-8,
    eps_grad = 1e-4
  )

  expect_s3_class(ana, "rnas_regimen_analysis")
  expect_true(all(c("trayectoria_regimen", "segmentos", "resumen", "configuracion") %in% names(ana)))
  expect_true(is.data.frame(ana$trayectoria_regimen))
  expect_true(is.data.frame(ana$segmentos))
  expect_true(is.data.frame(ana$resumen))
})


test_that("rnas_analizar_regimenes_neuron rechaza objeto no dinamico", {
  expect_error(rnas_analizar_regimenes_neuron(list()))
})


test_that("rnas_analizar_regimenes_neuron cubre toda la trayectoria en segmentos", {
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
    T = 30,
    activation = "tanh"
  )

  ana <- rnas_analizar_regimenes_neuron(dyn, ventana = 3)

  expect_equal(sum(ana$segmentos$duracion), nrow(ana$trayectoria_regimen))
})


test_that("print.rnas_regimen_analysis retorna invisiblemente el objeto", {
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
    T = 5,
    activation = "tanh"
  )

  ana <- rnas_analizar_regimenes_neuron(dyn)

  expect_output(print(ana), "Analisis de regimenes")
})
