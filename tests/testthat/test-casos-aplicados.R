test_that("rnas_caso_std2_controlado retorna caso estructurado", {
  caso <- rnas_caso_std2_controlado(
    n = 30,
    horizonte = 1,
    eta0 = 0.05,
    T = 10,
    seed = 123
  )

  expect_s3_class(caso, "rnas_caso_aplicado_std2")
  expect_true(is.data.frame(caso$resumen))
  expect_equal(caso$resumen$caso, "STD2-RNAS")
  expect_equal(caso$resumen$n_obs, 29)
  expect_true(caso$resumen$loss_final < caso$resumen$loss_inicial)
})


test_that("rnas_caso_std2_controlado valida n", {
  expect_error(
    rnas_caso_std2_controlado(n = 2, horizonte = 1)
  )
})


test_that("rnas_caso_simuds_controlado retorna caso estructurado", {
  caso <- rnas_caso_simuds_controlado(
    n = 20,
    horizonte = 1,
    eta0 = 0.05,
    T = 10,
    seed = 123
  )

  expect_s3_class(caso, "rnas_caso_aplicado_simuds")
  expect_true(is.data.frame(caso$resumen))
  expect_equal(caso$resumen$caso, "SimuDS-RNAS")
  expect_equal(caso$resumen$n_obs, 3 * (20 - 1))
  expect_true(caso$resumen$loss_final < caso$resumen$loss_inicial)
})


test_that("rnas_caso_simuds_controlado valida n", {
  expect_error(
    rnas_caso_simuds_controlado(n = 2, horizonte = 1)
  )
})


test_that("rnas_caso_fnl_controlado retorna caso factible", {
  caso <- rnas_caso_fnl_controlado(
    n = 30,
    seed = 123
  )

  expect_s3_class(caso, "rnas_caso_aplicado_fnl")
  expect_true(is.data.frame(caso$resumen))
  expect_equal(caso$resumen$caso, "FNL-RNAS")
  expect_true(caso$resumen$factible)
  expect_true(is.data.frame(caso$diagnostico))
})


test_that("rnas_caso_fnl_controlado valida n", {
  expect_error(
    rnas_caso_fnl_controlado(n = 3)
  )
})


test_that("rnas_consolidar_casos consolida resumenes", {
  c1 <- rnas_caso_std2_controlado(n = 30, T = 10)
  c2 <- rnas_caso_simuds_controlado(n = 20, T = 10)
  c3 <- rnas_caso_fnl_controlado(n = 30)

  tab <- rnas_consolidar_casos(list(c1, c2, c3))

  expect_true(is.data.frame(tab))
  expect_equal(nrow(tab), 3)
  expect_true(all(c("caso", "n_obs", "n_features", "loss_final") %in% names(tab)))
})


test_that("rnas_consolidar_casos rechaza lista vacia", {
  expect_error(rnas_consolidar_casos(list()))
})


test_that("rnas_consolidar_casos rechaza objeto sin resumen", {
  expect_error(rnas_consolidar_casos(list(list(a = 1))))
})


test_that("rnas_ejecutar_casos_aplicados retorna objeto integral", {
  res <- rnas_ejecutar_casos_aplicados(
    n_std2 = 30,
    n_simuds = 20,
    n_fnl = 30,
    horizonte = 1,
    eta0 = 0.05,
    T = 10,
    seed = 123
  )

  expect_s3_class(res, "rnas_casos_aplicados")
  expect_true(is.data.frame(res$resumen_casos))
  expect_equal(nrow(res$resumen_casos), 3)
  expect_equal(res$diagnostico$n_casos, 3)
})


test_that("prints de casos aplicados producen salida", {
  c1 <- rnas_caso_std2_controlado(n = 30, T = 5)
  c2 <- rnas_caso_simuds_controlado(n = 20, T = 5)
  c3 <- rnas_caso_fnl_controlado(n = 30)

  expect_output(print(c1), "Caso aplicado STD2-RNAS")
  expect_output(print(c2), "Caso aplicado SimuDS-RNAS")
  expect_output(print(c3), "Caso aplicado FNL-RNAS")

  res <- rnas_ejecutar_casos_aplicados(
    n_std2 = 30,
    n_simuds = 20,
    n_fnl = 30,
    T = 5
  )

  expect_output(print(res), "Casos aplicados DS Neural RNAS")
})


test_that("casos aplicados conservan configuracion", {
  res <- rnas_ejecutar_casos_aplicados(
    n_std2 = 30,
    n_simuds = 20,
    n_fnl = 30,
    horizonte = 1,
    eta0 = 0.05,
    T = 10,
    activation = "tanh",
    seed = 123
  )

  expect_equal(res$configuracion$horizonte, 1)
  expect_equal(res$configuracion$eta0, 0.05)
  expect_equal(res$configuracion$activation, "tanh")
})
