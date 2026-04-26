test_that("rnas_sigmoid devuelve valores entre 0 y 1", {
  z <- seq(-10, 10, length.out = 100)
  s <- rnas_sigmoid(z)

  expect_true(all(s > 0))
  expect_true(all(s < 1))
  expect_equal(rnas_sigmoid(0), 0.5)
})


test_that("rnas_dsigmoid devuelve derivadas no negativas", {
  z <- seq(-10, 10, length.out = 100)
  ds <- rnas_dsigmoid(z)

  expect_true(all(ds >= 0))
  expect_equal(rnas_dsigmoid(0), 0.25)
})


test_that("rnas_tanh devuelve valores entre -1 y 1", {
  z <- seq(-10, 10, length.out = 100)
  t <- rnas_tanh(z)

  expect_true(all(t > -1))
  expect_true(all(t < 1))
  expect_equal(rnas_tanh(0), 0)
})


test_that("rnas_dtanh devuelve derivadas no negativas", {
  z <- seq(-10, 10, length.out = 100)
  dt <- rnas_dtanh(z)

  expect_true(all(dt >= 0))
  expect_equal(rnas_dtanh(0), 1)
})


test_that("rnas_get_activation recupera activacion y derivada", {
  act <- rnas_get_activation("tanh")

  expect_true(is.list(act))
  expect_true(is.function(act$activation))
  expect_true(is.function(act$derivative))
  expect_equal(act$name, "tanh")

  expect_equal(act$activation(0), 0)
  expect_equal(act$derivative(0), 1)
})


test_that("activaciones rechazan entradas no numericas", {
  expect_error(rnas_sigmoid("a"))
  expect_error(rnas_dsigmoid("a"))
  expect_error(rnas_tanh("a"))
  expect_error(rnas_dtanh("a"))
})
