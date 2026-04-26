# ============================================================
# Helpers generales DSNeuralRNASLab
# ============================================================

generar_datos_neurona_base <- function() {
  X <- matrix(
    c(0.5, -0.2,
      1.0,  0.3,
      -0.7,  0.8,
      0.0,  0.0),
    ncol = 2,
    byrow = TRUE
  )

  y <- c(0.4, 0.8, -0.2, 0.1)

  list(X = X, y = y)
}

generar_datos_std2_app <- function(n = 60, seed = 123) {
  set.seed(seed)

  tiempo <- seq_len(n)
  base <- tanh(seq(-1, 1, length.out = n))

  datos <- data.frame(
    tiempo = tiempo,
    y = base + stats::rnorm(n, sd = 0.02),
    C = base,
    x = sin(seq(0, 4, length.out = n)) / 5,
    e = stats::rnorm(n, sd = 0.03),
    gap = stats::rnorm(n, sd = 0.02),
    speed = abs(c(NA_real_, diff(base))),
    stringsAsFactors = FALSE
  )

  datos$speed[1] <- datos$speed[2]
  datos
}

generar_trayectorias_simuds_app <- function(n = 40, escenario_sel = "todos") {
  escenarios <- c("base", "shock", "recuperacion")

  tray <- do.call(
    rbind,
    lapply(escenarios, function(esc) {
      t <- seq_len(n)
      base <- tanh(seq(-1, 1, length.out = n))

      s <- switch(
        esc,
        base = base,
        shock = base + ifelse(t > floor(n / 2), -0.15, 0),
        recuperacion = base + ifelse(t > floor(n / 2), 0.10, 0)
      )

      data.frame(
        escenario = esc,
        t = t,
        s = s,
        stringsAsFactors = FALSE
      )
    })
  )

  if (!identical(escenario_sel, "todos")) {
    tray <- tray[tray$escenario == escenario_sel, , drop = FALSE]
  }

  tray
}

mostrar_codigo <- function(codigo) {
  tags$pre(class = "codigo-r", codigo)
}

tabla_dt <- function(data) {
  DT::datatable(
    data,
    options = list(
      pageLength = 10,
      scrollX = TRUE,
      dom = "tip"
    ),
    rownames = FALSE
  )
}
