# ============================================================
# Ejemplo 11: Integracion con STD2, SimuDS y FNL
# Capitulo 12: Integracion de DS Neural RNAS
# ============================================================

library(DSNeuralRNAS)

# Crear carpetas de salida si no existen
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/results", recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Integracion STD2-RNAS
# ============================================================

set.seed(123)

n <- 60
tiempo <- seq_len(n)

datos_std2 <- data.frame(
  tiempo = tiempo,
  y = tanh(seq(-1, 1, length.out = n)) + rnorm(n, sd = 0.02),
  C = tanh(seq(-1, 1, length.out = n)),
  x = sin(seq(0, 4, length.out = n)) / 5,
  e = rnorm(n, sd = 0.03),
  gap = rnorm(n, sd = 0.02),
  speed = abs(c(NA, diff(tanh(seq(-1, 1, length.out = n)))))
)

datos_std2$speed[1] <- datos_std2$speed[2]

prep_std2 <- rnas_preparar_features_std2(
  datos_std2 = datos_std2,
  target = "y",
  features = c("C", "x", "e", "gap", "speed"),
  horizonte = 1
)

print(prep_std2)

res_std2 <- rnas_integrar_std2(
  datos_std2 = datos_std2,
  target = "y",
  features = c("C", "x", "e", "gap", "speed"),
  horizonte = 1,
  eta0 = 0.05,
  T = 100,
  activation = "tanh"
)

print(res_std2)

tabla_resumen_std2 <- rnas_resumen_integracion(res_std2)

print(tabla_resumen_std2)

write.csv(
  prep_std2$datos_alineados,
  file = "outputs/tables/cap12_std2_datos_alineados.csv",
  row.names = FALSE
)

write.csv(
  tabla_resumen_std2,
  file = "outputs/tables/cap12_resumen_std2_rnas.csv",
  row.names = FALSE
)

# ============================================================
# 2. Integracion SimuDS-RNAS
# ============================================================

trayectorias_simuds <- do.call(
  rbind,
  lapply(c("base", "shock", "recuperacion"), function(esc) {
    t <- 1:40

    s <- switch(
      esc,
      base = tanh(seq(-1, 1, length.out = 40)),
      shock = tanh(seq(-1, 1, length.out = 40)) + ifelse(t > 20, -0.15, 0),
      recuperacion = tanh(seq(-1, 1, length.out = 40)) + ifelse(t > 20, 0.10, 0)
    )

    data.frame(
      escenario = esc,
      t = t,
      s = s,
      stringsAsFactors = FALSE
    )
  })
)

prep_simuds <- rnas_preparar_trayectorias_simuds(
  trayectorias_simuds = trayectorias_simuds,
  estado_cols = "s",
  escenario_col = "escenario",
  tiempo_col = "t",
  horizonte = 1
)

res_simuds <- rnas_integrar_simuds(
  trayectorias_simuds = trayectorias_simuds,
  estado_cols = "s",
  target_estado = "s",
  escenario_col = "escenario",
  tiempo_col = "t",
  horizonte = 1,
  eta0 = 0.05,
  T = 100,
  activation = "tanh"
)

print(res_simuds)

tabla_resumen_simuds <- rnas_resumen_integracion(res_simuds)

print(tabla_resumen_simuds)

write.csv(
  prep_simuds$pares,
  file = "outputs/tables/cap12_simuds_pares_transicion.csv",
  row.names = FALSE
)

write.csv(
  tabla_resumen_simuds,
  file = "outputs/tables/cap12_resumen_simuds_rnas.csv",
  row.names = FALSE
)

# ============================================================
# 3. Integracion FNL-RNAS
# ============================================================

X_fnl <- prep_std2$X_rnas[, 1:2, drop = FALSE]
y_fnl <- prep_std2$y_rnas

restricciones <- list(
  norma_theta = function(theta) sum(theta^2) - 10,
  cota_w1 = function(theta) theta[1] - 3
)

fnl <- rnas_formular_fnl(
  X = X_fnl,
  y = y_fnl,
  activation = "tanh",
  restricciones = restricciones
)

theta_eval <- c(0.1, 0.1, 0.0)

valor_objetivo <- fnl$objetivo(theta_eval)
valores_restricciones <- fnl$evaluar_restricciones(theta_eval)
factible <- fnl$es_factible(theta_eval)

tabla_fnl <- data.frame(
  indicador = c(
    "valor_objetivo",
    names(valores_restricciones),
    "factible"
  ),
  valor = c(
    valor_objetivo,
    valores_restricciones,
    as.numeric(factible)
  )
)

print(fnl)
print(tabla_fnl)

write.csv(
  tabla_fnl,
  file = "outputs/tables/cap12_diagnostico_fnl_rnas.csv",
  row.names = FALSE
)

tabla_resumen_fnl <- rnas_resumen_integracion(fnl)

write.csv(
  tabla_resumen_fnl,
  file = "outputs/tables/cap12_resumen_fnl_rnas.csv",
  row.names = FALSE
)

# ============================================================
# 4. Resumen integral
# ============================================================

tabla_resumen_integral <- rbind(
  tabla_resumen_std2,
  tabla_resumen_simuds,
  tabla_resumen_fnl[, names(tabla_resumen_std2)]
)

print(tabla_resumen_integral)

write.csv(
  tabla_resumen_integral,
  file = "outputs/tables/cap12_resumen_integral.csv",
  row.names = FALSE
)

res_cap12 <- list(
  datos_std2 = datos_std2,
  prep_std2 = prep_std2,
  integracion_std2 = res_std2,
  trayectorias_simuds = trayectorias_simuds,
  prep_simuds = prep_simuds,
  integracion_simuds = res_simuds,
  formulacion_fnl = fnl,
  diagnostico_fnl = tabla_fnl,
  resumen_integral = tabla_resumen_integral
)

saveRDS(
  res_cap12,
  file = "outputs/results/cap12_resultados_integracion.rds"
)

# ============================================================
# 5. Figuras simples
# ============================================================

pdf("outputs/figures/cap12_std2_serie_senales.pdf", width = 7, height = 5)

plot(
  datos_std2$tiempo,
  datos_std2$y,
  type = "l",
  lwd = 2,
  xlab = "Tiempo",
  ylab = "Valor",
  main = "Serie y señal estructural STD2"
)

lines(datos_std2$tiempo, datos_std2$C, lwd = 2, lty = 2)

legend(
  "topleft",
  legend = c("y", "C"),
  lty = c(1, 2),
  lwd = 2,
  bty = "n"
)

grid()

dev.off()

pdf("outputs/figures/cap12_simuds_trayectorias.pdf", width = 7, height = 5)

plot(
  NULL,
  xlim = range(trayectorias_simuds$t),
  ylim = range(trayectorias_simuds$s),
  xlab = "Tiempo",
  ylab = "Estado s",
  main = "Trayectorias simuladas SimuDS"
)

for (esc in unique(trayectorias_simuds$escenario)) {
  sub <- trayectorias_simuds[trayectorias_simuds$escenario == esc, ]
  lines(sub$t, sub$s, lwd = 2)
}

legend(
  "topleft",
  legend = unique(trayectorias_simuds$escenario),
  lty = 1,
  lwd = 2,
  bty = "n"
)

grid()

dev.off()
