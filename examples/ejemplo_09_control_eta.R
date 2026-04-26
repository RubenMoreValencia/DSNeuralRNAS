# ============================================================
# Ejemplo 09: Control de tasa de aprendizaje
# Capitulo 10: Politicas de control de eta
# ============================================================

library(DSNeuralRNAS)

# Crear carpetas de salida si no existen
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/results", recursive = TRUE, showWarnings = FALSE)

# Datos controlados de ejemplo
X <- matrix(
  c(0.5, -0.2,
    1.0,  0.3,
    -0.7,  0.8,
    0.0,  0.0),
  ncol = 2,
  byrow = TRUE
)

y <- c(0.4, 0.8, -0.2, 0.1)

w0 <- c(0.8, 0.3)
b0 <- 0.1
theta0 <- rnas_pack_neuron_params(w = w0, b = b0)

eta0 <- 0.1
eta_min <- 0.01
eta_max <- 0.1
T <- 100
activation <- "tanh"

# 1. Politicas a comparar
politicas <- list(
  constante = "constante",
  temporal = list(
    tipo = "temporal",
    alpha = 0.01
  ),
  mejora = list(
    tipo = "mejora",
    tau_loss = 0.01,
    gamma = 0.95
  ),
  regimen = list(
    tipo = "regimen",
    gamma_refinamiento = 0.98,
    gamma_saturacion = 0.90,
    gamma_inestabilidad = 0.50,
    gamma_estabilizacion = 0.95
  ),
  inestabilidad = list(
    tipo = "inestabilidad",
    gamma = 0.50
  )
)

# 2. Comparacion de politicas
comp <- rnas_comparar_politicas_eta(
  X = X,
  y = y,
  theta0 = theta0,
  politicas = politicas,
  eta0 = eta0,
  eta_min = eta_min,
  eta_max = eta_max,
  T = T,
  activation = activation
)

print(comp)

tabla_comparacion <- comp$comparacion

print(tabla_comparacion)

write.csv(
  tabla_comparacion,
  file = "outputs/tables/cap10_comparacion_politicas_eta.csv",
  row.names = FALSE
)

# 3. Extraer trayectorias principales
tray_constante <- comp$modelos$constante$trayectoria
tray_temporal <- comp$modelos$temporal$trayectoria
tray_mejora <- comp$modelos$mejora$trayectoria
tray_regimen <- comp$modelos$regimen$trayectoria
tray_inestabilidad <- comp$modelos$inestabilidad$trayectoria

# 4. Tabla parcial de trayectoria para politica por regimen
iter_representativas <- c(0, 1, 2, 5, 10, 25, 50, 75, 100)

tabla_regimen_parcial <- tray_regimen[
  tray_regimen$iter %in% iter_representativas,
  c("iter", "loss", "grad_norm", "eta", "speed", "regimen", "accion_eta")
]

print(tabla_regimen_parcial)

write.csv(
  tabla_regimen_parcial,
  file = "outputs/tables/cap10_trayectoria_regimen_eta_parcial.csv",
  row.names = FALSE
)

write.csv(
  tray_regimen,
  file = "outputs/tables/cap10_trayectoria_regimen_eta.csv",
  row.names = FALSE
)

# 5. Resumen de acciones de control por politica de regimen
tabla_acciones_regimen <- as.data.frame(table(tray_regimen$accion_eta))
names(tabla_acciones_regimen) <- c("accion_eta", "frecuencia")

print(tabla_acciones_regimen)

write.csv(
  tabla_acciones_regimen,
  file = "outputs/tables/cap10_acciones_regimen_eta.csv",
  row.names = FALSE
)

# 6. Guardar objeto integral
res_cap10 <- list(
  datos = list(
    X = X,
    y = y,
    w0 = w0,
    b0 = b0,
    theta0 = theta0,
    eta0 = eta0,
    eta_min = eta_min,
    eta_max = eta_max,
    T = T,
    activation = activation
  ),
  politicas = politicas,
  comparacion = tabla_comparacion,
  modelos = comp$modelos,
  trayectoria_regimen_parcial = tabla_regimen_parcial,
  acciones_regimen = tabla_acciones_regimen
)

saveRDS(
  res_cap10,
  file = "outputs/results/cap10_resultados_control_eta.rds"
)

# 7. Figura: perdida por politica
pdf("outputs/figures/cap10_perdida_politicas_eta.pdf", width = 7, height = 5)

plot(
  tray_constante$iter,
  tray_constante$loss,
  type = "l",
  lwd = 2,
  xlab = "Iteracion",
  ylab = "Perdida MSE",
  main = "Perdida por politica de tasa"
)

lines(tray_temporal$iter, tray_temporal$loss, lwd = 2, lty = 2)
lines(tray_mejora$iter, tray_mejora$loss, lwd = 2, lty = 3)
lines(tray_regimen$iter, tray_regimen$loss, lwd = 2, lty = 4)
lines(tray_inestabilidad$iter, tray_inestabilidad$loss, lwd = 2, lty = 5)

legend(
  "topright",
  legend = c("constante", "temporal", "mejora", "regimen", "inestabilidad"),
  lty = 1:5,
  lwd = 2,
  bty = "n"
)

grid()

dev.off()

# 8. Figura: eta por politica
pdf("outputs/figures/cap10_eta_politicas.pdf", width = 7, height = 5)

plot(
  tray_constante$iter,
  tray_constante$eta,
  type = "l",
  lwd = 2,
  ylim = range(c(
    tray_constante$eta,
    tray_temporal$eta,
    tray_mejora$eta,
    tray_regimen$eta,
    tray_inestabilidad$eta
  )),
  xlab = "Iteracion",
  ylab = "Eta",
  main = "Evolucion de eta por politica"
)

lines(tray_temporal$iter, tray_temporal$eta, lwd = 2, lty = 2)
lines(tray_mejora$iter, tray_mejora$eta, lwd = 2, lty = 3)
lines(tray_regimen$iter, tray_regimen$eta, lwd = 2, lty = 4)
lines(tray_inestabilidad$iter, tray_inestabilidad$eta, lwd = 2, lty = 5)

legend(
  "topright",
  legend = c("constante", "temporal", "mejora", "regimen", "inestabilidad"),
  lty = 1:5,
  lwd = 2,
  bty = "n"
)

grid()

dev.off()
