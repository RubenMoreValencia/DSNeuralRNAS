# ============================================================
# Ejemplo 08: Regimenes dinamicos del aprendizaje
# Capitulo 9: Regimenes dinamicos
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

eta <- 0.1
dt <- 1
T <- 100
activation <- "tanh"

# 1. Dinamica base
res_dyn <- rnas_integrar_dinamica_neuron(
  X = X,
  y = y,
  theta0 = theta0,
  eta = eta,
  dt = dt,
  T = T,
  activation = activation
)

# 2. Calculo de senales dinamicas
senales <- rnas_calcular_senales_regimen(
  trayectoria = res_dyn$trayectoria,
  loss_col = "loss",
  grad_col = "grad_norm",
  speed_col = "speed",
  ventana = 5
)

tabla_senales_parcial <- senales[
  senales$iter %in% c(0, 1, 2, 5, 10, 25, 50, 75, 100),
  c(
    "iter", "time", "loss", "grad_norm", "speed",
    "delta_loss", "reduccion_relativa",
    "delta_grad", "delta_speed",
    "loss_suav", "grad_suav", "speed_suav"
  )
]

print(tabla_senales_parcial)

write.csv(
  senales,
  file = "outputs/tables/cap09_senales_regimen.csv",
  row.names = FALSE
)

write.csv(
  tabla_senales_parcial,
  file = "outputs/tables/cap09_senales_parcial.csv",
  row.names = FALSE
)

# 3. Analisis completo de regimenes
res_reg <- rnas_analizar_regimenes_neuron(
  object = res_dyn,
  ventana = 5,
  tau_loss = 0.01,
  tau_grad = 1e-3,
  eps_loss = 1e-8,
  eps_grad = 1e-4,
  tau_speed = Inf,
  usar_suavizado = TRUE
)

print(res_reg)

# 4. Trayectoria con regimenes
tray_reg <- res_reg$trayectoria_regimen

tabla_regimen_parcial <- tray_reg[
  tray_reg$iter %in% c(0, 1, 2, 5, 10, 25, 50, 75, 100),
  c(
    "iter", "time", "loss", "grad_norm", "speed",
    "reduccion_relativa", "regimen"
  )
]

print(tabla_regimen_parcial)

write.csv(
  tray_reg,
  file = "outputs/tables/cap09_trayectoria_regimen.csv",
  row.names = FALSE
)

write.csv(
  tabla_regimen_parcial,
  file = "outputs/tables/cap09_trayectoria_regimen_parcial.csv",
  row.names = FALSE
)

# 5. Segmentos de regimenes
segmentos <- res_reg$segmentos

print(segmentos)

write.csv(
  segmentos,
  file = "outputs/tables/cap09_segmentos_regimen.csv",
  row.names = FALSE
)

# 6. Resumen de regimenes
resumen <- res_reg$resumen

print(resumen)

write.csv(
  resumen,
  file = "outputs/tables/cap09_resumen_regimenes.csv",
  row.names = FALSE
)

# 7. Tabla general de configuracion
tabla_config <- data.frame(
  indicador = c(
    "ventana",
    "tau_loss",
    "tau_grad",
    "eps_loss",
    "eps_grad",
    "usar_suavizado",
    "n_estados",
    "n_segmentos",
    "n_regimenes"
  ),
  valor = c(
    res_reg$configuracion$ventana,
    res_reg$configuracion$tau_loss,
    res_reg$configuracion$tau_grad,
    res_reg$configuracion$eps_loss,
    res_reg$configuracion$eps_grad,
    res_reg$configuracion$usar_suavizado,
    nrow(res_reg$trayectoria_regimen),
    nrow(res_reg$segmentos),
    length(unique(res_reg$trayectoria_regimen$regimen))
  )
)

print(tabla_config)

write.csv(
  tabla_config,
  file = "outputs/tables/cap09_configuracion_regimenes.csv",
  row.names = FALSE
)

# 8. Guardar objeto integral
res_cap09 <- list(
  datos = list(
    X = X,
    y = y,
    w0 = w0,
    b0 = b0,
    theta0 = theta0,
    eta = eta,
    dt = dt,
    T = T,
    activation = activation
  ),
  dinamica = res_dyn,
  senales = senales,
  senales_parcial = tabla_senales_parcial,
  analisis_regimenes = res_reg,
  trayectoria_regimen = tray_reg,
  trayectoria_regimen_parcial = tabla_regimen_parcial,
  segmentos = segmentos,
  resumen = resumen,
  configuracion = tabla_config
)

saveRDS(
  res_cap09,
  file = "outputs/results/cap09_resultados_regimenes.rds"
)

# 9. Figura: perdida con regimenes
pdf("outputs/figures/cap09_perdida_regimenes.pdf", width = 7, height = 5)

plot(
  tray_reg$iter,
  tray_reg$loss,
  type = "l",
  lwd = 2,
  xlab = "Iteracion",
  ylab = "Perdida MSE",
  main = "Perdida y regimenes dinamicos"
)

puntos <- tray_reg$iter %in% c(0, 1, 2, 5, 10, 25, 50, 75, 100)

points(
  tray_reg$iter[puntos],
  tray_reg$loss[puntos],
  pch = 19
)

text(
  tray_reg$iter[puntos],
  tray_reg$loss[puntos],
  labels = substr(tray_reg$regimen[puntos], 1, 3),
  pos = 3,
  cex = 0.7
)

grid()

dev.off()

# 10. Figura: frecuencia de regimenes
pdf("outputs/figures/cap09_frecuencia_regimenes.pdf", width = 7, height = 5)

barplot(
  height = resumen$frecuencia,
  names.arg = resumen$regimen,
  las = 2,
  ylab = "Frecuencia",
  main = "Frecuencia de regimenes dinamicos"
)

grid()

dev.off()
