# ============================================================
# Ejemplo 10: Control optimo y meta-aprendizaje geometrico
# Capitulo 11: Control optimo y meta-aprendizaje geometrico
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

# 1. Gradiente inicial para evaluar costos locales
grad0 <- rnas_grad_neuron(
  X = X,
  y = y,
  w = w0,
  b = b0,
  activation = activation
)

grad0_vec <- c(grad0$grad_w, grad0$grad_b)

eta_grid <- seq(eta_min, eta_max, length.out = 10)

opt0 <- rnas_eta_opt_local(
  X = X,
  y = y,
  theta = theta0,
  grad = grad0_vec,
  eta_grid = eta_grid,
  activation = activation,
  alpha = 0,
  beta = 0
)

tabla_costos_inicial <- opt0$costos

print(tabla_costos_inicial)

write.csv(
  tabla_costos_inicial,
  file = "outputs/tables/cap11_costos_locales_iniciales.csv",
  row.names = FALSE
)

tabla_opt_inicial <- data.frame(
  n_tasas = length(eta_grid),
  eta_opt_inicial = opt0$eta_opt,
  costo_min_inicial = opt0$costo_min,
  loss_plus_opt = opt0$loss_plus
)

print(tabla_opt_inicial)

write.csv(
  tabla_opt_inicial,
  file = "outputs/tables/cap11_opt_local_inicial.csv",
  row.names = FALSE
)

# 2. Politicas de meta-control geometrico
politicas_meta <- list(
  opt_local = list(
    metodo = "opt_local",
    eta_grid = eta_grid,
    alpha = 0,
    beta = 0
  ),
  opt_local_penalizada = list(
    metodo = "opt_local",
    eta_grid = eta_grid,
    alpha = 0.01,
    beta = 1
  ),
  curvatura = list(
    metodo = "curvatura",
    alpha = 1
  ),
  lambda_max = list(
    metodo = "lambda_max",
    alpha = 1
  )
)

comp_meta <- rnas_comparar_meta_politicas(
  X = X,
  y = y,
  theta0 = theta0,
  politicas = politicas_meta,
  eta0 = eta0,
  eta_min = eta_min,
  eta_max = eta_max,
  T = T,
  activation = activation
)

print(comp_meta)

tabla_comparacion_meta <- comp_meta$comparacion

print(tabla_comparacion_meta)

write.csv(
  tabla_comparacion_meta,
  file = "outputs/tables/cap11_comparacion_meta_politicas.csv",
  row.names = FALSE
)

# 3. Trayectorias principales
tray_opt <- comp_meta$modelos$opt_local$trayectoria
tray_pen <- comp_meta$modelos$opt_local_penalizada$trayectoria
tray_curv <- comp_meta$modelos$curvatura$trayectoria
tray_lam <- comp_meta$modelos$lambda_max$trayectoria

iter_representativas <- c(0, 1, 2, 5, 10, 25, 50, 75, 100)

tabla_opt_parcial <- tray_opt[
  tray_opt$iter %in% iter_representativas,
  c("iter", "loss", "grad_norm", "eta", "speed", "kappa", "lambda_max", "costo_local")
]

print(tabla_opt_parcial)

write.csv(
  tabla_opt_parcial,
  file = "outputs/tables/cap11_trayectoria_opt_local_parcial.csv",
  row.names = FALSE
)

tabla_geo_parcial <- tray_curv[
  tray_curv$iter %in% iter_representativas,
  c("iter", "loss", "grad_norm", "eta", "speed", "kappa", "lambda_max")
]

print(tabla_geo_parcial)

write.csv(
  tabla_geo_parcial,
  file = "outputs/tables/cap11_trayectoria_curvatura_parcial.csv",
  row.names = FALSE
)

# 4. Guardar trayectorias completas
write.csv(
  tray_opt,
  file = "outputs/tables/cap11_trayectoria_opt_local.csv",
  row.names = FALSE
)

write.csv(
  tray_curv,
  file = "outputs/tables/cap11_trayectoria_curvatura.csv",
  row.names = FALSE
)

write.csv(
  tray_lam,
  file = "outputs/tables/cap11_trayectoria_lambda_max.csv",
  row.names = FALSE
)

# 5. Guardar objeto integral
res_cap11 <- list(
  datos = list(
    X = X,
    y = y,
    w0 = w0,
    b0 = b0,
    theta0 = theta0,
    eta0 = eta0,
    eta_min = eta_min,
    eta_max = eta_max,
    eta_grid = eta_grid,
    T = T,
    activation = activation
  ),
  costos_iniciales = tabla_costos_inicial,
  opt_local_inicial = tabla_opt_inicial,
  comparacion = tabla_comparacion_meta,
  modelos = comp_meta$modelos,
  trayectoria_opt_local_parcial = tabla_opt_parcial,
  trayectoria_curvatura_parcial = tabla_geo_parcial
)

saveRDS(
  res_cap11,
  file = "outputs/results/cap11_resultados_control_optimo.rds"
)

# 6. Figura: perdida por politica meta-geometrica
pdf("outputs/figures/cap11_perdida_meta_politicas.pdf", width = 7, height = 5)

plot(
  tray_opt$iter,
  tray_opt$loss,
  type = "l",
  lwd = 2,
  xlab = "Iteracion",
  ylab = "Perdida MSE",
  main = "Perdida por politica de meta-control"
)

lines(tray_pen$iter, tray_pen$loss, lwd = 2, lty = 2)
lines(tray_curv$iter, tray_curv$loss, lwd = 2, lty = 3)
lines(tray_lam$iter, tray_lam$loss, lwd = 2, lty = 4)

legend(
  "topright",
  legend = c("opt_local", "opt_penalizada", "curvatura", "lambda_max"),
  lty = 1:4,
  lwd = 2,
  bty = "n"
)

grid()

dev.off()

# 7. Figura: eta por politica meta-geometrica
pdf("outputs/figures/cap11_eta_meta_politicas.pdf", width = 7, height = 5)

plot(
  tray_opt$iter,
  tray_opt$eta,
  type = "l",
  lwd = 2,
  ylim = range(c(tray_opt$eta, tray_pen$eta, tray_curv$eta, tray_lam$eta)),
  xlab = "Iteracion",
  ylab = "Eta",
  main = "Evolucion de eta por politica meta-geometrica"
)

lines(tray_pen$iter, tray_pen$eta, lwd = 2, lty = 2)
lines(tray_curv$iter, tray_curv$eta, lwd = 2, lty = 3)
lines(tray_lam$iter, tray_lam$eta, lwd = 2, lty = 4)

legend(
  "topright",
  legend = c("opt_local", "opt_penalizada", "curvatura", "lambda_max"),
  lty = 1:4,
  lwd = 2,
  bty = "n"
)

grid()

dev.off()
