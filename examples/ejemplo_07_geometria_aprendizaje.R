# ============================================================
# Ejemplo 07: Paisaje de perdida y geometria del aprendizaje
# Capitulo 8: Geometria del aprendizaje
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

# 1. Construccion de malla de perdida
w1_seq <- seq(0.70, 0.90, length.out = 41)
w2_seq <- seq(0.20, 0.45, length.out = 41)

grid_loss <- rnas_loss_grid_neuron(
  X = X,
  y = y,
  w1_seq = w1_seq,
  w2_seq = w2_seq,
  b = b0,
  activation = activation
)

tabla_grid_resumen <- data.frame(
  n_w1 = length(w1_seq),
  n_w2 = length(w2_seq),
  n_filas = nrow(grid_loss),
  loss_min = min(grid_loss$loss),
  loss_max = max(grid_loss$loss),
  w1_min_loss = grid_loss$w1[which.min(grid_loss$loss)],
  w2_min_loss = grid_loss$w2[which.min(grid_loss$loss)]
)

print(tabla_grid_resumen)

write.csv(
  tabla_grid_resumen,
  file = "outputs/tables/cap08_resumen_malla_perdida.csv",
  row.names = FALSE
)

write.csv(
  grid_loss,
  file = "outputs/tables/cap08_malla_perdida.csv",
  row.names = FALSE
)

# 2. Dinamica base para obtener trayectoria
res_dyn <- rnas_integrar_dinamica_neuron(
  X = X,
  y = y,
  theta0 = theta0,
  eta = eta,
  dt = dt,
  T = T,
  activation = activation
)

# 3. Resumen geometrico en estado inicial
geo_ini <- rnas_resumen_geometria_neuron(
  X = X,
  y = y,
  theta = theta0,
  activation = activation,
  h = 1e-4
)

theta_final <- res_dyn$theta_final

# 4. Resumen geometrico en estado final
geo_fin <- rnas_resumen_geometria_neuron(
  X = X,
  y = y,
  theta = theta_final,
  activation = activation,
  h = 1e-4
)

tabla_geo_local <- data.frame(
  estado = c("inicial", "final"),
  loss = c(geo_ini$loss, geo_fin$loss),
  grad_norm = c(geo_ini$grad_norm, geo_fin$grad_norm),
  lambda_min = c(geo_ini$lambda_min, geo_fin$lambda_min),
  lambda_max = c(geo_ini$lambda_max, geo_fin$lambda_max),
  n_pos = c(geo_ini$n_pos, geo_fin$n_pos),
  n_neg = c(geo_ini$n_neg, geo_fin$n_neg),
  n_zero = c(geo_ini$n_zero, geo_fin$n_zero),
  clasificacion = c(geo_ini$clasificacion, geo_fin$clasificacion),
  curvatura = c(geo_ini$curvatura, geo_fin$curvatura)
)

print(tabla_geo_local)

write.csv(
  tabla_geo_local,
  file = "outputs/tables/cap08_resumen_geometria_local.csv",
  row.names = FALSE
)

# 5. Hessiano inicial y final
write.csv(
  geo_ini$H,
  file = "outputs/tables/cap08_hessiano_inicial.csv",
  row.names = TRUE
)

write.csv(
  geo_fin$H,
  file = "outputs/tables/cap08_hessiano_final.csv",
  row.names = TRUE
)

tabla_autovalores <- data.frame(
  indice = seq_along(geo_ini$eigenvalues),
  lambda_inicial = geo_ini$eigenvalues,
  lambda_final = geo_fin$eigenvalues
)

print(tabla_autovalores)

write.csv(
  tabla_autovalores,
  file = "outputs/tables/cap08_autovalores.csv",
  row.names = FALSE
)

# 6. Geometria sobre trayectoria
iter_representativas <- c(0, 1, 2, 5, 10, 25, 50, 75, 100)

geo_tray <- rnas_geometria_trayectoria_neuron(
  object = res_dyn,
  X = X,
  y = y,
  iteraciones = iter_representativas,
  h = 1e-4
)

print(geo_tray)

write.csv(
  geo_tray,
  file = "outputs/tables/cap08_geometria_trayectoria.csv",
  row.names = FALSE
)

# 7. Guardar objeto integral
res_cap08 <- list(
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
  malla = grid_loss,
  resumen_malla = tabla_grid_resumen,
  dinamica = res_dyn,
  geometria_inicial = geo_ini,
  geometria_final = geo_fin,
  resumen_geometria_local = tabla_geo_local,
  autovalores = tabla_autovalores,
  geometria_trayectoria = geo_tray
)

saveRDS(
  res_cap08,
  file = "outputs/results/cap08_resultados_geometria_aprendizaje.rds"
)

# 8. Figura: contorno del paisaje de perdida
pdf("outputs/figures/cap08_contorno_paisaje_perdida.pdf", width = 7, height = 5)

z_mat <- matrix(
  grid_loss$loss,
  nrow = length(w1_seq),
  ncol = length(w2_seq)
)

contour(
  x = w1_seq,
  y = w2_seq,
  z = z_mat,
  xlab = "w1",
  ylab = "w2",
  main = "Paisaje de perdida - Neurona RNAS"
)

points(w0[1], w0[2], pch = 19)
points(theta_final[1], theta_final[2], pch = 17)

legend(
  "topright",
  legend = c("Estado inicial", "Estado final"),
  pch = c(19, 17),
  bty = "n"
)

dev.off()

# 9. Figura: perdida y autovalores sobre trayectoria
pdf("outputs/figures/cap08_lambda_trayectoria.pdf", width = 7, height = 5)

plot(
  geo_tray$iter,
  geo_tray$lambda_max,
  type = "l",
  lwd = 2,
  xlab = "Iteracion",
  ylab = "Autovalor",
  main = "Autovalores extremos del Hessiano"
)

lines(
  geo_tray$iter,
  geo_tray$lambda_min,
  lwd = 2,
  lty = 2
)

legend(
  "topright",
  legend = c("lambda_max", "lambda_min"),
  lty = c(1, 2),
  lwd = 2,
  bty = "n"
)

grid()

dev.off()
