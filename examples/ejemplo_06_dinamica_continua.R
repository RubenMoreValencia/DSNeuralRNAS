# ============================================================
# Ejemplo 06: Dinamica continua aproximada DS Neural RNAS
# Capitulo 7: Aprendizaje como dinamica continua
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

# 1. Validacion de empaquetado y desempaquetado
pars0 <- rnas_unpack_neuron_params(theta0, d_input = ncol(X))

tabla_estado_inicial <- data.frame(
  componente = c("w1", "w2", "b"),
  valor = c(pars0$w, pars0$b)
)

print(tabla_estado_inicial)

write.csv(
  tabla_estado_inicial,
  file = "outputs/tables/cap07_estado_inicial.csv",
  row.names = FALSE
)

# 2. Campo de gradiente inicial
campo0 <- rnas_campo_gradiente_neuron(
  theta = theta0,
  X = X,
  y = y,
  eta = eta,
  t = 0,
  activation = activation
)

tabla_campo <- data.frame(
  componente = c("theta1", "theta2", "theta3"),
  theta = theta0,
  grad = campo0$grad,
  theta_dot = campo0$theta_dot
)

print(tabla_campo)

write.csv(
  tabla_campo,
  file = "outputs/tables/cap07_campo_gradiente_inicial.csv",
  row.names = FALSE
)

tabla_campo_resumen <- data.frame(
  indicador = c(
    "longitud_theta",
    "longitud_theta_dot",
    "loss_inicial",
    "grad_norm_inicial",
    "speed_inicial",
    "eta"
  ),
  valor = c(
    length(theta0),
    length(campo0$theta_dot),
    campo0$loss,
    campo0$grad_norm,
    campo0$speed,
    campo0$eta
  )
)

print(tabla_campo_resumen)

write.csv(
  tabla_campo_resumen,
  file = "outputs/tables/cap07_campo_resumen.csv",
  row.names = FALSE
)

# 3. Integracion dinamica mediante Euler
res_dyn <- rnas_integrar_dinamica_neuron(
  X = X,
  y = y,
  theta0 = theta0,
  eta = eta,
  dt = dt,
  T = T,
  activation = activation
)

print(res_dyn)

# 4. Resumen de dinamica
tabla_resumen <- rnas_resumen_dinamica_neuron(res_dyn)

print(tabla_resumen)

write.csv(
  tabla_resumen,
  file = "outputs/tables/cap07_resumen_dinamica.csv",
  row.names = FALSE
)

# 5. Trayectoria completa
tabla_trayectoria <- res_dyn$trayectoria

write.csv(
  tabla_trayectoria,
  file = "outputs/tables/cap07_trayectoria_dinamica.csv",
  row.names = FALSE
)

# 6. Trayectoria parcial para el libro
iter_representativas <- unique(c(
  0,
  1,
  2,
  5,
  10,
  25,
  50,
  75,
  100
))

tabla_trayectoria_libro <- tabla_trayectoria[
  tabla_trayectoria$iter %in% iter_representativas,
  c("iter", "time", "loss", "grad_norm", "speed", "eta", "b", "w1", "w2")
]

print(tabla_trayectoria_libro)

write.csv(
  tabla_trayectoria_libro,
  file = "outputs/tables/cap07_trayectoria_libro.csv",
  row.names = FALSE
)

# 7. Predicciones finales desde dinamica
pred_final <- rnas_predict_dinamica_neuron(res_dyn, X)

tabla_pred_final <- data.frame(
  obs = seq_len(nrow(X)),
  y_obs = y,
  y_hat_final = pred_final,
  error_final = pred_final - y,
  error_cuadrado_final = (pred_final - y)^2
)

print(tabla_pred_final)

write.csv(
  tabla_pred_final,
  file = "outputs/tables/cap07_predicciones_finales.csv",
  row.names = FALSE
)

# 8. Guardar objeto integral
res_cap07 <- list(
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
  estado_inicial = tabla_estado_inicial,
  campo_inicial = campo0,
  campo_tabla = tabla_campo,
  campo_resumen = tabla_campo_resumen,
  dinamica = res_dyn,
  resumen = tabla_resumen,
  trayectoria_libro = tabla_trayectoria_libro,
  predicciones_finales = tabla_pred_final
)

saveRDS(
  res_cap07,
  file = "outputs/results/cap07_resultados_dinamica_continua.rds"
)

# 9. Figura de perdida
pdf("outputs/figures/cap07_curva_perdida_dinamica.pdf", width = 7, height = 5)

plot(
  tabla_trayectoria$time,
  tabla_trayectoria$loss,
  type = "l",
  lwd = 2,
  xlab = "Tiempo",
  ylab = "Perdida MSE",
  main = "Evolucion de la perdida - Dinamica continua RNAS"
)

points(
  tabla_trayectoria_libro$time,
  tabla_trayectoria_libro$loss,
  pch = 19
)

grid()

dev.off()

# 10. Figura de norma del gradiente
pdf("outputs/figures/cap07_norma_gradiente_dinamica.pdf", width = 7, height = 5)

plot(
  tabla_trayectoria$time,
  tabla_trayectoria$grad_norm,
  type = "l",
  lwd = 2,
  xlab = "Tiempo",
  ylab = "Norma del gradiente",
  main = "Evolucion de la norma del gradiente - Dinamica RNAS"
)

points(
  tabla_trayectoria_libro$time,
  tabla_trayectoria_libro$grad_norm,
  pch = 19
)

grid()

dev.off()

# 11. Figura de velocidad dinamica
pdf("outputs/figures/cap07_velocidad_dinamica.pdf", width = 7, height = 5)

plot(
  tabla_trayectoria$time,
  tabla_trayectoria$speed,
  type = "l",
  lwd = 2,
  xlab = "Tiempo",
  ylab = "Velocidad ||dtheta/dt||",
  main = "Velocidad de aprendizaje - Dinamica RNAS"
)

points(
  tabla_trayectoria_libro$time,
  tabla_trayectoria_libro$speed,
  pch = 19
)

grid()

dev.off()
