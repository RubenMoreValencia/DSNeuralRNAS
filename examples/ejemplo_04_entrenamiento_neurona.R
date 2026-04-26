# ============================================================
# Ejemplo 04: Entrenamiento discreto de neurona RNAS
# Capítulo 5: Entrenamiento discreto de una neurona
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
eta <- 0.1
T <- 100
activation <- "tanh"

# 1. Validación de una actualización inicial
grad0 <- rnas_grad_neuron(
  X = X,
  y = y,
  w = w0,
  b = b0,
  activation = activation
)

upd0 <- rnas_update_params_neuron(
  w = w0,
  b = b0,
  grad_w = grad0$grad_w,
  grad_b = grad0$grad_b,
  eta = eta
)

tabla_update <- data.frame(
  parametro = c("w1", "w2", "b"),
  valor_inicial = c(w0, b0),
  gradiente = c(grad0$grad_w, grad0$grad_b),
  valor_actualizado = c(upd0$w_new, upd0$b_new)
)

print(tabla_update)

write.csv(
  tabla_update,
  file = "outputs/tables/cap05_update_parametros.csv",
  row.names = FALSE
)

# 2. Entrenamiento completo
res_train <- rnas_train_neuron(
  X = X,
  y = y,
  w0 = w0,
  b0 = b0,
  eta = eta,
  T = T,
  activation = activation
)

print(res_train)

# 3. Resumen del entrenamiento
tabla_resumen <- rnas_resumen_entrenamiento_neuron(res_train)

print(tabla_resumen)

write.csv(
  tabla_resumen,
  file = "outputs/tables/cap05_resumen_entrenamiento.csv",
  row.names = FALSE
)

# 4. Trayectoria completa
tabla_trayectoria <- res_train$trayectoria

print(head(tabla_trayectoria, 10))
print(tail(tabla_trayectoria, 10))

write.csv(
  tabla_trayectoria,
  file = "outputs/tables/cap05_trayectoria_entrenamiento.csv",
  row.names = FALSE
)

# 5. Tabla parcial de trayectoria para el libro
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
  c("iter", "loss", "grad_norm", "b", "w1", "w2")
]

print(tabla_trayectoria_libro)

write.csv(
  tabla_trayectoria_libro,
  file = "outputs/tables/cap05_trayectoria_libro.csv",
  row.names = FALSE
)

# 6. Predicciones finales
pred_final <- rnas_predict_neuron(res_train, X)

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
  file = "outputs/tables/cap05_predicciones_finales.csv",
  row.names = FALSE
)

# 7. Guardar objeto integral
res_cap05 <- list(
  datos = list(
    X = X,
    y = y,
    w0 = w0,
    b0 = b0,
    eta = eta,
    T = T,
    activation = activation
  ),
  update_inicial = tabla_update,
  entrenamiento = res_train,
  resumen = tabla_resumen,
  trayectoria_libro = tabla_trayectoria_libro,
  predicciones_finales = tabla_pred_final
)

saveRDS(
  res_cap05,
  file = "outputs/results/cap05_resultados_entrenamiento_neurona.rds"
)

# 8. Figura de pérdida
pdf("outputs/figures/cap05_curva_perdida_neurona.pdf", width = 7, height = 5)

plot(
  tabla_trayectoria$iter,
  tabla_trayectoria$loss,
  type = "l",
  lwd = 2,
  xlab = "Iteración",
  ylab = "Pérdida MSE",
  main = "Evolución de la pérdida - Neurona RNAS"
)

points(
  tabla_trayectoria_libro$iter,
  tabla_trayectoria_libro$loss,
  pch = 19
)

grid()

dev.off()

# 9. Figura de norma del gradiente
pdf("outputs/figures/cap05_norma_gradiente_neurona.pdf", width = 7, height = 5)

plot(
  tabla_trayectoria$iter,
  tabla_trayectoria$grad_norm,
  type = "l",
  lwd = 2,
  xlab = "Iteración",
  ylab = "Norma del gradiente",
  main = "Evolución de la norma del gradiente - Neurona RNAS"
)

points(
  tabla_trayectoria_libro$iter,
  tabla_trayectoria_libro$grad_norm,
  pch = 19
)

grid()

dev.off()
