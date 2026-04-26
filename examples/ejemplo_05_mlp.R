# ============================================================
# Ejemplo 05: Perceptron multicapa simple RNAS
# Capitulo 6: MLP simple
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

d_hidden <- 3
eta <- 0.1
T <- 200
activation <- "tanh"
seed <- 123

# 1. Inicializacion de parametros
params0 <- rnas_init_mlp(
  d_input = ncol(X),
  d_hidden = d_hidden,
  init_sd = 0.1,
  seed = seed
)

tabla_dimensiones <- data.frame(
  elemento = c("W", "b1", "v", "b2"),
  dimension = c(
    paste(dim(params0$W), collapse = " x "),
    as.character(length(params0$b1)),
    as.character(length(params0$v)),
    as.character(length(params0$b2))
  )
)

print(tabla_dimensiones)

write.csv(
  tabla_dimensiones,
  file = "outputs/tables/cap06_dimensiones_parametros.csv",
  row.names = FALSE
)

# 2. Forward inicial y perdida inicial
forward0 <- rnas_mlp_forward(
  X = X,
  params = params0,
  activation = activation,
  devolver_cache = TRUE
)

loss0 <- rnas_mlp_loss(
  X = X,
  y = y,
  params = params0,
  activation = activation,
  devolver_detalle = TRUE
)

tabla_forward_inicial <- data.frame(
  obs = seq_len(nrow(X)),
  y_obs = y,
  y_hat_inicial = forward0$y_hat,
  error_inicial = forward0$y_hat - y,
  error_cuadrado_inicial = (forward0$y_hat - y)^2
)

print(tabla_forward_inicial)

write.csv(
  tabla_forward_inicial,
  file = "outputs/tables/cap06_forward_inicial.csv",
  row.names = FALSE
)

# 3. Gradientes iniciales
grad0 <- rnas_mlp_backward(
  X = X,
  y = y,
  params = params0,
  activation = activation
)

tabla_grad_resumen <- data.frame(
  componente = c("grad_W", "grad_b1", "grad_v", "grad_b2", "grad_norm"),
  dimension = c(
    paste(dim(grad0$grad_W), collapse = " x "),
    as.character(length(grad0$grad_b1)),
    as.character(length(grad0$grad_v)),
    as.character(length(grad0$grad_b2)),
    "1"
  ),
  norma_o_valor = c(
    sqrt(sum(grad0$grad_W^2)),
    sqrt(sum(grad0$grad_b1^2)),
    sqrt(sum(grad0$grad_v^2)),
    abs(grad0$grad_b2),
    grad0$grad_norm
  )
)

print(tabla_grad_resumen)

write.csv(
  tabla_grad_resumen,
  file = "outputs/tables/cap06_gradientes_resumen.csv",
  row.names = FALSE
)

# 4. Entrenamiento completo
res_mlp <- rnas_train_mlp(
  X = X,
  y = y,
  d_hidden = d_hidden,
  params0 = params0,
  eta = eta,
  T = T,
  activation = activation,
  seed = seed
)

print(res_mlp)

# 5. Resumen del entrenamiento
tabla_resumen <- rnas_resumen_entrenamiento_mlp(res_mlp)

print(tabla_resumen)

write.csv(
  tabla_resumen,
  file = "outputs/tables/cap06_resumen_entrenamiento_mlp.csv",
  row.names = FALSE
)

# 6. Trayectoria completa y parcial
tabla_trayectoria <- res_mlp$trayectoria

write.csv(
  tabla_trayectoria,
  file = "outputs/tables/cap06_trayectoria_mlp.csv",
  row.names = FALSE
)

iter_representativas <- unique(c(
  0,
  1,
  2,
  5,
  10,
  25,
  50,
  100,
  150,
  200
))

tabla_trayectoria_libro <- tabla_trayectoria[
  tabla_trayectoria$iter %in% iter_representativas,
  c("iter", "loss", "grad_norm", "W_norm", "b1_norm", "v_norm", "b2")
]

print(tabla_trayectoria_libro)

write.csv(
  tabla_trayectoria_libro,
  file = "outputs/tables/cap06_trayectoria_libro.csv",
  row.names = FALSE
)

# 7. Predicciones finales
pred_final <- rnas_predict_mlp(res_mlp, X)

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
  file = "outputs/tables/cap06_predicciones_finales_mlp.csv",
  row.names = FALSE
)

# 8. Guardar objeto integral
res_cap06 <- list(
  datos = list(
    X = X,
    y = y,
    d_hidden = d_hidden,
    eta = eta,
    T = T,
    activation = activation,
    seed = seed
  ),
  params0 = params0,
  forward_inicial = forward0,
  perdida_inicial = loss0,
  gradiente_inicial = grad0,
  entrenamiento = res_mlp,
  resumen = tabla_resumen,
  trayectoria_libro = tabla_trayectoria_libro,
  predicciones_finales = tabla_pred_final
)

saveRDS(
  res_cap06,
  file = "outputs/results/cap06_resultados_mlp.rds"
)

# 9. Figura de perdida
pdf("outputs/figures/cap06_curva_perdida_mlp.pdf", width = 7, height = 5)

plot(
  tabla_trayectoria$iter,
  tabla_trayectoria$loss,
  type = "l",
  lwd = 2,
  xlab = "Iteracion",
  ylab = "Perdida MSE",
  main = "Evolucion de la perdida - MLP simple RNAS"
)

points(
  tabla_trayectoria_libro$iter,
  tabla_trayectoria_libro$loss,
  pch = 19
)

grid()

dev.off()

# 10. Figura de norma del gradiente
pdf("outputs/figures/cap06_norma_gradiente_mlp.pdf", width = 7, height = 5)

plot(
  tabla_trayectoria$iter,
  tabla_trayectoria$grad_norm,
  type = "l",
  lwd = 2,
  xlab = "Iteracion",
  ylab = "Norma del gradiente",
  main = "Evolucion de la norma del gradiente - MLP simple RNAS"
)

points(
  tabla_trayectoria_libro$iter,
  tabla_trayectoria_libro$grad_norm,
  pch = 19
)

grid()

dev.off()
