# ============================================================
# Ejemplo 03: Pérdida, gradiente y verificación diferencial
# Capítulo 4: Pérdida y gradiente
# ============================================================

library(DSNeuralRNAS)

# Crear carpetas de salida si no existen
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
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

w <- c(0.8, 0.3)
b <- 0.1
activation <- "tanh"

# 1. Cálculo de pérdida con detalle
res_loss <- rnas_loss_mse_neuron(
  X = X,
  y = y,
  w = w,
  b = b,
  activation = activation,
  devolver_detalle = TRUE
)

tabla_predicciones <- data.frame(
  obs = seq_len(nrow(X)),
  x_1 = X[, 1],
  x_2 = X[, 2],
  y_obs = y,
  z = res_loss$z,
  y_hat = res_loss$y_hat,
  error = res_loss$error,
  error_cuadrado = res_loss$error^2
)

print(tabla_predicciones)

write.csv(
  tabla_predicciones,
  file = "outputs/tables/cap04_predicciones_errores.csv",
  row.names = FALSE
)

# 2. Resumen de pérdida
tabla_loss <- data.frame(
  activation = activation,
  n = nrow(X),
  loss_mse = res_loss$loss,
  perdida_no_negativa = res_loss$loss >= 0
)

print(tabla_loss)

write.csv(
  tabla_loss,
  file = "outputs/tables/cap04_resumen_loss.csv",
  row.names = FALSE
)

# 3. Gradiente analítico
res_grad <- rnas_grad_neuron(
  X = X,
  y = y,
  w = w,
  b = b,
  activation = activation
)

tabla_gradiente <- data.frame(
  componente = c("grad_w1", "grad_w2", "grad_b"),
  valor = c(res_grad$grad_w, res_grad$grad_b)
)

print(tabla_gradiente)

write.csv(
  tabla_gradiente,
  file = "outputs/tables/cap04_gradiente_analitico.csv",
  row.names = FALSE
)

# 4. Gradiente numérico
res_grad_num <- rnas_grad_num_neuron(
  X = X,
  y = y,
  w = w,
  b = b,
  activation = activation,
  h = 1e-6
)

tabla_gradiente_num <- data.frame(
  componente = c("grad_w1", "grad_w2", "grad_b"),
  valor = c(res_grad_num$grad_w, res_grad_num$grad_b)
)

print(tabla_gradiente_num)

write.csv(
  tabla_gradiente_num,
  file = "outputs/tables/cap04_gradiente_numerico.csv",
  row.names = FALSE
)

# 5. Verificación diferencial
res_check <- rnas_grad_check_neuron(
  X = X,
  y = y,
  w = w,
  b = b,
  activation = activation,
  h = 1e-6,
  tol = 1e-5
)

tabla_check <- data.frame(
  indicador = c(
    "loss",
    "diff_abs",
    "error_rel",
    "tol",
    "verificado"
  ),
  valor = c(
    res_check$loss,
    res_check$diff_abs,
    res_check$error_rel,
    res_check$tol,
    res_check$verificado
  )
)

print(tabla_check)

write.csv(
  tabla_check,
  file = "outputs/tables/cap04_verificacion_gradiente.csv",
  row.names = FALSE
)

# 6. Guardar objeto integral de resultados
res_cap04 <- list(
  datos = list(
    X = X,
    y = y,
    w = w,
    b = b,
    activation = activation
  ),
  perdida = res_loss,
  gradiente_analitico = res_grad,
  gradiente_numerico = res_grad_num,
  verificacion = res_check,
  tablas = list(
    predicciones = tabla_predicciones,
    loss = tabla_loss,
    gradiente = tabla_gradiente,
    gradiente_numerico = tabla_gradiente_num,
    check = tabla_check
  )
)

saveRDS(
  res_cap04,
  file = "outputs/results/cap04_resultados_perdida_gradiente.rds"
)
