# ============================================================
# Ejemplo 02: Neurona individual DSNeuralRNAS
# Capitulo 3: Activaciones y neurona individual
# ============================================================

library(DSNeuralRNAS)

# Crear carpeta de salida si no existe
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/results", recursive = TRUE, showWarnings = FALSE)

# Parametros de ejemplo
x <- c(0.5, -0.2)
w <- c(0.8, 0.3)
b <- 0.1

# Evaluacion de neurona individual
res_tanh <- rnas_neuron_forward(
  x = x,
  w = w,
  b = b,
  activation = "tanh",
  devolver_z = TRUE
)

res_sigmoid <- rnas_neuron_forward(
  x = x,
  w = w,
  b = b,
  activation = "sigmoid",
  devolver_z = TRUE
)

# Tabla resumen de una observacion
tabla_neurona <- data.frame(
  caso = c("tanh", "sigmoid"),
  x_1 = x[1],
  x_2 = x[2],
  w_1 = w[1],
  w_2 = w[2],
  b = b,
  z = c(res_tanh$z, res_sigmoid$z),
  y_hat = c(res_tanh$y_hat, res_sigmoid$y_hat)
)

print(tabla_neurona)

write.csv(
  tabla_neurona,
  file = "outputs/tables/cap03_resultado_neurona_individual.csv",
  row.names = FALSE
)

saveRDS(
  tabla_neurona,
  file = "outputs/results/cap03_resultado_neurona_individual.rds"
)

# Evaluacion por lotes
X <- matrix(
  c(0.5, -0.2,
    1.0,  0.3,
    -0.7,  0.8,
    0.0,  0.0),
  ncol = 2,
  byrow = TRUE
)

res_batch <- rnas_neuron_forward_batch(
  X = X,
  w = w,
  b = b,
  activation = "tanh",
  devolver_z = TRUE
)

tabla_batch <- data.frame(
  obs = seq_len(nrow(X)),
  x_1 = X[, 1],
  x_2 = X[, 2],
  z = res_batch$z,
  y_hat = res_batch$y_hat,
  activation = res_batch$activation
)

print(tabla_batch)

write.csv(
  tabla_batch,
  file = "outputs/tables/cap03_resultado_neurona_batch.csv",
  row.names = FALSE
)

saveRDS(
  tabla_batch,
  file = "outputs/results/cap03_resultado_neurona_batch.rds"
)
