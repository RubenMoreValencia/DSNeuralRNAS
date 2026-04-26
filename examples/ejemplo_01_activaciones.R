# ============================================================
# Ejemplo 01: Activaciones base DSNeuralRNAS
# Capitulo 3: Activaciones y neurona individual
# ============================================================

library(DSNeuralRNAS)

# Crear carpeta de salida si no existe
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/results", recursive = TRUE, showWarnings = FALSE)

# Valores de prueba
z <- c(-3, -2, -1, 0, 1, 2, 3)

# Tabla de activaciones
tabla_activaciones <- data.frame(
  z = z,
  sigmoid = rnas_sigmoid(z),
  dsigmoid = rnas_dsigmoid(z),
  tanh = rnas_tanh(z),
  dtanh = rnas_dtanh(z)
)

print(tabla_activaciones)

# Guardar tabla
write.csv(
  tabla_activaciones,
  file = "outputs/tables/cap03_tabla_activaciones.csv",
  row.names = FALSE
)

saveRDS(
  tabla_activaciones,
  file = "outputs/results/cap03_tabla_activaciones.rds"
)

# Grafico base de activaciones
pdf("outputs/figures/cap03_activaciones_sigmoid_tanh.pdf", width = 7, height = 5)

z_grid <- seq(-5, 5, length.out = 300)

plot(
  z_grid,
  rnas_sigmoid(z_grid),
  type = "l",
  lwd = 2,
  ylim = c(-1, 1),
  xlab = "z",
  ylab = "Valor de activacion",
  main = "Activaciones base RNAS"
)

lines(z_grid, rnas_tanh(z_grid), lwd = 2, lty = 2)

abline(h = 0, lty = 3)
legend(
  "topleft",
  legend = c("Sigmoide", "Tanh"),
  lty = c(1, 2),
  lwd = 2,
  bty = "n"
)

dev.off()
