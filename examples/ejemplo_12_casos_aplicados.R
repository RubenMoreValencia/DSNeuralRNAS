# ============================================================
# Ejemplo 12: Casos aplicados y resultados consolidados
# Capitulo 13: Casos aplicados de DS Neural RNAS
# ============================================================

library(DSNeuralRNAS)

# Crear carpetas de salida si no existen
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/results", recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Ejecutar caso STD2-RNAS
# ============================================================

caso_std2 <- rnas_caso_std2_controlado(
  n = 60,
  horizonte = 1,
  eta0 = 0.05,
  T = 100,
  activation = "tanh",
  seed = 123
)

print(caso_std2)
print(caso_std2$resumen)

write.csv(
  caso_std2$datos,
  file = "outputs/tables/cap13_caso_std2_datos.csv",
  row.names = FALSE
)

write.csv(
  caso_std2$integracion$preparacion$datos_alineados,
  file = "outputs/tables/cap13_caso_std2_alineados.csv",
  row.names = FALSE
)

write.csv(
  caso_std2$resumen,
  file = "outputs/tables/cap13_caso_std2_resumen.csv",
  row.names = FALSE
)

# ============================================================
# 2. Ejecutar caso SimuDS-RNAS
# ============================================================

caso_simuds <- rnas_caso_simuds_controlado(
  n = 40,
  horizonte = 1,
  eta0 = 0.05,
  T = 100,
  activation = "tanh",
  seed = 123
)

print(caso_simuds)
print(caso_simuds$resumen)

write.csv(
  caso_simuds$trayectorias,
  file = "outputs/tables/cap13_caso_simuds_trayectorias.csv",
  row.names = FALSE
)

write.csv(
  caso_simuds$integracion$preparacion$pares,
  file = "outputs/tables/cap13_caso_simuds_pares.csv",
  row.names = FALSE
)

write.csv(
  caso_simuds$resumen,
  file = "outputs/tables/cap13_caso_simuds_resumen.csv",
  row.names = FALSE
)

# ============================================================
# 3. Ejecutar caso FNL-RNAS
# ============================================================

caso_fnl <- rnas_caso_fnl_controlado(
  n = 60,
  activation = "tanh",
  seed = 123
)

print(caso_fnl)
print(caso_fnl$resumen)
print(caso_fnl$diagnostico)

write.csv(
  caso_fnl$diagnostico,
  file = "outputs/tables/cap13_caso_fnl_diagnostico.csv",
  row.names = FALSE
)

write.csv(
  caso_fnl$resumen,
  file = "outputs/tables/cap13_caso_fnl_resumen.csv",
  row.names = FALSE
)

# ============================================================
# 4. Consolidar casos aplicados
# ============================================================

res_casos <- rnas_ejecutar_casos_aplicados(
  n_std2 = 60,
  n_simuds = 40,
  n_fnl = 60,
  horizonte = 1,
  eta0 = 0.05,
  T = 100,
  activation = "tanh",
  seed = 123
)

print(res_casos)
print(res_casos$resumen_casos)

write.csv(
  res_casos$resumen_casos,
  file = "outputs/tables/cap13_resumen_casos_aplicados.csv",
  row.names = FALSE
)

tabla_diagnostico <- data.frame(
  indicador = c(
    "casos_ejecutados",
    "casos_con_descenso",
    "casos_factibles"
  ),
  valor = c(
    res_casos$diagnostico$n_casos,
    res_casos$diagnostico$casos_con_descenso,
    res_casos$diagnostico$casos_factibles
  )
)

print(tabla_diagnostico)

write.csv(
  tabla_diagnostico,
  file = "outputs/tables/cap13_diagnostico_casos.csv",
  row.names = FALSE
)

# ============================================================
# 5. Guardar objeto integral
# ============================================================

saveRDS(
  res_casos,
  file = "outputs/results/cap13_resultados_casos_aplicados.rds"
)

# ============================================================
# 6. Figuras simples para el capítulo
# ============================================================

pdf("outputs/figures/cap13_caso_std2_serie.pdf", width = 7, height = 5)

plot(
  caso_std2$datos$tiempo,
  caso_std2$datos$y,
  type = "l",
  lwd = 2,
  xlab = "Tiempo",
  ylab = "Valor",
  main = "Caso STD2-RNAS: serie observada y componente estructural"
)

lines(
  caso_std2$datos$tiempo,
  caso_std2$datos$C,
  lwd = 2,
  lty = 2
)

legend(
  "topleft",
  legend = c("y", "C"),
  lty = c(1, 2),
  lwd = 2,
  bty = "n"
)

grid()

dev.off()

pdf("outputs/figures/cap13_caso_simuds_trayectorias.pdf", width = 7, height = 5)

plot(
  NULL,
  xlim = range(caso_simuds$trayectorias$t),
  ylim = range(caso_simuds$trayectorias$s),
  xlab = "Tiempo",
  ylab = "Estado s",
  main = "Caso SimuDS-RNAS: trayectorias por escenario"
)

for (esc in unique(caso_simuds$trayectorias$escenario)) {
  sub <- caso_simuds$trayectorias[caso_simuds$trayectorias$escenario == esc, ]
  lines(sub$t, sub$s, lwd = 2)
}

legend(
  "topleft",
  legend = unique(caso_simuds$trayectorias$escenario),
  lty = 1,
  lwd = 2,
  bty = "n"
)

grid()

dev.off()

pdf("outputs/figures/cap13_comparacion_loss_final.pdf", width = 7, height = 5)

tab_plot <- res_casos$resumen_casos
tab_plot <- tab_plot[!is.na(tab_plot$loss_final), ]

barplot(
  height = tab_plot$loss_final,
  names.arg = tab_plot$caso,
  las = 2,
  ylab = "Pérdida / valor objetivo final",
  main = "Comparación de resultados finales por caso"
)

grid()

dev.off()
