library(shiny)
library(DT)

if (!requireNamespace("DSNeuralRNAS", quietly = TRUE)) {
  stop("Debe instalar o cargar el paquete DSNeuralRNAS antes de ejecutar la app.")
}

library(DSNeuralRNAS)

source("R/helpers_app.R")
source("R/mod_inicio.R")
source("R/mod_cap03_activaciones.R")
source("R/mod_cap04_perdida_gradiente.R")
source("R/mod_cap05_entrenamiento_neurona.R")
source("R/mod_cap06_mlp.R")
source("R/mod_cap07_dinamica_continua.R")
source("R/mod_cap08_geometria.R")
source("R/mod_cap09_regimenes.R")
source("R/mod_cap10_control_eta.R")
source("R/mod_cap11_control_optimo.R")
source("R/mod_cap12_integracion.R")
source("R/mod_cap13_casos.R")
