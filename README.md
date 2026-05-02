# DSNeuralRNAS

**DSNeuralRNAS** es un proyecto académico y computacional desarrollado en lenguaje R para estudiar el aprendizaje neuronal artificial desde una perspectiva de dinámica de sistemas. El proyecto integra redes neuronales, trayectorias de aprendizaje, pérdida, gradiente, tasa de aprendizaje, curvatura, regímenes dinámicos, control y meta-aprendizaje geométrico.

El proyecto forma parte del desarrollo del libro **DS Neural RNAS**, orientado a construir una arquitectura formal, computacional y explicativa donde el aprendizaje no se observe únicamente como resultado final, sino como una trayectoria dinámica.

---

## Propósito del proyecto

El objetivo general de **DSNeuralRNAS** es proporcionar una base funcional en R para analizar procesos de aprendizaje neuronal como sistemas dinámicos. En este enfoque, una red neuronal no solo predice o ajusta datos, sino que produce una trayectoria interna compuesta por parámetros, errores, gradientes, tasas, curvaturas, regímenes y señales de estabilidad.

Desde esta perspectiva, el aprendizaje artificial se interpreta como un proceso dinámico de transformación, adaptación y control.

---

## Idea central

DS Neural RNAS parte de una **base dinámica observada**, formada por las variables y comportamientos originales del sistema, y produce una **base dinámica aprendida**, formada por los parámetros, señales, trayectorias internas, regímenes y posibles variables emergentes generadas durante el aprendizaje.

La teoría del meta-aprendizaje por dinámica estudia cómo ambas bases pueden integrarse para comprender el sistema completo.

---

## Componentes principales

El paquete contiene funciones asociadas a los principales bloques técnicos del modelo:

- Activaciones y neurona individual.
- Pérdida, gradiente y verificación numérica.
- Entrenamiento discreto de una neurona.
- MLP simple.
- Dinámica continua del aprendizaje.
- Geometría del aprendizaje.
- Regímenes dinámicos.
- Control de tasa de aprendizaje.
- Control óptimo y meta-aprendizaje geométrico.
- Integración con STD2, SimuDS y formulaciones no lineales.
- Casos aplicados y demostraciones funcionales.

---

## Instalación

El paquete puede instalarse desde GitHub mediante `remotes`:

```r
install.packages("remotes")

remotes::install_github(
  "RubenMoreValencia/DSNeuralRNAS",
  force = TRUE,
  upgrade = "never"
)
```

Luego puede cargarse con:

```r
library(DSNeuralRNAS)
```

Para verificar la instalación:

```r
packageVersion("DSNeuralRNAS")
find.package("DSNeuralRNAS")
```

---

## Ejemplo básico: neurona individual

```r
library(DSNeuralRNAS)

x <- c(0.5, -0.2)
w <- c(0.8, 0.3)
b <- 0.1

y_hat <- rnas_neuron_forward(
  x = x,
  w = w,
  b = b,
  activation = "tanh"
)

y_hat
```

---

## Ejemplo básico: pérdida y gradiente

```r
X <- matrix(
  c(
    0.5, -0.2,
    1.0,  0.3,
   -0.7,  0.8,
    0.0,  0.0
  ),
  ncol = 2,
  byrow = TRUE
)

y <- c(0.4, 0.8, -0.2, 0.1)

w <- c(0.8, 0.3)
b <- 0.1

loss <- rnas_loss_mse_neuron(
  X = X,
  y = y,
  w = w,
  b = b,
  activation = "tanh"
)

grad <- rnas_grad_neuron(
  X = X,
  y = y,
  w = w,
  b = b,
  activation = "tanh"
)

loss
grad
```

---

## Ejemplo básico: entrenamiento con control de tasa

```r
theta0 <- c(0.8, 0.3, 0.1)

modelo_eta <- rnas_train_neuron_control_eta(
  X = X,
  y = y,
  theta0 = theta0,
  eta0 = 0.1,
  policy = "constante",
  eta_min = 0.01,
  eta_max = 0.1,
  T = 100,
  activation = "tanh"
)

head(modelo_eta$trayectoria)
tail(modelo_eta$trayectoria)
```

---

## Ejemplo básico: dinámica continua

```r
dinamica <- rnas_integrar_dinamica_neuron(
  X = X,
  y = y,
  theta0 = theta0,
  eta = 0.1,
  dt = 0.1,
  T = 100,
  activation = "tanh"
)

head(dinamica$trayectoria)
tail(dinamica$trayectoria)
```

---

## Ejemplo básico: geometría del aprendizaje

```r
theta <- c(0.8, 0.3, 0.1)

H <- rnas_hessian_num_neuron(
  X = X,
  y = y,
  theta = theta,
  activation = "tanh",
  h = 1e-4
)

rnas_autovalores_hessian(H)

v <- c(1, 0, 0)

rnas_curvatura_direccional(
  H = H,
  v = v
)
```

---

## Ejemplo básico: regímenes dinámicos

```r
analisis_reg <- rnas_analizar_regimenes_neuron(
  object = dinamica,
  ventana = 5L,
  tau_loss = 0.01,
  tau_grad = 0.001,
  eps_loss = 1e-8,
  eps_grad = 1e-4,
  tau_speed = Inf,
  loss_alta = NULL,
  usar_suavizado = TRUE
)

analisis_reg
```

---

## Aplicación web DSNeuralRNASLab

El proyecto cuenta con una aplicación Shiny de exploración funcional:

- Web: <https://npupmx-ruben0a0-more0valencia.shinyapps.io/DSNeuralRNASLab/>
- Repositorio del laboratorio: <https://github.com/RubenMoreValencia/DSNeuralRNASLab>

La aplicación permite seleccionar capítulos técnicos, modificar parámetros, ejecutar funciones del paquete, revisar código de referencia, visualizar tablas, analizar gráficas e interpretar resultados.

---

## Repositorio general

Repositorio del paquete principal:

- <https://github.com/RubenMoreValencia/DSNeuralRNAS>

---

## Relación con la teoría del meta-aprendizaje por dinámica

DSNeuralRNAS constituye una base formal y computacional para avanzar hacia una teoría del meta-aprendizaje por dinámica. Esta teoría busca estudiar no solo qué aprende un modelo, sino cómo se construye la trayectoria de aprendizaje y qué componentes dinámicos emergen durante ese proceso.

En esta línea, los pesos, sesgos, pérdidas, gradientes, tasas, curvaturas, regímenes y trayectorias no se consideran únicamente elementos técnicos del entrenamiento, sino posibles datos dinámicos generados por el aprendizaje.

La propuesta distingue dos bases:

- **Base dinámica observada:** variables y comportamientos originales del sistema.
- **Base dinámica aprendida:** parámetros, señales, trayectorias internas, regímenes y posibles variables emergentes generadas por el aprendizaje.

La integración de ambas bases permite avanzar hacia la comprensión funcional del sistema completo.

---

## Estado del proyecto

El proyecto se encuentra en desarrollo académico y computacional. Su propósito es apoyar investigación, docencia, experimentación y construcción teórica en torno a redes neuronales, dinámica de sistemas y meta-aprendizaje por trayectoria.

---

## Autor

**Rubén Alexander More Valencia**

Profesor del Departamento Académico de Investigación de Operaciones.

Director de Investigación.

Escuela de Posgrado de la Universidad Nacional de Piura.

Correo: <rmorev@unp.edu.pe>

---

## Nota

Este proyecto tiene finalidad académica, investigativa y experimental. Las funciones y ejemplos deben entenderse como parte de una arquitectura en desarrollo, orientada a construir una lectura dinámica del aprendizaje neuronal artificial y su posible extensión hacia sistemas complejos.
