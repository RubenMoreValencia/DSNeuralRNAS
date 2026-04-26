# ============================================================
# Modulo Inicio: Presentacion general DSNeuralRNASLab
# ============================================================

mod_inicio_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h2("DSNeuralRNASLab"),

    div(
      class = "bloque-explicacion",
      h3("Laboratorio interactivo del libro DS Neural RNAS"),
      p(
        "Esta aplicación Shiny acompaña el libro académico-científico ",
        strong("DS Neural RNAS"),
        ", y permite revisar de forma interactiva los principales componentes técnicos del modelo."
      ),
      p(
        "El propósito de la app es facilitar la exploración funcional del paquete ",
        code("DSNeuralRNAS"),
        ", permitiendo modificar parámetros, ejecutar funciones, observar resultados tabulares y visualizar gráficas sin que el usuario tenga que escribir código."
      )
    ),

    fluidRow(
      column(
        6,
        h3("Datos del proyecto"),

        tags$table(
          class = "table table-bordered table-striped",
          tags$tbody(
            tags$tr(
              tags$th("Título"),
              tags$td("DS Neural RNAS")
            ),
            tags$tr(
              tags$th("Subtítulo"),
              tags$td("Arquitectura de aprendizaje dinámico entre redes neuronales y dinámica de sistemas")
            ),
            tags$tr(
              tags$th("Aplicación"),
              tags$td("DSNeuralRNASLab")
            ),
            tags$tr(
              tags$th("Tipo de recurso"),
              tags$td("Laboratorio Shiny académico para validación, exploración y docencia")
            ),
            tags$tr(
              tags$th("Paquete R asociado"),
              tags$td(code("DSNeuralRNAS"))
            )
          )
        )
      ),

      column(
        6,
        h3("Autoría"),

        tags$table(
          class = "table table-bordered table-striped",
          tags$tbody(
            tags$tr(
              tags$th("Autor"),
              tags$td("Rubén Alexander More Valencia")
            ),
            tags$tr(
              tags$th("Cargo académico"),
              tags$td("Profesor del Departamento Académico de Investigación de Operaciones")
            ),
            tags$tr(
              tags$th("Responsabilidad institucional"),
              tags$td("Director de Investigación")
            ),
            tags$tr(
              tags$th("Institución"),
              tags$td("Escuela de Posgrado de la Universidad Nacional de Piura")
            ),
            tags$tr(
              tags$th("Correo"),
              tags$td(tags$a(href = "mailto:rmorev@unp.edu.pe", "rmorev@unp.edu.pe"))
            )
          )
        )
      )
    ),

    hr(),

    h3("Relación entre libro, paquete R y aplicación"),

    div(
      class = "bloque-explicacion",
      tags$ul(
        tags$li(strong("Libro:"), " desarrolla los fundamentos matemáticos, la arquitectura conceptual, los algoritmos, la interpretación DS Neural y la discusión académica."),
        tags$li(strong("Paquete R:"), " implementa las funciones, pruebas unitarias, ejemplos reproducibles y salidas computacionales."),
        tags$li(strong("Aplicación Shiny:"), " permite experimentar con los módulos técnicos del libro mediante parámetros editables, tablas, gráficas y código de referencia.")
      )
    ),

    h3("Estructura del laboratorio por capítulos"),

    tags$table(
      class = "table table-bordered table-hover",
      tags$thead(
        tags$tr(
          tags$th("Capítulo"),
          tags$th("Módulo"),
          tags$th("Propósito funcional")
        )
      ),
      tags$tbody(
        tags$tr(
          tags$td("Capítulo 3"),
          tags$td("Activaciones y neurona individual"),
          tags$td("Evaluar funciones de activación, pesos, sesgo, entrada neuronal y salida activada.")
        ),
        tags$tr(
          tags$td("Capítulo 4"),
          tags$td("Pérdida, gradiente y verificación"),
          tags$td("Calcular pérdida MSE, gradiente analítico, gradiente numérico y consistencia de la derivación.")
        ),
        tags$tr(
          tags$td("Capítulo 5"),
          tags$td("Entrenamiento discreto de neurona"),
          tags$td("Entrenar una neurona RNAS y revisar pérdida, gradiente, trayectoria y parámetros finales.")
        ),
        tags$tr(
          tags$td("Capítulo 6"),
          tags$td("MLP simple"),
          tags$td("Ejecutar una arquitectura multicapa simple, revisar predicción, pérdida y trayectoria de entrenamiento.")
        ),
        tags$tr(
          tags$td("Capítulo 7"),
          tags$td("Dinámica continua del aprendizaje"),
          tags$td("Interpretar el aprendizaje como trayectoria dinámica con paso temporal, velocidad, pérdida y gradiente.")
        ),
        tags$tr(
          tags$td("Capítulo 8"),
          tags$td("Geometría del aprendizaje"),
          tags$td("Analizar paisaje de pérdida, Hessiano, autovalores, curvatura y geometría de trayectoria.")
        ),
        tags$tr(
          tags$td("Capítulo 9"),
          tags$td("Regímenes dinámicos"),
          tags$td("Clasificar fases del aprendizaje según pérdida, gradiente, velocidad, señales suavizadas y segmentos.")
        ),
        tags$tr(
          tags$td("Capítulo 10"),
          tags$td("Control de tasa de aprendizaje"),
          tags$td("Explorar políticas de control de eta, factores gamma, pérdida, gradiente, velocidad y trayectoria.")
        ),
        tags$tr(
          tags$td("Capítulo 11"),
          tags$td("Control óptimo y meta-aprendizaje geométrico"),
          tags$td("Evaluar eta óptima local, costo por eta, curvatura, lambda máxima y comparación de políticas meta.")
        ),
        tags$tr(
          tags$td("Capítulo 12"),
          tags$td("Integración STD2, SimuDS y FNL"),
          tags$td("Probar integraciones con señales temporales, trayectorias simuladas y formulación funcional no lineal.")
        ),
        tags$tr(
          tags$td("Capítulo 13"),
          tags$td("Casos aplicados"),
          tags$td("Ejecutar casos controlados y comparar resultados consolidados de las integraciones aplicadas.")
        )
      )
    ),

    hr(),

    h3("Modo de uso"),

    div(
      class = "bloque-explicacion",
      tags$ol(
        tags$li("Seleccione un capítulo en el panel lateral."),
        tags$li("Modifique los parámetros disponibles según el módulo."),
        tags$li("Ejecute el modelo o análisis correspondiente."),
        tags$li("Revise el código de referencia, los datos, las tablas, las gráficas y la interpretación generada."),
        tags$li("Compare los resultados con las secciones técnicas del libro.")
      )
    ),

    h3("Nota académica"),

    div(
      class = "bloque-explicacion",
      p(
        "Los ejemplos incluidos en la aplicación tienen finalidad académica, demostrativa y de validación funcional. ",
        "No sustituyen una calibración completa del modelo sobre datos reales, pero permiten comprender la lógica computacional de cada capítulo técnico."
      ),
      p(
        "La aplicación se diseñó para apoyar lectura, docencia, revisión metodológica y experimentación inicial con el paquete ",
        code("DSNeuralRNAS"),
        "."
      )
    )
  )
}


mod_inicio_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    # Modulo informativo sin calculos reactivos.
  })
}
