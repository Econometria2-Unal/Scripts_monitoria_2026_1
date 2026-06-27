#' Universidad Nacional de Colombia
#' Facultad de Ciencias Economicas
#'
#' Econometria II | Monitoria
#' Funciones auxiliares para la simulacion normal multivariada.

# Este archivo contiene funciones de apoyo para las secciones finales del
# script simulacion_normal_multivariada.R. La idea pedagogica es que el script
# principal conserve la lectura conceptual, mientras que los detalles tecnicos
# de comparacion, graficacion y exportacion quedan encapsulados aqui.

imprimir_parrafo = function(texto, ancho = 90) {
  cat(paste(strwrap(texto, width = ancho), collapse = "\n"), "\n\n", sep = "")
}

# ===
# 6. Comparacion de resultados manuales y con mvtnorm ----
# ===

comparar_manual_mvtnorm = function(metodo,
                                   U_manual,
                                   U_mvtnorm,
                                   tolerancia = 1e-12) {
  diferencia_maxima = max(abs(U_manual - U_mvtnorm))
  son_iguales = isTRUE(
    all.equal(
      U_manual,
      U_mvtnorm,
      tolerance = tolerancia,
      check.attributes = FALSE
    )
  )
  
  resumen = data.frame(
    metodo = metodo,
    iguales_manual_mvtnorm = son_iguales,
    diferencia_maxima = diferencia_maxima,
    media_u_1 = mean(U_manual[, "u_1"]),
    media_u_2 = mean(U_manual[, "u_2"]),
    var_u_1 = var(U_manual[, "u_1"]),
    cov_u_1_u_2 = cov(U_manual[, "u_1"], U_manual[, "u_2"]),
    var_u_2 = var(U_manual[, "u_2"]),
    cor_u_1_u_2 = cor(U_manual[, "u_1"], U_manual[, "u_2"])
  )
  
  return(resumen)
}

crear_tabla_comparacion_mvtnorm = function(U_manual_espectral,
                                           U_mvtnorm_espectral,
                                           U_manual_svd,
                                           U_mvtnorm_svd,
                                           U_manual_cholesky,
                                           U_mvtnorm_cholesky) {
  tabla_comparacion = rbind(
    comparar_manual_mvtnorm(
      "Descomposicion espectral",
      U_manual_espectral,
      U_mvtnorm_espectral
    ),
    comparar_manual_mvtnorm(
      "SVD",
      U_manual_svd,
      U_mvtnorm_svd
    ),
    comparar_manual_mvtnorm(
      "Cholesky",
      U_manual_cholesky,
      U_mvtnorm_cholesky
    )
  )
  
  return(tabla_comparacion)
}

imprimir_resumen_comparacion_mvtnorm = function(tabla_comparacion,
                                                U_manual_espectral,
                                                Sigma) {
  imprimir_parrafo(
    "Lectura conceptual de la comparacion: la simulacion manual y mvtnorm parten de la misma matriz Z de normales estandar. Por eso, si las matrices raiz de Sigma fueron construidas de la misma manera, ambas simulaciones deben coincidir observacion por observacion."
  )
  
  imprimir_parrafo(
    "La columna iguales_manual_mvtnorm indica si las dos matrices simuladas son numericamente iguales para cada metodo. La columna diferencia_maxima reporta el mayor error absoluto elemento a elemento. Valores TRUE y diferencias cercanas a cero muestran que la simulacion manual reproduce lo que hace mvtnorm internamente."
  )
  
  cat("\nComparacion manual vs. mvtnorm:\n")
  print(tabla_comparacion, row.names = FALSE)
  
  cat("\nCovarianza muestral - metodo espectral manual:\n")
  print(cov(U_manual_espectral))
  
  imprimir_parrafo(
    "La covarianza muestral no tiene que ser exactamente igual a Sigma porque se obtiene con una muestra finita. Sin embargo, con muchas observaciones debe estar muy cerca de la matriz teorica objetivo."
  )
  
  cat("\nCovarianza teorica objetivo Sigma:\n")
  print(Sigma)
  
  invisible(tabla_comparacion)
}

# ===
# 7. Graficas 2D interactivas con plotly ----
# ===

preparar_muestra_grafica = function(U,
                                    n_puntos = 8000,
                                    semilla_graficos = 202601) {
  set.seed(semilla_graficos)
  indices = sample(seq_len(nrow(U)), size = min(n_puntos, nrow(U)))
  datos = data.frame(
    u_1 = U[indices, 1],
    u_2 = U[indices, 2]
  )
  return(datos)
}

graficar_normal_bivariada = function(U,
                                     titulo,
                                     color_puntos = "#1f77b4",
                                     etiqueta_x = "u_1",
                                     etiqueta_y = "u_2",
                                     n_puntos = 8000,
                                     semilla_graficos = 202601) {
  datos_grafica = preparar_muestra_grafica(
    U = U,
    n_puntos = n_puntos,
    semilla_graficos = semilla_graficos
  )
  
  grafica = plotly::plot_ly()
  
  grafica = plotly::add_trace(
    grafica,
    data = datos_grafica,
    x = ~u_1,
    y = ~u_2,
    type = "histogram2dcontour",
    colorscale = "Viridis",
    reversescale = TRUE,
    showscale = FALSE,
    contours = list(
      coloring = "lines",
      showlabels = TRUE
    ),
    hoverinfo = "skip",
    name = "Contornos"
  )
  
  grafica = plotly::add_trace(
    grafica,
    data = datos_grafica,
    x = ~u_1,
    y = ~u_2,
    type = "scattergl",
    mode = "markers",
    marker = list(
      size = 4,
      color = color_puntos,
      opacity = 0.35
    ),
    hovertemplate = paste(
      etiqueta_x, ": %{x:.3f}<br>",
      etiqueta_y, ": %{y:.3f}<extra></extra>"
    ),
    name = "Observaciones simuladas"
  )
  
  grafica = plotly::layout(
    grafica,
    title = list(text = titulo),
    xaxis = list(title = etiqueta_x),
    yaxis = list(title = etiqueta_y, scaleanchor = "x", scaleratio = 1),
    legend = list(orientation = "h", x = 0, y = -0.15),
    margin = list(l = 60, r = 30, b = 80, t = 70)
  )
  
  return(grafica)
}

crear_graficas_2d_normal_multivariada = function(Z,
                                                 U_manual_espectral,
                                                 U_manual_svd,
                                                 U_manual_cholesky,
                                                 n_puntos = 8000,
                                                 semilla_graficos = 202601) {
  graficas = list(
    normal_estandar = graficar_normal_bivariada(
      Z,
      "Normal estandar bivariada no correlacionada",
      color_puntos = "#2a6fbb",
      etiqueta_x = "z_1",
      etiqueta_y = "z_2",
      n_puntos = n_puntos,
      semilla_graficos = semilla_graficos
    ),
    espectral = graficar_normal_bivariada(
      U_manual_espectral,
      "Normal bivariada correlacionada - Descomposicion espectral",
      color_puntos = "#c03a2b",
      n_puntos = n_puntos,
      semilla_graficos = semilla_graficos
    ),
    svd = graficar_normal_bivariada(
      U_manual_svd,
      "Normal bivariada correlacionada - SVD",
      color_puntos = "#2b8c56",
      n_puntos = n_puntos,
      semilla_graficos = semilla_graficos
    ),
    cholesky = graficar_normal_bivariada(
      U_manual_cholesky,
      "Normal bivariada correlacionada - Cholesky",
      color_puntos = "#6f4bb3",
      n_puntos = n_puntos,
      semilla_graficos = semilla_graficos
    )
  )
  
  return(graficas)
}

imprimir_descripcion_graficas_2d = function(n_observaciones,
                                            n_puntos_grafica) {
  imprimir_parrafo(
    paste0(
      "Lectura conceptual de las graficas 2D: se simularon ",
      n_observaciones,
      " observaciones, pero se grafica una submuestra de ",
      n_puntos_grafica,
      " puntos para que el archivo interactivo sea manejable."
    )
  )
  
  imprimir_parrafo(
    "La nube de la normal estandar no correlacionada debe verse aproximadamente circular. En cambio, la normal correlacionada debe verse alargada en direccion positiva, porque rho = 0.90 implica que valores altos de u_1 tienden a venir acompanados por valores altos de u_2."
  )
  
  imprimir_parrafo(
    "Los contornos resumen zonas de mayor y menor concentracion de probabilidad. Para una normal bivariada correlacionada, esos contornos tienen forma eliptica y su inclinacion refleja el signo y magnitud de la correlacion."
  )
}

# ===
# 8. Graficas 3D interactivas de densidad con plotly ----
# ===

crear_grilla_densidad = function(media,
                                 Sigma,
                                 n_grilla = 80,
                                 multiplicador_sd = 3.5) {
  desv = sqrt(diag(Sigma))
  
  eje_x = seq(
    from = media[1] - multiplicador_sd * desv[1],
    to = media[1] + multiplicador_sd * desv[1],
    length.out = n_grilla
  )
  
  eje_y = seq(
    from = media[2] - multiplicador_sd * desv[2],
    to = media[2] + multiplicador_sd * desv[2],
    length.out = n_grilla
  )
  
  densidad = outer(
    eje_x,
    eje_y,
    Vectorize(function(x, y) {
      mvtnorm::dmvnorm(
        x = c(x, y),
        mean = media,
        sigma = Sigma
      )
    })
  )
  
  return(list(
    x = eje_x,
    y = eje_y,
    z = t(densidad)
  ))
}

graficar_densidad_3d = function(media,
                                Sigma,
                                titulo,
                                etiqueta_x = "u_1",
                                etiqueta_y = "u_2",
                                escala_color = "Viridis",
                                n_grilla = 80,
                                multiplicador_sd = 3.5) {
  grilla = crear_grilla_densidad(
    media = media,
    Sigma = Sigma,
    n_grilla = n_grilla,
    multiplicador_sd = multiplicador_sd
  )
  
  grafica = plotly::plot_ly(
    x = grilla$x,
    y = grilla$y,
    z = grilla$z,
    type = "surface",
    colorscale = escala_color,
    contours = list(
      z = list(
        show = TRUE,
        usecolormap = TRUE,
        highlightcolor = "#ffffff",
        project = list(z = TRUE)
      )
    ),
    hovertemplate = paste(
      etiqueta_x, ": %{x:.3f}<br>",
      etiqueta_y, ": %{y:.3f}<br>",
      "Densidad: %{z:.5f}<extra></extra>"
    )
  )
  
  grafica = plotly::layout(
    grafica,
    title = list(text = titulo),
    scene = list(
      xaxis = list(title = etiqueta_x),
      yaxis = list(title = etiqueta_y),
      zaxis = list(title = "Densidad"),
      aspectmode = "cube",
      camera = list(
        eye = list(x = 1.55, y = -1.65, z = 1.15)
      )
    ),
    margin = list(l = 0, r = 0, b = 0, t = 70)
  )
  
  return(grafica)
}

crear_graficas_3d_normal_multivariada = function(media,
                                                 Sigma,
                                                 n_grilla = 80,
                                                 multiplicador_sd = 3.5) {
  graficas = list(
    normal_estandar_3d = graficar_densidad_3d(
      media = c(z_1 = 0, z_2 = 0),
      Sigma = diag(2),
      titulo = "Densidad 3D - Normal estandar bivariada",
      etiqueta_x = "z_1",
      etiqueta_y = "z_2",
      escala_color = "Viridis",
      n_grilla = n_grilla,
      multiplicador_sd = multiplicador_sd
    ),
    espectral_3d = graficar_densidad_3d(
      media = media,
      Sigma = Sigma,
      titulo = "Densidad 3D - Descomposicion espectral",
      escala_color = "YlOrRd",
      n_grilla = n_grilla,
      multiplicador_sd = multiplicador_sd
    ),
    svd_3d = graficar_densidad_3d(
      media = media,
      Sigma = Sigma,
      titulo = "Densidad 3D - SVD",
      escala_color = "Plasma",
      n_grilla = n_grilla,
      multiplicador_sd = multiplicador_sd
    ),
    cholesky_3d = graficar_densidad_3d(
      media = media,
      Sigma = Sigma,
      titulo = "Densidad 3D - Cholesky",
      escala_color = "Portland",
      n_grilla = n_grilla,
      multiplicador_sd = multiplicador_sd
    )
  )
  
  return(graficas)
}

imprimir_descripcion_graficas_3d = function() {
  imprimir_parrafo(
    "Lectura conceptual de las graficas 3D: la superficie muestra la funcion de densidad teorica de la normal bivariada. La altura representa que tan probable es observar combinaciones cercanas a cada punto (u_1, u_2)."
  )
  
  imprimir_parrafo(
    "Para la normal estandar no correlacionada, la superficie es simetrica alrededor de cero. Para la normal correlacionada, la base de la superficie se estira de acuerdo con Sigma. Los tres metodos de descomposicion generan la misma distribucion objetivo, por eso sus superficies teoricas son iguales salvo por el titulo y la escala de color."
  )
}

# ===
# 9. Exportacion de graficas a archivos HTML ----
# ===

obtener_directorio_proyecto_exportacion = function() {
  candidatos = c(
    if (exists("directorio_proyecto", envir = .GlobalEnv)) {
      as.character(get("directorio_proyecto", envir = .GlobalEnv))
    },
    tryCatch(as.character(here::here()), error = function(e) NA_character_),
    getwd(),
    normalizePath(file.path(getwd(), ".."), winslash = "/", mustWork = FALSE),
    normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = FALSE),
    normalizePath(file.path(getwd(), "..", "..", ".."), winslash = "/", mustWork = FALSE)
  )
  
  candidatos = unique(candidatos[!is.na(candidatos) & nzchar(candidatos)])
  
  for (candidato in candidatos) {
    if (dir.exists(file.path(candidato, directorio_codigo_R))) {
      return(candidato)
    }
  }
  
  stop(
    "No se pudo ubicar la raiz del proyecto para exportar las graficas HTML. ",
    "Ejecuta primero el bloque de rutas con here::i_am()."
  )
}

usar_directorio_proyecto = function(expr) {
  directorio_trabajo_original = getwd()
  on.exit(setwd(directorio_trabajo_original), add = TRUE)
  
  setwd(obtener_directorio_proyecto_exportacion())
  force(expr)
}

definir_directorio_graficas = function(nombre_directorio = "html_nm",
                                       directorio_base = directorio_codigo_R) {
  directorio_graficas = fs::path(directorio_base, nombre_directorio)
  directorio_proyecto_exportacion = obtener_directorio_proyecto_exportacion()
  directorio_graficas_abs = fs::path_abs(
    directorio_graficas,
    start = directorio_proyecto_exportacion
  )
  
  dir.create(directorio_graficas_abs, recursive = TRUE, showWarnings = FALSE)
  
  if (!dir.exists(directorio_graficas_abs)) {
    stop(
      "No se pudo crear el directorio de graficas HTML: ",
      directorio_graficas
    )
  }
  
  return(directorio_graficas)
}
guardar_grafica_html = function(grafica,
                                nombre_archivo,
                                directorio_graficas) {
  ruta_archivo = fs::path(directorio_graficas, nombre_archivo)
  directorio_proyecto_exportacion = obtener_directorio_proyecto_exportacion()
  ruta_archivo_abs = fs::path_abs(
    ruta_archivo,
    start = directorio_proyecto_exportacion
  )
  directorio_graficas_abs = fs::path_dir(ruta_archivo_abs)
  ruta_temporal = file.path(tempdir(), nombre_archivo)
  
  if (file.exists(ruta_temporal)) {
    unlink(ruta_temporal)
  }
  
  htmlwidgets::saveWidget(
    widget = grafica,
    file = ruta_temporal,
    selfcontained = TRUE,
    title = tools::file_path_sans_ext(nombre_archivo)
  )
  
  dir.create(directorio_graficas_abs, recursive = TRUE, showWarnings = FALSE)
  
  if (file.exists(ruta_archivo_abs)) {
    unlink(ruta_archivo_abs, force = TRUE)
  }
  
  copia_exitosa = suppressWarnings(
    file.copy(
      from = ruta_temporal,
      to = ruta_archivo_abs,
      overwrite = FALSE
    )
  )
  
  if (!isTRUE(copia_exitosa) || !file.exists(ruta_archivo_abs)) {
    try(
      fs::file_copy(
        path = ruta_temporal,
        new_path = ruta_archivo_abs,
        overwrite = TRUE
      ),
      silent = TRUE
    )
  }
  
  if (!file.exists(ruta_archivo_abs)) {
    stop(
      "No se pudo copiar la grafica HTML a: ",
      ruta_archivo
    )
  }
  
  cat(
    "Grafica guardada: ",
    ruta_archivo,
    "\n",
    sep = ""
  )
  
  invisible(ruta_archivo)
}
exportar_graficas_html = function(graficas,
                                  directorio_graficas = definir_directorio_graficas()) {
  if (is.null(names(graficas)) || any(names(graficas) == "")) {
    stop("El objeto graficas debe ser una lista nombrada: nombre_archivo.html = grafica.")
  }
  
  directorio_proyecto_exportacion = obtener_directorio_proyecto_exportacion()
  directorio_graficas_abs = fs::path_abs(
    directorio_graficas,
    start = directorio_proyecto_exportacion
  )
  dir.create(directorio_graficas_abs, recursive = TRUE, showWarnings = FALSE)
  
  rutas = mapply(
    FUN = function(nombre_archivo, grafica) {
      guardar_grafica_html(
        grafica = grafica,
        nombre_archivo = nombre_archivo,
        directorio_graficas = directorio_graficas
      )
    },
    nombre_archivo = names(graficas),
    grafica = graficas,
    SIMPLIFY = FALSE,
    USE.NAMES = TRUE
  )
  
  cat(
    "\nDirectorio con graficas HTML:\n",
    directorio_graficas,
    "\n",
    sep = ""
  )
  
  invisible(rutas)
}

imprimir_descripcion_exportacion_html = function(directorio_graficas) {
  imprimir_parrafo(
    "Lectura conceptual de la exportacion: cada grafica se guarda como HTML autocontenido para que pueda abrirse en un navegador sin depender de la sesion de R. Esto facilita compartir las visualizaciones y revisarlas despues de ejecutar el script."
  )
  
  imprimir_parrafo(
    paste0(
      "Los archivos se guardaran en: ",
      directorio_graficas
    )
  )
}
mostrar_graficas_interactivo = function(graficas) {
  if (interactive()) {
    invisible(lapply(graficas, print))
  }
}
