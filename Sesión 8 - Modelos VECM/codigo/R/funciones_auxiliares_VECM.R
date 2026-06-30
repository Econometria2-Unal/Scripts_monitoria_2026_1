# Funciones auxiliares para modelos VECM y metodologia Johansen

# 1. Funciones auxiliares generales ----

#' Crea una capa de linea para graficas de ggplot2.
#'
#' @param ... Argumentos adicionales que se entregan a `geom_line`.
#' @param ancho Grosor de la linea.
#'
#' @return Una capa de ggplot2 para agregar lineas a una grafica.
geom_linea_actual = function(..., ancho = 0.5){
  if (packageVersion("ggplot2") >= "3.4.0") {
    geom_line(..., linewidth = ancho)
  } else {
    geom_line(..., size = ancho)
  }
}

#' Crea un elemento de linea compatible con varias versiones de ggplot2.
#'
#' @param ... Argumentos adicionales que se entregan a `element_line`.
#' @param ancho Grosor de la linea.
#'
#' @return Un elemento de tema para lineas.
element_linea_actual = function(..., ancho = 0.5){
  if (packageVersion("ggplot2") >= "3.4.0") {
    element_line(..., linewidth = ancho)
  } else {
    element_line(..., size = ancho)
  }
}

#' Crea un elemento rectangular compatible con varias versiones de ggplot2.
#'
#' @param ... Argumentos adicionales que se entregan a `element_rect`.
#' @param ancho Grosor del borde.
#'
#' @return Un elemento de tema rectangular.
element_rect_actual = function(..., ancho = 0.5){
  if (packageVersion("ggplot2") >= "3.4.0") {
    element_rect(..., linewidth = ancho)
  } else {
    element_rect(..., size = ancho)
  }
}

#' Grafica series de tiempo de un modelo VECM.
#'
#' @param series Objeto con una o varias series de tiempo.
#' @param titulo Titulo de la grafica.
#' @param subtitulo Subtitulo de la grafica. Por defecto no se muestra.
#' @param colores Vector nombrado con los colores de cada serie.
#' @param etiquetas Etiquetas que se quieren mostrar en la leyenda.
#' @param etiqueta_color Titulo de la leyenda de color.
#'
#' @return Una grafica de ggplot2 con las series de tiempo.
graficar_series_vecm = function(series, titulo, subtitulo = NULL,
                                colores = NULL, etiquetas = NULL,
                                etiqueta_color = ""){
  datos_series = as.data.frame(series) %>%
    mutate(tiempo = as.numeric(time(series))) %>%
    pivot_longer(cols = -tiempo, names_to = "variable", values_to = "valor")

  grafica = datos_series %>%
    ggplot(aes(x = tiempo, y = valor, color = variable)) +
    geom_linea_actual(ancho = 0.65) +
    scale_x_continuous(
      minor_breaks = NULL,
      expand = expansion(mult = 0.03)
    ) +
    scale_y_continuous(
      minor_breaks = NULL,
      expand = expansion(mult = 0.05)
    ) +
    theme_light(base_size = 10) +
    labs(title = titulo, subtitle = subtitulo, x = "", y = "",
         color = etiqueta_color) +
    theme(
      plot.title = element_text(size = 11, hjust = 0.5, margin = margin(b = 4)),
      plot.subtitle = element_text(size = 9, hjust = 0.5, color = "grey35"),
      axis.text = element_text(size = 9, color = "grey15"),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.grid.major = element_linea_actual(color = "grey80", ancho = 0.45),
      panel.grid.minor = element_blank(),
      panel.border = element_rect_actual(color = "grey80", fill = NA, ancho = 0.5),
      plot.margin = margin(4, 6, 4, 6)
    ) +
    guides(color = guide_legend(override.aes = list(size = 1.1)))

  if (!is.null(colores) && !is.null(etiquetas)) {
    grafica = grafica + scale_color_manual(values = colores, labels = etiquetas)
  } else if (!is.null(colores)) {
    grafica = grafica + scale_color_manual(values = colores)
  }

  return(grafica)
}

#' Grafica pronosticos de un VECM reparametrizado.
#'
#' @param pronostico Resultado de `predict` aplicado al modelo reparametrizado.
#'
#' @return Una grafica de ggplot2 con pronosticos e intervalos de confianza.
graficar_pronostico_vecm = function(pronostico){
  datos_pronostico = imap_dfr(pronostico$fcst, function(matriz, variable){
    as.data.frame(matriz) %>%
      mutate(
        paso = seq_len(n()),
        variable = variable
      ) %>%
      rename(
        pronostico = fcst,
        inferior = lower,
        superior = upper
      )
  })

  datos_pronostico %>%
    ggplot(aes(x = paso, y = pronostico)) +
    geom_ribbon(aes(ymin = inferior, ymax = superior),
                fill = "grey70", alpha = 0.35) +
    geom_linea_actual(ancho = 0.8, color = "royalblue") +
    facet_wrap(~ variable, scales = "free_y") +
    theme_light() +
    xlab("Pasos adelante") +
    ylab("") +
    ggtitle("Pronostico VECM") +
    theme(plot.title = element_text(size = 11, hjust = 0.5))
}


#' Grafica un fanchart de pronosticos de un VECM reparametrizado.
#'
#' @param datos Series observadas usadas como historia del fanchart.
#' @param pronostico Resultado de `predict` aplicado al modelo reparametrizado.
#' @param variables Variables que se quieren graficar. Si no se entrega, se usan
#'   todas las variables incluidas en el pronostico.
#' @param colores Vector con el color de la linea de pronostico y el color del
#'   intervalo.
#'
#' @return Una grafica de ggplot2 con el fanchart de cada variable.
graficar_fanchart_vecm = function(datos, pronostico, variables = NULL,
                                  colores = c("blue", "lightblue")){
  datos_observados = as.data.frame(datos)

  if (is.null(variables)) {
    variables = names(pronostico$fcst)
  }

  variables_no_disponibles = setdiff(variables, colnames(datos_observados))
  if (length(variables_no_disponibles) > 0) {
    stop("Hay variables que no estan en los datos observados.")
  }

  variables_sin_pronostico = setdiff(variables, names(pronostico$fcst))
  if (length(variables_sin_pronostico) > 0) {
    stop("Hay variables que no estan en el objeto de pronostico.")
  }

  datos_historia = datos_observados %>%
    mutate(indice = seq_len(n()) - 1) %>%
    select(indice, one_of(variables)) %>%
    gather(key = "variable", value = "valor", -indice)

  datos_pronostico = imap_dfr(pronostico$fcst[variables], function(matriz, variable){
    as.data.frame(matriz) %>%
      transmute(
        indice = nrow(datos_observados) + row_number() - 1,
        variable = variable,
        pronostico = fcst,
        inferior = lower,
        superior = upper
      )
  })

  datos_conexion = datos_historia %>%
    group_by(variable) %>%
    filter(indice == max(indice)) %>%
    slice(1) %>%
    ungroup() %>%
    select(variable, x = indice, y = valor) %>%
    left_join(
      datos_pronostico %>%
        group_by(variable) %>%
        filter(indice == min(indice)) %>%
        slice(1) %>%
        ungroup() %>%
        select(variable, xend = indice, yend = pronostico),
      by = "variable"
    )

  ggplot() +
    geom_linea_actual(
      data = datos_historia,
      aes(x = indice, y = valor),
      color = "black",
      ancho = 0.6
    ) +
    geom_ribbon(
      data = datos_pronostico,
      aes(x = indice, ymin = inferior, ymax = superior),
      fill = colores[2],
      alpha = 0.75
    ) +
    geom_linea_actual(
      data = datos_pronostico,
      aes(x = indice, y = pronostico),
      color = colores[1],
      ancho = 0.8
    ) +
    geom_segment(
      data = datos_conexion,
      aes(x = x, y = y, xend = xend, yend = yend),
      color = colores[1],
      size = 0.7
    ) +
    facet_wrap(
      ~ variable,
      ncol = 1,
      scales = "free_y",
      labeller = as_labeller(function(variable) paste("Fanchart para", variable))
    ) +
    theme_light(base_size = 10) +
    xlab("") +
    ylab("") +
    theme(
      strip.text = element_text(size = 11, face = "bold"),
      strip.background = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_rect_actual(color = "black", fill = NA, ancho = 0.6),
      axis.text = element_text(size = 9, color = "black")
    )
}


# 2. Funciones auxiliares para impulso-respuesta ----

#' Extrae datos de una funcion impulso-respuesta.
#'
#' @param IRF Resultado de `irf` aplicado a un modelo reparametrizado.
#' @param impulso Variable que recibe el choque.
#' @param respuesta Variable cuya respuesta se quiere analizar.
#' @param pasos_adelante Horizontes de respuesta.
#'
#' @return Un `data.frame` con horizonte, IRF y limites del intervalo.
extraer_datos_irf = function(IRF, impulso, respuesta, pasos_adelante){
  data.frame(
    pasos_adelante = pasos_adelante,
    irf = as.numeric(IRF$irf[[impulso]][, respuesta]),
    inferior = as.numeric(IRF$Lower[[impulso]][, respuesta]),
    superior = as.numeric(IRF$Upper[[impulso]][, respuesta])
  )
}

#' Grafica una funcion impulso-respuesta.
#'
#' @param IRF_data_frame Tabla con horizonte, IRF y limites del intervalo.
#' @param titulo Titulo de la grafica.
#'
#' @return Una grafica de ggplot2 con la funcion impulso-respuesta.
graficar_datos_irf = function(IRF_data_frame, titulo){
  IRF_data_frame %>%
    ggplot(aes(x = pasos_adelante, y = irf, ymin = inferior,
               ymax = superior)) +
    geom_hline(yintercept = 0, color = "red") +
    geom_ribbon(fill = "grey", alpha = 0.2) +
    geom_linea_actual(ancho = 0.7) +
    theme_light() +
    ggtitle(titulo) +
    ylab("") +
    xlab("Pasos adelante") +
    theme(plot.title = element_text(size = 11, hjust = 0.5),
          axis.title.y = element_text(size = 11))
}

#' Extrae y grafica una funcion impulso-respuesta.
#'
#' @param IRF Resultado de `irf` aplicado a un modelo reparametrizado.
#' @param impulso Variable que recibe el choque.
#' @param respuesta Variable cuya respuesta se quiere analizar.
#' @param pasos_adelante Horizontes de respuesta.
#' @param titulo Titulo de la grafica.
#'
#' @return Una grafica de ggplot2 con la funcion impulso-respuesta.
graficar_irf_extraida = function(IRF, impulso, respuesta, pasos_adelante, titulo){
  extraer_datos_irf(IRF, impulso, respuesta, pasos_adelante) %>%
    graficar_datos_irf(titulo)
}

#' Calcula y grafica una funcion impulso-respuesta.
#'
#' @param var Modelo reparametrizado como VAR en niveles.
#' @param impulso Variable que recibe el choque.
#' @param respuesta Variable cuya respuesta se quiere analizar.
#' @param pasos_adelante Horizontes de respuesta.
#' @param ortog Indica si se usan respuestas ortogonalizadas.
#' @param int_conf Nivel de confianza del intervalo.
#' @param titulo Titulo de la grafica.
#' @param semilla Semilla para reproducir los resultados.
#' @param runs Numero de repeticiones usadas para calcular los intervalos.
#'
#' @return Una grafica de ggplot2 con la funcion impulso-respuesta.
impulso_respuesta = function(var, impulso, respuesta, pasos_adelante = NULL,
                             ortog = TRUE, int_conf = 0.95, titulo = "",
                             semilla = NULL, runs = 100, ...){
  argumentos_extra = list(...)

  if (is.null(pasos_adelante) && "pasos_adelantes" %in% names(argumentos_extra)) {
    pasos_adelante = argumentos_extra$pasos_adelantes
  }

  if (is.null(pasos_adelante)) {
    stop("Debe indicar los horizontes en pasos_adelante.")
  }

  total_pasos_futuros = length(pasos_adelante) - 1
  IRF = irf(var, impulse = impulso, response = respuesta,
            n.ahead = total_pasos_futuros, ortho = ortog, ci = int_conf,
            seed = semilla, runs = runs)

  graph = graficar_irf_extraida(IRF, impulso, respuesta, pasos_adelante, titulo)

  return(graph)
}

#' Grafica una grilla de funciones impulso-respuesta.
#'
#' @param var Modelo reparametrizado como VAR en niveles.
#' @param variables Nombres de las variables del modelo.
#' @param pasos_adelante Horizontes de respuesta.
#' @param ortog Indica si se grafican respuestas ortogonalizadas.
#' @param int_conf Nivel de confianza de los intervalos.
#' @param prefijo_titulo Texto inicial usado en los titulos de las graficas.
#' @param semilla Semilla para reproducir los resultados.
#' @param runs Numero de repeticiones usadas para calcular los intervalos.
#'
#' @return Una lista con el objeto IRF, las combinaciones y las graficas.
graficar_grilla_irf = function(var, variables, pasos_adelante, ortog, int_conf,
                               prefijo_titulo, semilla = NULL, runs = 100){

  total_pasos_futuros = length(pasos_adelante) - 1

  IRF = irf(var, impulse = variables, response = variables,
            n.ahead = total_pasos_futuros, ortho = ortog, ci = int_conf,
            seed = semilla, runs = runs)

  combinaciones = expand.grid(impulso = variables, respuesta = variables,
                              stringsAsFactors = FALSE) %>%
    as_tibble() %>%
    mutate(
      impulso_titulo = gsub("_", "", impulso),
      respuesta_titulo = gsub("_", "", respuesta),
      titulo = paste0(prefijo_titulo, " de ", impulso_titulo,
                      " - respuesta de ", respuesta_titulo)
    )

  graficas = pmap(
    list(combinaciones$impulso, combinaciones$respuesta, combinaciones$titulo),
    function(impulso, respuesta, titulo){
      graficar_irf_extraida(IRF, impulso, respuesta, pasos_adelante, titulo)
    }
  )

  list(
    objeto_irf = IRF,
    combinaciones = combinaciones,
    graficas = graficas
  )
}
