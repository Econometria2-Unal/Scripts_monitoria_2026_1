# Funciones auxiliares para graficacion - Script VAR

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

#' Grafica una serie de tiempo.
#'
#' @param serie Serie que se quiere graficar.
#' @param titulo Titulo de la grafica.
#' @param color Color de la linea.
#' @param cortes_y Cortes del eje vertical. Por defecto ggplot2 los calcula.
#'
#' @return Una grafica de ggplot2 con la serie de tiempo.
graficar_ts = function(serie, titulo, color, cortes_y = waiver()){
  datos_serie = data.frame(
    tiempo = as.numeric(time(serie)),
    valor = as.numeric(serie)
  )
  
  inicio_decada = floor(min(datos_serie$tiempo) / 10) * 10
  fin_decada = floor(max(datos_serie$tiempo) / 10) * 10
  marcas_x = seq(
    from = inicio_decada,
    to = fin_decada,
    by = 10
  )
  
  datos_serie %>% 
    ggplot(aes(x = tiempo, y = valor)) +
    geom_linea_actual(ancho = 0.45, color = color) +
    scale_x_continuous(
      breaks = marcas_x,
      minor_breaks = NULL,
      expand = expansion(mult = 0.05)
    ) +
    scale_y_continuous(
      breaks = cortes_y,
      minor_breaks = NULL,
      expand = expansion(mult = 0.05)
    ) +
    theme_light(base_size = 10) +
    ggtitle(titulo) +
    xlab("") +
    ylab("") +
    theme(
      plot.title = element_text(size = 11, hjust = 0.5, margin = margin(b = 4)),
      axis.text = element_text(size = 9, color = "grey15"),
      panel.grid.major = element_linea_actual(color = "grey80", ancho = 0.45),
      panel.grid.minor = element_blank(),
      panel.border = element_rect_actual(color = "grey80", fill = NA, ancho = 0.5),
      plot.margin = margin(4, 6, 4, 6)
    )
}

#' Grafica una variable del pronostico bootstrap.
#'
#' @param datos_bootstrap Tabla en formato largo con las columnas `tiempo`,
#'   `variable` y `valor`.
#' @param nombre_variable Nombre de la variable que se quiere graficar.
#' @param colores Vector nombrado con los colores asociados a cada variable.
#'
#' @return Una grafica de ggplot2 con el pronostico bootstrap de la variable.
graficar_bootstrap_variable = function(datos_bootstrap, nombre_variable, colores){
  datos_variable = datos_bootstrap %>%
    filter(variable == nombre_variable)
  
  datos_variable %>%
    ggplot(aes(x = tiempo, y = valor)) +
    geom_linea_actual(ancho = 0.8, color = colores[[nombre_variable]]) +
    scale_x_continuous(
      breaks = seq(
        from = floor(min(datos_variable$tiempo) * 2) / 2,
        to = ceiling(max(datos_variable$tiempo) * 2) / 2,
        by = 0.5
      ),
      minor_breaks = NULL,
      expand = expansion(mult = 0.05)
    ) +
    scale_y_continuous(minor_breaks = NULL, expand = expansion(mult = 0.05)) +
    theme_light(base_size = 10) +
    ggtitle(nombre_variable) +
    xlab("") +
    ylab("") +
    theme(
      plot.title = element_text(size = 11, hjust = 0.5, margin = margin(b = 4)),
      axis.text = element_text(size = 9, color = "grey15"),
      panel.grid.major = element_linea_actual(color = "grey80", ancho = 0.45),
      panel.grid.minor = element_blank(),
      panel.border = element_rect_actual(color = "grey80", fill = NA, ancho = 0.5),
      plot.margin = margin(4, 6, 4, 6)
    )
}

#' Grafica pronosticos de un modelo VAR.
#'
#' @param pronostico Resultado de `predict` aplicado a un modelo VAR.
#'
#' @return Una grafica de ggplot2 con pronosticos e intervalos de confianza.
graficar_pronostico_var = function(pronostico){
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
    ggtitle("Pronostico VAR") +
    theme(plot.title = element_text(size = 11, hjust = 0.5))
}


# 2. Funciones auxiliares para graficar errores simulados ----

#' Grafica diagnosticos de los errores simulados.
#'
#' @param u_t Matriz o tabla con los errores simulados.
#' @param errores Nombres de las columnas que contienen los errores.
#' @param cor_u_muestral Matriz de correlaciones de los errores.
#' @param bins Numero de barras usadas en los histogramas.
#'
#' @return Una lista con graficas de series, histogramas, QQ plots,
#' correlaciones y los datos usados.
graficar_diagnostico_errores = function(u_t, errores, cor_u_muestral = cor(u_t),
                                        bins = 25){
  errores_df = as.data.frame(u_t) %>% 
    mutate(periodo = seq_len(nrow(u_t))) %>% 
    pivot_longer(cols = all_of(errores), names_to = "error", values_to = "valor")
  
  resumen_errores_graf = errores_df %>% 
    group_by(error) %>% 
    summarise(media = mean(valor),
              desv = sd(valor),
              minimo = min(valor),
              maximo = max(valor),
              .groups = "drop")
  
  densidad_normal_errores = resumen_errores_graf %>% 
    mutate(valor = map2(minimo, maximo, ~ seq(.x, .y, length.out = 100))) %>% 
    unnest(valor) %>% 
    mutate(densidad = dnorm(valor, mean = media, sd = desv))
  
  g_errores_ts = errores_df %>% 
    ggplot(aes(x = periodo, y = valor, color = error)) +
    geom_linea_actual(ancho = 0.6) +
    facet_wrap(~ error, ncol = 1, scales = "free_y") +
    theme_light() +
    guides(color = "none") +
    xlab("") +
    ylab("") +
    ggtitle("Errores simulados")
  
  g_hist_errores = errores_df %>% 
    ggplot(aes(x = valor)) +
    geom_histogram(aes(y = after_stat(density)), bins = bins,
                   fill = "lightblue", color = "white") +
    geom_linea_actual(data = densidad_normal_errores,
                      aes(x = valor, y = densidad),
                      ancho = 0.7, color = "firebrick4") +
    facet_wrap(~ error, scales = "free") +
    theme_light() +
    xlab("") +
    ylab("Densidad") +
    ggtitle("Distribucion empirica vs. normal")
  
  g_qq_errores = errores_df %>% 
    ggplot(aes(sample = valor)) +
    stat_qq(color = "royalblue", alpha = 0.7) +
    stat_qq_line(color = "firebrick4") +
    facet_wrap(~ error, scales = "free") +
    theme_light() +
    xlab("Cuantiles teoricos") +
    ylab("Cuantiles muestrales") +
    ggtitle("QQ plots de los errores")
  
  cor_errores_df = as.data.frame(as.table(cor_u_muestral)) %>% 
    rename(error_1 = Var1, error_2 = Var2, correlacion = Freq)
  
  g_corr_errores = cor_errores_df %>% 
    ggplot(aes(x = error_1, y = error_2, fill = correlacion)) +
    geom_tile(color = "white") +
    geom_text(aes(label = round(correlacion, 2)), size = 4) +
    scale_fill_gradient2(low = "firebrick4", mid = "white", high = "royalblue",
                         midpoint = 0, limits = c(-1, 1)) +
    coord_fixed() +
    theme_light() +
    xlab("") +
    ylab("") +
    ggtitle("Correlacion entre errores")
  
  list(
    series = g_errores_ts,
    histograma = g_hist_errores,
    qq = g_qq_errores,
    correlacion = g_corr_errores,
    datos = errores_df
  )
}


# 3. Funciones auxiliares para impulso-respuesta ----

#' Extrae datos de una funcion impulso-respuesta.
#'
#' @param IRF Resultado de `irf` aplicado a un modelo VAR.
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
#' @param IRF Resultado de `irf` aplicado a un modelo VAR.
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
#' @param var Modelo VAR estimado.
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
impulso_respuesta = function(var, impulso, respuesta, pasos_adelante, ortog, 
                             int_conf, titulo, semilla = NULL, runs = 100){

  total_pasos_futuros = length(pasos_adelante) - 1
  IRF = irf(var, impulse = impulso, response = respuesta,
            n.ahead = total_pasos_futuros, ortho = ortog, ci = int_conf,
            seed = semilla, runs = runs)
  
  graph = graficar_irf_extraida(IRF, impulso, respuesta, pasos_adelante, titulo)
  
  return(graph)
}

#' Grafica una grilla de funciones impulso-respuesta.
#'
#' @param var Modelo VAR estimado.
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
