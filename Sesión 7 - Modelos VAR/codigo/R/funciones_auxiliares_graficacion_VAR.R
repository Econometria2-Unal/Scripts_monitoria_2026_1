# Funciones auxiliares para graficacion - Script VAR

# 1. Funciones auxiliares generales ----

geom_linea_actual = function(..., ancho = 0.5){
  if (packageVersion("ggplot2") >= "3.4.0") {
    geom_line(..., linewidth = ancho)
  } else {
    geom_line(..., size = ancho)
  }
}

graficar_ts = function(serie, titulo, color){
  data.frame(
    tiempo = as.numeric(time(serie)),
    valor = as.numeric(serie)
  ) %>% 
    ggplot(aes(x = tiempo, y = valor)) +
    geom_linea_actual(ancho = 1, color = color) +
    theme_light() +
    ggtitle(titulo) +
    xlab("") +
    ylab("") +
    theme(plot.title = element_text(size = 11, hjust = 0.5))
}

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

extraer_datos_irf = function(IRF, impulso, respuesta, pasos_adelante){
  data.frame(
    pasos_adelante = pasos_adelante,
    irf = as.numeric(IRF$irf[[impulso]][, respuesta]),
    inferior = as.numeric(IRF$Lower[[impulso]][, respuesta]),
    superior = as.numeric(IRF$Upper[[impulso]][, respuesta])
  )
}

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

graficar_irf_extraida = function(IRF, impulso, respuesta, pasos_adelante, titulo){
  extraer_datos_irf(IRF, impulso, respuesta, pasos_adelante) %>% 
    graficar_datos_irf(titulo)
}

impulso_respuesta = function(var, impulso, respuesta, pasos_adelante, ortog, 
                             int_conf, titulo, semilla = NULL, runs = 100){
  
  "Funcion disenada por German Camilo Rodriguez"
  
  "Calcula las funciones impulso respuesta ortogonalizadas y no ortogonalizadas 
  y devuelve una grafica IRF o OIRF dependiendo la especificacion"
  
  total_pasos_futuros = length(pasos_adelante) - 1
  IRF = irf(var, impulse = impulso, response = respuesta,
            n.ahead = total_pasos_futuros, ortho = ortog, ci = int_conf,
            seed = semilla, runs = runs)
  
  graph = graficar_irf_extraida(IRF, impulso, respuesta, pasos_adelante, titulo)
  
  return(graph)
}

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
