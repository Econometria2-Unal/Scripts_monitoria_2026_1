#' Universidad Nacional de Colombia
#' Facultad de Ciencias Economicas
#'
#' Econometria II | Monitoria
#' Sesion 7: Modelos de vectores autorregresivos
#'
#' Simulacion de una normal multivariada (bivariada) correlacionada
#' usando descomposicion espectral, SVD y Cholesky.

# NOTA MUY IMPORTANTE: Para entender el Script, se sugiere que se lea a la par tanto
#                      el Script, como las notas teóricas en PDF que acompañan el script  

# ===
# Tabla de contenidos ===
# ===

#' Simulación:
  #' 1. Simulacion de una normal estandar bivariada no correlacionada
  #' 2. Simulacion de una normal bivariada correlacionada
    #' 2.1 Generación de la matriz de varianzas-covarianzas de la distribucion normal bivariada correlacionada
    #' 2.2 Simulacion de una normal bivariada correlacionada de manera manual y por mvtnorm
      #' 2.2.1 Simulacion manual: descomposición espectral, SVD y Cholesky
      #' 2.2.2 Simulacion con mvtnorm: descomposición espectral, SVD y Cholesky
#' 
#' Comparacion de los resultados de la simulación manual vs simulación con mvtnorm
#'Graficación de la distribuciones normales multivariadas obtenidas: 
  #'  Graficas 2D interactivas con plotly
  #'  Graficas 3D interactivas de densidad con plotly
  #'  Exportacion de graficas a archivos HTML


# Importación de paquetes ---

# Paquetes necesarios para este script:
# - mvtnorm: simulacion de normales multivariadas.
# - plotly: graficas interactivas.
# - htmlwidgets: exportacion de graficas plotly a archivos HTML.

# Permite simular una distribción normal multivariada mediante
# diferentes tipos de descomposición matricial de la matriz de varianzas y
# covarianzas
library(mvtnorm)

# Permite hacer gráficas 2D y 3D interactivas 
library(plotly)

# Permite exportar archivos HTML desde R
library(htmlwidgets)

# Manejo de rutas relativas ---

# Fijar la ruta del archivo actual como referencia para here().
here::i_am(
  "Sesión 6 - Distribux Normal Multi/codigo/R/simulacion_normal_multivariada.R"
)

# Directorios principales del proyecto.
directorio_proyecto = here::here()
directorio_sesion_normal_multivariada = fs::path(
  "Sesión 6 - Distribux Normal Multi"
)
directorio_codigo_R = fs::path(
  directorio_sesion_normal_multivariada,
  "codigo",
  "R"
)

# Importacion de funciones auxiliares para comparacion, graficacion y exportacion.
ruta_funciones_auxiliares_normal_multivariada = fs::path(
  directorio_proyecto,
  directorio_codigo_R,
  "funciones_auxiliares_distribux_normal_multivariada.R"
)

source(
  ruta_funciones_auxiliares_normal_multivariada,
  encoding = "UTF-8"
)
# Fijamos la semilla para que la simulacion sea reproducible.
set.seed(82901)

# Numero de observaciones
n_observaciones = 100000 # Entre más datos sean simulados, los resultados
                         # computacionales convergerán mejor a los teóricos

# Vector de medias de la distribución normal bivariada correlacionada.
media = c(u_1 = 0, u_2 = 0)


# ===
# Simulación ====
# ===

# ===
# 1. Simulacion de una normal estandar bivariada no correlacionada ----
# ===

# El primer paso para simular una distribución normal bivariada correlacionada
# es simular una distribución normal estandar bivariada no correlacionada

# Note: Tenga en cuenta que simular de una distribución normal p-variada
#       no correlacionada es equivalente a simular p distribuciones normales
#       univariadas independientes. En distribuciones normales multivariadas
#       que no haya correlación entre sus V.A. componentes, implica necesiamente
#       que sus V.A. componentes son independientes. 

# Primero simulamos dos errores normales estandar univariados independientes:
#
#   z_1 ~ N(0, 1)
#   z_2 ~ N(0, 1)
#
# Nota: Al juntarlos en una matriz (que será la que guarde la información de la 
# distribución normal multivariada) Z = (z_1, z_2), cada fila es una observacion de
# una normal estandar bivariada no correlacionada: Z ~ N_2(0, I_2), es decir, 
# las columnas de Z (la distribución normal bivariada) serán los resultados de 
# simular z_1 y z_2, las distribuciones normales univariadas estándar. 

# Simulo las dos distribuciones normales estándar independientes univariadas
normales_estandar = rnorm(n_observaciones * 2)

# Construyo la matriz Z, que será la simulación de la distribución normal
# bivariada estándar no correlacionada
Z = matrix(normales_estandar, ncol = 2, byrow = TRUE)

# Le doy nombres a las columnas de Z
colnames(Z) = c("z_1", "z_2")

cat("\nCorrelacion muestral de los errores estandar no correlacionados:\n")
print(cor(Z)) # Note que la matriz de varianzas y covarianzas de Z es muy cercana
              # a la matriz identidad 2x2

# ===
# 2. Simulacion de una normal bivariada correlacionada ----
# ===

# El segundo paso, es simular la distribución normal bivariada correlacionada a partir de la
# distribución normal bivariada estándar no correlacionada que se simulo en el paso anterior. 

# ===
# 2.1 Generación de la matriz de varianzas-covarianzas de la distribucion normal bivariada correlacionada ----
# ===

# Lo primero que se debe hacer para simular la distribución normal bivariada correlacionada, 
# es especificar la matriz de varianzas-covarianzas (Sigma) de ésta distribución

# Usaremos la misma matriz Sigma para las tres descomposiciones que veremos: 
  # 1. Descomposición Espectral
  # 2. SVD
  # 3. Descomposición de Cholesky

# Las desviaciones estándar de cada una de las V.A. que componen la distribución
# normal bivariada correlacionada son: 
desv_estandar = c(u_1 = 1.0, u_2 = 1.5)

# Correlacion teorica entre las V.A. que conforman la distribución
# normal bivariada correlacionada es: 
rho = 0.90

# Construcción de la matriz de varianzas y covarianzas (Sigma) de la distribución
# normal bivariada correlacionada
Sigma = matrix(
  c(
    desv_estandar["u_1"]^2, # Varianza de u_1
    rho * prod(desv_estandar), # Covarianza entre u_1 y u_2, cov(u_1, u_2) = rho * std(u_1) * std(u_2)
    rho * prod(desv_estandar),
    desv_estandar["u_2"]^2 # Varianza de u_2
  ),
  nrow = 2,
  byrow = TRUE
)
colnames(Sigma) = rownames(Sigma) = names(media)

cat("\nMatriz de varianzas-covarianzas teorica Sigma:\n")
print(Sigma)

cat("\nMatriz de correlaciones teorica asociada a Sigma:\n")
print(cov2cor(Sigma))

# ===
# 2.2 Simulacion de una normal bivariada correlacionada de manera manual y por mvtnorm ----
# ===

# ===
# 2.2.1 Simulacion manual: Para descomposición espectral, SVD y Cholesky ----
# ===

# Idea de la simulacion manual:
#
# Si Z ~ N_2(0, I_2) y encontramos una matriz P tal que:
#
#   P %*% P^{'} = Sigma, # Todas las descomposiciones de Sigma, consisten 
#                       en encontrar la matriz P, que permita hacer dicha descomposición
#
# entonces:
#
#   U = Z %*% P^{'} + media # Nota: Se multiplica P^{'} porque los vectores son filas
#                                   En las notas del PDF, se multiplica por P, porque los vectores son columnas
#
# tiene distribucion normal bivariada con matriz de covarianzas Sigma.
#
# La funcion siguiente replica la matrices de descomposición P usados internamente por
# mvtnorm::rmvnorm() para los metodos "eigen", "svd" y "chol". Asi la
# comparacion manual vs. mvtnorm se puede hacer observacion por observacion.

# 1. Se encuentra la matriz P, que se obtiene la descomposición de Sigma

# Nota: Para entender estas función, leer las notas sobre los diferentes tipos de 
#       descomposición que hay de la matriz Sigma

descomposicion_sigma = function(Sigma, metodo) {
  # Para realizar la Descomposición Espectral de Sigma
  if (metodo == "espectral") {
    
    # Encuentra los valores y vectores propios de Sigma
    ev = eigen(Sigma, symmetric = TRUE)
    
    # La matriz P = Q * Lambda^{1/2}
    matriz_P_trans = t(ev$vectors %*% (t(ev$vectors) * sqrt(pmax(ev$values, 0))))
  
  # Para realizar la SVD
  } else if (metodo == "svd") {
    
    # Encuentra los valores y vectores singulares a izquierda y derecha de Sigma
    sv = svd(Sigma)
  
    # La matriz P = U * D^{1/2}
    matriz_P_trans = t(sv$v %*% (t(sv$u) * sqrt(pmax(sv$d, 0))))
  
  # Para realizar la Descomposición de Cholesky     
  } else if (metodo == "cholesky") {
    
    # Realiza la Descomposición de Cholesky de Sigma 
    cholesky = chol(Sigma, pivot = TRUE)
    
    # La matriz P = L, donde L es la matriz triangular superior de Cholesky
    matriz_P_trans = cholesky[, order(attr(cholesky, "pivot"))]
    
  }
  
  # La función retorna la matriz P^{'}
  return(matriz_P_trans)
}

# 2. Se simula la distribución normal bivariada correlacionada (U) 
#    A partir de la distribución normal bivariada estándar no correlacionada (Z)
#    y la matriz de descomposición (matriz_P)

simulacion_manual_distribucion_normal_multiv_correlac = function(Z, media, Sigma, metodo) {
  
  # Nota: U será la matriz que contiene los resultados de simular la distribución normal
  #       bivariada correlacionada
  
  # Matriz que se obtiene de la descomposición de la matriz Sigma
  matriz_P_trans = descomposicion_sigma(Sigma, metodo)
  
  # Se simula la distribución normal bivariada correlacionada (U), usando la distribución
  # normal bivariada estándar no correlacionada (Z) y la matriz de descomposición (matriz_P)
  U = Z %*% matriz_P_trans
  
  # Sumarle la media a U
  U = sweep(U, 2, media, "+")
  
  # Se ponen los nombres de las V.A. que conforman la distribución normal bivariada correlacionada
  colnames(U) = names(media)
  
  # La función retorna la matriz U asociada con la distribución normal bivariada 
  # correlacionada
  return(U)
}

# Simulación manual de la distribución normal multivariada correlacionada usando: 
  
# 1. Descomposición Espectral (Sigma = Q * Lambda * Q^{'})
U_manual_espectral = simulacion_manual_distribucion_normal_multiv_correlac(Z, 
                                                                           media, 
                                                                           Sigma, 
                                                                           "espectral")

# 2. SVD (Sigma = U * D * V^{'})
U_manual_svd = simulacion_manual_distribucion_normal_multiv_correlac(Z, 
                                                                     media, 
                                                                     Sigma, 
                                                                     "svd")

# 3. Descomposición de Cholesky (Sigma = L * L^{'})
U_manual_cholesky = simulacion_manual_distribucion_normal_multiv_correlac(Z, 
                                                                          media, 
                                                                          Sigma, 
                                                                          "cholesky")


# ===
# 2.2.2 Simulacion con mvtnorm: descomposición espectral, SVD y Cholesky ----
# ===

# Nota: Para simular, rmvnorm() normalmente genera sus propias normales estandar. 
# Pero, para efectos de comparación, con la simulacion manual, aqui le entregamos a rmvnorm() 
# el mismo vector de normales estandar que se simulo manualmente para construir 
# la matriz Z.

normales_estandar_fijas = as.vector(t(Z)) # Se vectoriza la matriz Z asociada a la
                                          # distribucion normal bivariada estandar
                                          # que se obtuvo en el metodo manual.

generador_desde_Z = function(n) {
  return(normales_estandar_fijas)
}

# Simulación usando el paquete mvtnorm de la distribución normal multivariada 
# correlacionada usando: 

# 1. Descomposición Espectral (Sigma = Q * Lambda * Q^{'})
U_mvtnorm_espectral = mvtnorm::rmvnorm(
  n = n_observaciones,
  mean = media,
  sigma = Sigma,
  method = "eigen",
  pre0.9_9994 = FALSE,
  rnorm = generador_desde_Z
)

# 2. SVD (Sigma = U * D * V^{'})
U_mvtnorm_svd = mvtnorm::rmvnorm(
  n = n_observaciones,
  mean = media,
  sigma = Sigma,
  method = "svd",
  pre0.9_9994 = FALSE,
  rnorm = generador_desde_Z
)

# 3. Descomposición de Cholesky (Sigma = L * L^{'})
U_mvtnorm_cholesky = mvtnorm::rmvnorm(
  n = n_observaciones,
  mean = media,
  sigma = Sigma,
  method = "chol",
  pre0.9_9994 = FALSE,
  rnorm = generador_desde_Z
)


# ===
# Comparacion de los resultados de la simulación manual vs simulación con mvtnorm ----
# ===

# Nota: En esta seccion se verifica que la simulacion manual y mvtnorm producen
# exactamente los mismos datos cuando ambas simulaciones emplean mismas distribuciones
# normales estandar. Este es el chequeo computacional mas importante del
# script: muestra que las diferences descomposiciones de la matriz Sigma
# realizadas manualmente generan la misma distribución normal multivariada correlacionada
# que genera la funcion especializada mvtnorm::rmvnorm().

# Tabla de comparación entre las diferentes distribuciones multivariadas correlacionadas
# generadas
tabla_comparacion = crear_tabla_comparacion_mvtnorm(
  U_manual_espectral = U_manual_espectral, 
  U_mvtnorm_espectral = U_mvtnorm_espectral,
  U_manual_svd = U_manual_svd,
  U_mvtnorm_svd = U_mvtnorm_svd,
  U_manual_cholesky = U_manual_cholesky,
  U_mvtnorm_cholesky = U_mvtnorm_cholesky
)

# Imprime un resumen comparativo de los resultados de las diferentes simulaciones 
imprimir_resumen_comparacion_mvtnorm(
  tabla_comparacion = tabla_comparacion,
  U_manual_espectral = U_manual_espectral,
  Sigma = Sigma
)

# ===
# Graficación de la distribuciones normales multivariadas obtenidas ----
# ===

# ===
# Graficas 2D interactivas de plotly ----
# ===

# Nota: Las gráfica 2D muestra la nube de puntos de las diferentes observaciones
# o datos simuladoas a partir de las diferentes distribuciones normales multivariadas

# Las graficas 2D permiten ver la geometria de las distribuciones normales multivariadas
# Simuladas. La distribución normal estandar bivariada no correlacionada debe verse 
# proyectada en los plano casi circular; la normal multivariada correlacionada proyecta
# en el plano debe verse como una nube inclinada y alargada. Esa inclinacion
# se debe a la correlación positiva de 0.9 entre u_1 y u_2.
n_puntos_grafica = 8000
semilla_graficos = 202601

# Imprime una descripción gráfica de los gráficos 2D simulados
imprimir_descripcion_graficas_2d(
  n_observaciones = n_observaciones,
  n_puntos_grafica = n_puntos_grafica
)

# Se generan las gráficas 2D de las distribuciones normal multivariada
graficas_2d = crear_graficas_2d_normal_multivariada(
  Z = Z,
  U_manual_espectral = U_manual_espectral,
  U_manual_svd = U_manual_svd,
  U_manual_cholesky = U_manual_cholesky,
  n_puntos = n_puntos_grafica,
  semilla_graficos = semilla_graficos
)

# Las gráficas 2D generadas para los diferentes tipos de distribución normal multivariadas simuladas: 

g_normal_estandar = graficas_2d$normal_estandar; g_normal_estandar # Normal multivariada estándar no correlacionada
g_espectral = graficas_2d$espectral;  g_espectral # Normal multivariada correlacionada (usando desc. espectral)
g_svd = graficas_2d$svd; g_svd # Normal multivariada correlacionada (usando SVD)
g_cholesky = graficas_2d$cholesky; g_cholesky # Normal multivariada correlacionada (usando Chokesky)


# ===
# Graficas 3D interactivas de plotly ----
# ===

# Nota: Las gráfica 3D muestran las funciones de densidad asociadas a 
#       las diferentes distribuciones normales multivariadas simuladas


# Las graficas 3D no muestran puntos simulados: muestran la funcion de densidad
# teorica evaluada sobre una grilla. Es otra forma de visualizar y entender la
# distribución normal multivariada simulada.

# Imprime una descripción gráfica de los gráficos 3D simulados
imprimir_descripcion_graficas_3d()

# Se generan las gráficas 3D de las distribuciones normal multivariada
graficas_3d = crear_graficas_3d_normal_multivariada(
  media = media,
  Sigma = Sigma
)

# Las gráficas 3D generadas para los diferentes tipos de distribución normal multivariadas simuladas: 

g_normal_estandar_3d = graficas_3d$normal_estandar_3d; g_normal_estandar_3d # Normal multivariada estándar no correlacionada
g_espectral_3d = graficas_3d$espectral_3d; g_espectral_3d # Normal multivariada correlacionada (usando desc. espectral)
g_svd_3d = graficas_3d$svd_3d; g_svd_3d # Normal multivariada correlacionada (usando SVD)
g_cholesky_3d = graficas_3d$cholesky_3d; g_cholesky_3d # Normal multivariada correlacionada (usando Chokesky)


# ===
# Exportacion de graficas a archivos HTML ----
# ===

# Nota: Recomendación, ir a la carpeta "html_nm" y
#       abrir los archivos HTML de las gráficas 2D y 3D de las distribuciones 
#       normal multivariadas para visualizarlas desde el navegador. Esas gráficas 
#       son interactivas, y permite visualizar mejor los conceptos

# Finalmente, se exportan las graficas a HTML. Así, las figuras
# pueden abrirse en cualquier navegador y compartirse sin depender de que la
# sesion de R siga abierta.

# Se define el directorio donde se almacenaran las gráficas HTML
directorio_graficas = definir_directorio_graficas(nombre_directorio = "html_nm")

# Imprime una descripción de la exportación de los HTML
imprimir_descripcion_exportacion_html(
  directorio_graficas = directorio_graficas
)

# Se genera una "named list" que contiene las gráficas que se van a exportar 
# a HTML
graficas_html = list(
  "01_std_2d.html" = g_normal_estandar,
  "02_eig_2d.html" = g_espectral,
  "03_svd_2d.html" = g_svd,
  "04_chol_2d.html" = g_cholesky,
  "05_std_3d.html" = g_normal_estandar_3d,
  "06_eig_3d.html" = g_espectral_3d,
  "07_svd_3d.html" = g_svd_3d,
  "08_chol_3d.html" = g_cholesky_3d
)

# Se exportan las gráficas en formato HTML para poder abrir desde el navegador
exportar_graficas_html(
  graficas = graficas_html,
  directorio_graficas = directorio_graficas
)

# Si se ejecuta el script interactivamente, tambien se muestran las graficas
# en el visor de RStudio. Al usar Rscript, se guardan en HTML sin abrir ventanas.
mostrar_graficas_interactivo(graficas_html)
