#' Universidad Nacional de Colombia
#' Facultad de Ciencias Económicas
#'
#' Econometría II | Monitoría
#' Sesión 7: Modelos de vectores autorregresivos - Series Simuladas 
#'
#' Semestre: 2026-1


# ===
# Tabla de contenidos ===
# ===

#' 1. Simulación de un proceso VAR(1) con 3 variables
#' 2. Metodologia Box-Jenkins para series multivariadas
#'  2.1. Identificación
#'  2.2. Estimación
#'  2.3. Validación de supuestos
#'  2.4. Pronóstico y funciones Impulso respuesta 


# Nota: Tips prácticos en R
## Para limpiar el entorno de trabajo se puede correr el comando: " rm(list = ls()) "
## Para cerrar todas las gráficas actualmente abiertas se puede correr el comando: " dev.off() "
## Para resetear R se puede usar el comando: 


# Importación de paquetes ----

# Paquetes del tidyverse (Para el manejo, manipulación y graficación de datos)
library(tidyverse)

# Paquete para trabajar modelos VAR y VECM en R 
library(vars)

# Paquetes adicionales para trabajar con series de tiempo en R
library(urca) # Para realizar tests de raíz unitaria y de Johansen

# Paquetes de graficación
library(gridExtra) # Para concatenar gráficas en un solo plot

# Paquete para leer archivos .xlsx (archivos Excel)
library(readxl)        

# Paquetes para simular de una distribución normal multivariada
library(MASS) 
library(mvtnorm)

# Nota: Cualquiera de los paquetes MASS y mvtnorm sirve para simular de una
#       distribución normal multivariada. No obstante, "mvtnorm" se especializa
#       en distribución normal multivariada; el paquete "MASS" trabaja más temas
#       de estadística en general. 


# ===
# 1. Simulación de un proceso VAR(1) con 3 variables ====
# ===

# Fijamos la semilla para que siempre dé el mismo resultado
semilla_simulacion = 82901
set.seed(semilla_simulacion) 

# Determinamos un tamaño de muestra de 500 observaciones
T = 5000 # Nota: Entre más muestra, mejor será observar la convergencia
        #       de los resultados simulados a los teóricos

variables = c("y_1", "y_2", "y_3")
errores = c("u_1", "u_2", "u_3")

# Se va a simular un modelo VAR_1 cuya ecuación está dada por: 
## Y_t = A_0 + A_1 Y_{t-1} + u_t

# Nota: Y_t es una matriz de 3 variables (una variable por columna). 
#       u_t también es una matriz de 3 variables, donde cada columna de la 
#       matriz es el error asociado a cada variable de la matriz Y_t.

# Y_t Se crea como una matriz de 0s. Luego se llena con valores reales, cuando
#     ocurre la simulación
Y_t <- matrix(0, nrow = T, ncol = length(variables),
              dimnames = list(NULL, variables)) 

# En esta simulación se quiere que los errores estén correlacionados y que,
# además, tengan desviaciones estándar diferentes. Para ello simulamos:
#
#   u_t ~ N_3(0, Sigma_u)
#
# La matriz P_chol es triangular inferior. Esto permite construir una matriz de
# varianzas y covarianzas:
#
#   Sigma_u = P_chol P_chol'
#
# Esta construcción es coherente con una identificación recursiva tipo Cholesky:
# y_1 es contemporáneamente más exógena que y_2 y y_3, mientras que y_2 es más
# exógena que y_3. Los errores reducidos u_t estarán correlacionados, pero los
# choques estructurales eps_t que los generan son ortogonales.
#
# Importante: la correlación entre errores no garantiza por sí sola un orden de
# exogeneidad. El orden se impone mediante la estructura triangular de P_chol
# para la relación contemporánea. La matriz A_1, definida más abajo, gobierna
# los efectos rezagados y no necesita ser triangular para usar Cholesky.
P_chol = matrix(c(0.70, 0.00, 0.00,
                  0.35, 1.10, 0.00,
                  0.25, 0.55, 1.60),
                nrow = 3, byrow = TRUE,
                dimnames = list(errores, c("eps_1", "eps_2", "eps_3")))

Sigma_u_teorica = P_chol %*% t(P_chol)
Sigma_u_teorica

cor_u_teorica = cov2cor(Sigma_u_teorica)
cor_u_teorica

desv_u_teoricas = sqrt(diag(Sigma_u_teorica))
desv_u_teoricas

# Forma manual equivalente:
# 1. Simular choques estructurales normales estándar e independientes.
# 2. Transformarlos con P_chol para obtener errores reducidos normales
#    multivariados, correlacionados y con diferentes desviaciones estándar.
set.seed(semilla_simulacion)
eps_t_manual = matrix(rnorm(T * length(errores)), nrow = T, ncol = length(errores),
                      dimnames = list(NULL, c("eps_1", "eps_2", "eps_3")))
u_t_manual = eps_t_manual %*% t(P_chol)
colnames(u_t_manual) = errores

# Simulación automática desde una normal multivariada.
# MASS::mvrnorm() y mvtnorm::rmvnorm() son funciones estándar en R para simular:
#
#   u_t ~ N_3(0, Sigma_u)
#
# En este script distinguimos los errores simulados con cada función para poder
# compararlos. Para la simulación principal del VAR usamos u_t_mvtnorm, porque
# permite indicar explícitamente method = "chol".
media_u = rep(0, length(errores))
names(media_u) = errores

set.seed(semilla_simulacion)
u_t_mass = MASS::mvrnorm(n = T, mu = media_u, Sigma = Sigma_u_teorica)
colnames(u_t_mass) = errores

set.seed(semilla_simulacion)
u_t_mvtnorm = mvtnorm::rmvnorm(n = T, mean = media_u, sigma = Sigma_u_teorica,
                               method = "chol")
colnames(u_t_mvtnorm) = errores

# Errores usados en la simulación del VAR.
u_t = u_t_mvtnorm

# Verificación: la simulación manual, MASS::mvrnorm() y mvtnorm::rmvnorm()
# representan la misma distribución objetivo. No se espera que sean idénticas
# observación por observación, porque cada método puede usar una raíz matricial
# distinta de Sigma_u. La equivalencia importante es:
#
#   Var(u_t_manual) = P_chol P_chol' = Sigma_u
#   Var(u_t_mass)    = Sigma_u
#   Var(u_t_mvtnorm) = Sigma_u
covarianza_manual_teorica = P_chol %*% t(P_chol)
misma_covarianza_teorica = all.equal(covarianza_manual_teorica, Sigma_u_teorica)
misma_covarianza_teorica

son_mismas_observaciones_mass_mvtnorm = all.equal(u_t_mass, u_t_mvtnorm)
son_mismas_observaciones_mass_mvtnorm

comparacion_simulacion_manual_mass_mvtnorm = list(
  medias_manual = colMeans(u_t_manual),
  medias_mass = colMeans(u_t_mass),
  medias_mvtnorm = colMeans(u_t_mvtnorm),
  covarianza_manual = cov(u_t_manual),
  covarianza_mass = cov(u_t_mass),
  covarianza_mvtnorm = cov(u_t_mvtnorm),
  covarianza_teorica = Sigma_u_teorica,
  max_dif_cov_manual_vs_teorica = max(abs(cov(u_t_manual) - Sigma_u_teorica)),
  max_dif_cov_mass_vs_teorica = max(abs(cov(u_t_mass) - Sigma_u_teorica)),
  max_dif_cov_mvtnorm_vs_teorica = max(abs(cov(u_t_mvtnorm) - Sigma_u_teorica))
)
comparacion_simulacion_manual_mass_mvtnorm

resumen_comparacion_errores = bind_rows(
  data.frame(metodo = "manual", error = errores,
             media = colMeans(u_t_manual),
             desviacion_estandar = apply(u_t_manual, 2, sd)),
  data.frame(metodo = "MASS::mvrnorm", error = errores,
             media = colMeans(u_t_mass),
             desviacion_estandar = apply(u_t_mass, 2, sd)),
  data.frame(metodo = "mvtnorm::rmvnorm", error = errores,
             media = colMeans(u_t_mvtnorm),
             desviacion_estandar = apply(u_t_mvtnorm, 2, sd))
)
resumen_comparacion_errores

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Análisis descriptivo de u_t   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

resumen_errores = data.frame(
  error = errores,
  media = colMeans(u_t),
  desviacion_estandar = apply(u_t, 2, sd),
  varianza = apply(u_t, 2, var)
)
resumen_errores

# Matriz muestral de varianzas y covarianzas de los errores simulados.
Sigma_u_muestral = cov(u_t)
Sigma_u_muestral

# Matriz muestral de correlaciones de los errores simulados.
cor_u_muestral = cor(u_t)
cor_u_muestral

# Funciones auxiliares para graficar con ggplot2 sin depender de ggfortify
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
    ggtitle("Pronóstico VAR") +
    theme(plot.title = element_text(size = 11, hjust = 0.5))
}

errores_df = as.data.frame(u_t) %>% 
  mutate(periodo = seq_len(T)) %>% 
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
  geom_histogram(aes(y = after_stat(density)), bins = 25,
                 fill = "lightblue", color = "white") +
  geom_linea_actual(data = densidad_normal_errores,
                    aes(x = valor, y = densidad),
                    ancho = 0.7, color = "firebrick4") +
  facet_wrap(~ error, scales = "free") +
  theme_light() +
  xlab("") +
  ylab("Densidad") +
  ggtitle("Distribución empírica vs. normal")

g_qq_errores = errores_df %>% 
  ggplot(aes(sample = valor)) +
  stat_qq(color = "royalblue", alpha = 0.7) +
  stat_qq_line(color = "firebrick4") +
  facet_wrap(~ error, scales = "free") +
  theme_light() +
  xlab("Cuantiles teóricos") +
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
  ggtitle("Correlación entre errores")

x11();grid.arrange(g_errores_ts, g_hist_errores, ncol = 2)
x11();grid.arrange(g_qq_errores, g_corr_errores, ncol = 2)

# Definimos el vector constante A_0
A_0 = c(0.5, 0.2, -0.1) 

# Definimos la matriz de coeficientes autorregresivos.
A_1 = matrix(c(0.35, 0.08, 0.04,
               0.25, 0.30, 0.06,
               0.15, 0.20, 0.25),
             nrow = 3, byrow = TRUE,
             dimnames = list(variables, paste0("L1.", variables))) # Matriz 3x3

A_0
A_1

# La matriz A_1 no es triangular inferior. Por tanto, la simulación permite
# efectos rezagados cruzados entre las tres variables. Esto separa claramente
# la dinámica del VAR de la identificación contemporánea de Cholesky: el orden
# recursivo y_1, y_2, y_3 se mantiene por el orden de las columnas de Y_t y por
# la estructura triangular de P_chol, no porque A_1 sea triangular.

# Nota: La idea de la simulación 

# Función que permite simular un VAR(1) 

sim_VAR1 = function(Y_t, A_0, A_1, u_t, T){
  for (i in 2:T) {
    # Se usa la fórmula de un VAR(1) para llenar cada una de las filas de Y_t
    Y_t[i,] = as.numeric(A_0 + A_1 %*% Y_t[i-1,] + u_t[i,]) # Y_t = A_0 + A_1 Y_{t-1} + u_t
  }  
  return(Y_t)
}

# Nota: La simulación lo que busca es modelar las variables a partir de la 
#       fórmula de un VAR(1) en forma reducida: Y_t = A_0 + A_1 Y_{t-1} + u_t

# Nota: La función sim_VAR1 lo que busca es llenar mediante un ciclo, cada una 
#       de las filas (iteración por iteración) de Y_t. La matriz  
Y_t = sim_VAR1(Y_t, A_0, A_1, u_t, T) # 
                            # de ceros Y_t con valores

# Convertimos la serie en un objeto ts

Y_t = ts(Y_t, start=c(1900,1), frequency=4)


#~~~~~~~~~~~~~~~~~~~~~~#
# Gráficas de la serie #
#~~~~~~~~~~~~~~~~~~~~~~#

y1 = graficar_ts(Y_t[,"y_1"], titulo = "Variable y_1", color = "lightblue")
y2 = graficar_ts(Y_t[,"y_2"], titulo = "Variable y_2", color = "royalblue")
y3 = graficar_ts(Y_t[,"y_3"], titulo = "Variable y_3", color = "darkorange")

x11();grid.arrange(y1,y2,y3,ncol=3)

# Recuerden que los modelos VAR requieren de series estacionarias. 

adf1= ur.df(Y_t[,"y_1"], lags=3,selectlags = "AIC",type="none")
summary(adf1) # Rechazo H0, la serie es I(0)

adf2= ur.df(Y_t[,"y_2"], lags=3,selectlags = "AIC",type="none")
summary(adf2) # Rechazo H0, la serie es I(0)

adf3= ur.df(Y_t[,"y_3"], lags=3,selectlags = "AIC",type="none")
summary(adf3) # Rechazo H0, la serie es I(0)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### 2. Metodología Box-Jenkins para series multivariadas ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###### 2.1. Identificación ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Ya tenemos nuestra serie multivariada, veamos que rezago nos recomienda la 
# función VARselect() 

# Selección de rezagos para un VAR con tendencia e intercepto.
VARselect(Y_t, lag.max=6,type = "both", season = NULL)

# Selección de rezagos para un VAR con sólo intercepto.
VARselect(Y_t, lag.max=6,type = "const", season = NULL)

# Selección de rezagos para un VAR sin términos determinísticos.
VARselect(Y_t, lag.max=6,type = "none", season = NULL)

# Como el proceso generador de datos es un VAR(1), esperamos que los criterios
# de información favorezcan rezagos bajos, especialmente p = 1.

#~~~~~~~~~~~~~~~~~~~~~~~~~#
###### 2.2. Estimación ####
#~~~~~~~~~~~~~~~~~~~~~~~~~#

# Para seleccionar el VAR(1), verificamos si tiene intercepto y deriva

# VAR con tendencia e intercepto
V.tr = VAR(Y_t, p=1, type="both", season=NULL)
summary(V.tr) # La tendencia no es significativa, analizamos const

# VAR con intercepto.
V.dr= VAR(Y_t, p=1, type="const", season=NULL) 
summary(V.dr) # El intercepto es significativo en una ecuación, veamos none

# VAR sin términos determinísticos.
V.no = VAR(Y_t, p=1, type="none", season=NULL)  
summary(V.no)

# Elegimos el modelo con constante, pues se ha visto que tienen constante signi-
# ficativa.

# Raíces del proceso. Deben ser menores a |1| para que sea estacionario. 

roots(V.dr) #El proceso es estable.

# Ahora analizamos cada uno de los coeficientes estimados. 

#Coeficientes:

Bcoef(V.dr) # Podemos ver todos los coeficientes con Bcoef


A_1 # Matriz teórica usada en la simulación.
Acoef(V.dr) # Las estimaciones deberían ser cercanas a A_1.

# Matriz de varianzas y covarianzas de los residuales

Sigma.est = summary(V.dr)$covres
Sigma.est 

# Análisis en conjunto
summary(V.dr)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###### 2.3. Validación de supuestos ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~ No autocorrelación serial ~~#

# PT.asymptotic es para muestra grande y "PT.adjusted" para muestra pequeña.
P.75=serial.test(V.dr, lags.pt = 75, type = "PT.asymptotic");P.75 # No rechazo
P.30=serial.test(V.dr, lags.pt = 30, type = "PT.asymptotic");P.30 # No rechazo
P.20=serial.test(V.dr, lags.pt = 20, type = "PT.asymptotic");P.20 # No rechazo
P.10=serial.test(V.dr, lags.pt = 10, type = "PT.asymptotic");P.10 # No rechazo


# Graficamos los residuales para 20 pasos_adelantes: se grafican los residuales, 
# su distribución, la ACF y PACF de los residuales y a ACF y PACF de los 
# residuales al cuadrado (proxy para heterocedasticidad)

x11()
plot(P.20, names = "y_1") # Residuales de la primera serie
plot(P.20, names = "y_2") # Residuales de la segunda serie
plot(P.20, names = "y_3") # Residuales de la tercera serie

#~~ Homocedasticidad ~~#

# Test tipo ARCH multivariado
arch.test(V.dr, lags.multi = 24, multivariate.only = TRUE) # No rechazo
arch.test(V.dr, lags.multi = 12, multivariate.only = TRUE) # No rechazo

#~~ Normalidad ~~#

# Jarque-Bera para series multivariadas.  
normality.test(V.dr) #No rechazo, se cumple el supuesto.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###### 2.4. Pronóstico y funciones Impulso respuesta ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~#
# Pronóstico #
#~~~~~~~~~~~~#

x11()
pronostico_var = predict(V.dr, n.ahead = 12,ci=0.95) 
pronostico_var
graficar_pronostico_var(pronostico_var) 

# Versión Fanchart
fanchart(predict(V.dr, n.ahead = 12), colors = c("blue","lightblue"))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Funciones de impulso respuesta #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Definimos el número pasos adelante

pasos_adelante = 0:18

# Función que me permite calcular y graficar las funciones impulso respuesta 

# (A cada función impulso respuesta le asigno una gráfica)

impulso_respuesta = function(var, impulso, respuesta, pasos_adelante, ortog, 
                             int_conf, titulo){
  
  "Función diseñada por German Camilo Rodriguez" 
  
 "Calcula las funciones impulso respuesta ortogonalizadas y no ortogonalizadas 
  y devuelve una grafíca IRF o OIRF dependiendo la especificación"
  
  # Cáclulo de la función impulso respuesta
  total_pasos_futuros = length(pasos_adelante) - 1
  IRF = irf(var, impulse=impulso, response=respuesta, n.ahead = total_pasos_futuros, 
            ortho=ortog, ci = int_conf)
  IRF_data_frame = data.frame(
    pasos_adelante = pasos_adelante,
    irf = as.numeric(IRF$irf[[impulso]][, respuesta]),
    inferior = as.numeric(IRF$Lower[[impulso]][, respuesta]),
    superior = as.numeric(IRF$Upper[[impulso]][, respuesta])
  )
  # Gráfica de la función impulso respuesta
  graph = IRF_data_frame %>% 
    ggplot(aes(x = pasos_adelante, y = irf, ymin = inferior, 
               ymax = superior)) +
    geom_hline(yintercept = 0, color="red") +
    geom_ribbon(fill="grey", alpha=0.2) +
    geom_linea_actual(ancho = 0.7) +
    theme_light() +
    ggtitle(titulo)+
    ylab("")+
    xlab("Pasos adelante") +
    theme(plot.title = element_text(size = 11, hjust=0.5),
          axis.title.y = element_text(size=11))    
  return(graph)
}

# IRF de las variables del sistema ante distintos choques exógenos.

y1_y1 = impulso_respuesta(V.dr, "y_1", "y_1", pasos_adelante, ortog = FALSE,
                          int_conf = 0.95, titulo = "Impulso de y1 - respuesta de y1")
y1_y2 = impulso_respuesta(V.dr, "y_1", "y_2", pasos_adelante, ortog = FALSE,
                          int_conf = 0.95, titulo = "Impulso de y1 - respuesta de y2")
y1_y3 = impulso_respuesta(V.dr, "y_1", "y_3", pasos_adelante, ortog = FALSE,
                          int_conf = 0.95, titulo = "Impulso de y1 - respuesta de y3")
y2_y1 = impulso_respuesta(V.dr, "y_2", "y_1", pasos_adelante, ortog = FALSE,
                          int_conf = 0.95, titulo = "Impulso de y2 - respuesta de y1")
y2_y2 = impulso_respuesta(V.dr, "y_2", "y_2", pasos_adelante, ortog = FALSE, 
                          int_conf = 0.95, titulo = "Impulso de y2 - respuesta de y2")
y2_y3 = impulso_respuesta(V.dr, "y_2", "y_3", pasos_adelante, ortog = FALSE, 
                          int_conf = 0.95, titulo = "Impulso de y2 - respuesta de y3")
y3_y1 = impulso_respuesta(V.dr, "y_3", "y_1", pasos_adelante, ortog = FALSE,
                          int_conf = 0.95, titulo = "Impulso de y3 - respuesta de y1")
y3_y2 = impulso_respuesta(V.dr, "y_3", "y_2", pasos_adelante, ortog = FALSE, 
                          int_conf = 0.95, titulo = "Impulso de y3 - respuesta de y2")
y3_y3 = impulso_respuesta(V.dr, "y_3", "y_3", pasos_adelante, ortog = FALSE, 
                          int_conf = 0.95, titulo = "Impulso de y3 - respuesta de y3")

x11()
grid.arrange(y1_y1,y1_y2,y1_y3,
             y2_y1,y2_y2,y2_y3,
             y3_y1,y3_y2,y3_y3,ncol=3)

# IRF Ortogonalizadas. 
#
# Cuando ortog = TRUE, la función irf() usa la descomposición de Cholesky de la
# matriz de varianzas y covarianzas de los residuales. En este script el orden
# de las variables es y_1, y_2, y_3; por tanto, la identificación recursiva
# interpreta a y_1 como la variable contemporáneamente más exógena, luego y_2 y
# finalmente y_3. Esta es una restricción de identificación: los errores
# reducidos pueden estar correlacionados, pero los choques ortogonalizados son
# los que se interpretan como innovaciones estructurales recursivas.

y1_y1_ortog = impulso_respuesta(V.dr, "y_1", "y_1", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y1 - respuesta de y1")
y1_y2_ortog = impulso_respuesta(V.dr, "y_1", "y_2", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y1 - respuesta de y2")
y1_y3_ortog = impulso_respuesta(V.dr, "y_1", "y_3", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y1 - respuesta de y3")
y2_y1_ortog = impulso_respuesta(V.dr, "y_2", "y_1", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y2 - respuesta de y1")
y2_y2_ortog = impulso_respuesta(V.dr, "y_2", "y_2", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y2 - respuesta de y2")
y2_y3_ortog = impulso_respuesta(V.dr, "y_2", "y_3", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y2 - respuesta de y3")
y3_y1_ortog = impulso_respuesta(V.dr, "y_3", "y_1", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y3 - respuesta de y1")
y3_y2_ortog = impulso_respuesta(V.dr, "y_3", "y_2", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y3 - respuesta de y2")
y3_y3_ortog = impulso_respuesta(V.dr, "y_3", "y_3", pasos_adelante, ortog = TRUE,
                                int_conf = 0.95, titulo = "Impulso ortogonal de y3 - respuesta de y3")

x11()
grid.arrange(y1_y1_ortog,y1_y2_ortog,y1_y3_ortog,
             y2_y1_ortog,y2_y2_ortog,y2_y3_ortog,
             y3_y1_ortog,y3_y2_ortog,y3_y3_ortog,ncol=3)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Descomposición de varianza del error de pronóstico #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

# Aquí veremos la proporción de la varianza de error de pronóstico de cada variable
# explicada por las variables dentro del sistema

x11()
fevd(V.dr, n.ahead = 18)
plot(fevd(V.dr, n.ahead = 18),col=c("orange3", "firebrick4", "royalblue4"))
 
# VAR(1) --> VMA(infinito)

# Representación donde se obtienen las IRF sencillas.


Phi(V.dr, nstep=10) # Esta función nos calcula la matriz de coeficientes 
                   # n pasos adelante

# Coeficientes de las IRF ortogonales

Psi(V.dr, nstep=10)  

