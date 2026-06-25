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


# ===
# 1. Simulación de un proceso VAR(1) con 3 variables ====
# ===

# Fijamos la semilla para que siempre dé el mismo resultado
set.seed(82901) 

# Determinamos un tamaño de muestra de 300 observaciones
T = 300 

# Se va a simular un modelo VAR_1 cuya ecuación está dada por: 
## Y_t = A_0 + A_1 Y_{t-1} + u_t

# Nota: Y_t es una matriz de 2 variables (una variable por columna). 
#       u_t también es una matriz de 2 variables, donde cada columna de la 
#       matriz es el correspondiente error asociado a cada variable de la matriz
#       Y_t

# Y_t Se crea como una matriz de 0s. Luego se llena con valores reales, cuando
#     ocurre la simulación
Y_t <- cbind(rep(0, T),rep(0, T)) 

# u_t es una matriz de errores normales independientes  
u_t = cbind(rnorm(T), rnorm(T))

# Definimos el vector constante A_0
A_0 = rbind(0.5,0.5) 

# Definimos la matriz de coeficientes autorregresivos.
A_1 = cbind(c(0.3, 0.5), c(0.2, 0.6)) # Matriz 2x2

# Nota: La idea de la simulación 

# Función que permite simular un VAR(1) 

sim_VAR1 = function(Y_t, A_1, u_t, T){
  for (i in 2:T) {
    # Se usa la fórmula de un VAR(1) para llenar cada una de las filas de Y_t
    Y_t[i,] = A_0 + A_1 %*% Y_t[i-1,] + u_t[i,] # Y_t = A_0 + A_1 Y_{t-1} + u_t
  }  
  return(Y_t)
}

# Nota: La simulación lo que busca es modelar las variables a partir de la 
#       fórmula de un VAR(1) en forma reducida: Y_t = A_0 + A_1 Y_{t-1} + u_t

# Nota: La función sim_VAR1 lo que busca es llenar mediante un ciclo, cada una 
#       de las filas (iteración por iteración) de Y_t. La matriz  
Y_t = sim_VAR1(Y_t, A_1, u_t, T) # 
                            # de ceros Y_t con valores

# Convertimos la serie en un objeto ts

Y_t = ts(Y_t, start=c(1900,1), frequency=4)

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


#~~~~~~~~~~~~~~~~~~~~~~#
# Gráficas de la serie #
#~~~~~~~~~~~~~~~~~~~~~~#

y1 = graficar_ts(Y_t[,1], titulo = "Variable y_1", color = "lightblue")
y2 = graficar_ts(Y_t[,2], titulo = "Variable y_2", color = "royalblue")

x11();grid.arrange(y1,y2,ncol=2)

# Recuerden que los modelos VAR requieren de series estacionarias. 

adf1= ur.df(Y_t[,1], lags=3,selectlags = "AIC",type="none")
summary(adf1) # Rechazo H0, la serie es I(0)

adf2= ur.df(Y_t[,2], lags=3,selectlags = "AIC",type="none")
summary(adf2) # Rechazo H0, la serie es I(0)


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

# Todos los criterios en cada prueba, recomiendan un VAR(1)

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


Acoef(V.dr) # Los valores teóricos eran: a11=0.3; a12=0.2; a21=0.5; a22=0.6. 
            # Las estimaciones son cercanas.

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
plot(P.20, names = "Series.1") # Los residuales de la primera serie se comportan bien
plot(P.20, names = "Series.2") # Los residuales de la segunda serie se comportan bien

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
  IRF_data_frame = data.frame(IRF$irf,IRF$Lower,IRF$Upper, pasos_adelante)
  # Gráfica de la función impulso respuesta
  graph = IRF_data_frame %>% 
    ggplot(aes(x=IRF_data_frame[,4], y=IRF_data_frame[,1], ymin=IRF_data_frame[,2], 
               ymax=IRF_data_frame[,3] )) +
    geom_hline(yintercept = 0, color="red") +
    geom_ribbon(fill="grey", alpha=0.2) +
    geom_line() +
    theme_light() +
    ggtitle(titulo)+
    ylab("")+
    xlab("Pasos adelante") +
    theme(plot.title = element_text(size = 11, hjust=0.5),
          axis.title.y = element_text(size=11))    
  return(graph)
}

# IRF de las variables del sistema ante distintos choques exógenos.

y1.y1 = impulso_respuesta(V.dr, "Series.1", "Series.1", pasos_adelante, ortog = F,
                          int_conf = 0.95, titulo = "Impulso de y1 - respuesta de y1")
y1.y2 = impulso_respuesta(V.dr, "Series.1", "Series.2", pasos_adelante, ortog = F,
                          int_conf = 0.95, titulo = "Impulso de y1 - respuesta de y2")
y2.y1 = impulso_respuesta(V.dr, "Series.2", "Series.1", pasos_adelante, ortog = F,
                          int_conf = 0.95, titulo = "Impulso de y2 - respuesta de y1")
y2.y2 = impulso_respuesta(V.dr, "Series.2", "Series.2", pasos_adelante, ortog = F, 
                          int_conf = 0.95, titulo = "Impulso de y2 - respuesta de y2")

x11()
grid.arrange(y1.y1,y1.y2,y2.y1,y2.y2,ncol=2)

# IRF Ortogonalizadas. 

y1.y1. = impulso_respuesta(V.dr, "Series.1", "Series.1", pasos_adelante, ortog = T,
                           int_conf = 0.95, titulo = "Impulso ortogonal de y1 - respuesta de y1")
y1.y2. = impulso_respuesta(V.dr, "Series.1", "Series.2", pasos_adelante, ortog = T,
                           int_conf = 0.95, titulo = "Impulso ortogonal de y1 - respuesta de y2")
y2.y1. = impulso_respuesta(V.dr, "Series.2", "Series.1", pasos_adelante, ortog = T,
                           int_conf = 0.95, titulo = "Impulso ortogonal de y2 - respuesta de y1")
y2.y2. = impulso_respuesta(V.dr, "Series.2", "Series.2", pasos_adelante, ortog = T,
                           int_conf = 0.95, titulo = "Impulso ortogonal de y2 - respuesta de y2")

x11()
grid.arrange(y1.y1.,y1.y2.,y2.y1.,y2.y2.,ncol=2)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Descomposición de varianza del error de pronóstico #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

# Aquí veremos la proporción de la varianza de error de pronóstico de cada variable
# explicada por las variables dentro del sistema

x11()
fevd(V.dr, n.ahead = 18)
plot(fevd(V.dr, n.ahead = 18),col=c("orange3", "firebrick4"))
 
# VAR(1) --> VMA(8^T)

# Representación donde se obtienen las IRF sencillas.


Phi(V.dr, nstep=10) # Esta función nos calcula la matriz de coeficientes 
                   # n pasos adelante

# Coeficientes de las IRF ortogonales

Psi(V.dr, nstep=10)  

