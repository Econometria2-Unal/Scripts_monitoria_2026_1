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
#'  1.1 Especificación de las condiciones de la simulación
#'  1.2 Simulación de los errores en forma reducida " u_t "
#'    1.2.1 Construcción de los errores " u_t " usando descomposición de Cholesky
#'    1.2.2 Propiedades de los errores en forma reducida " u_t "
#'  1.3 Simulación del VAR(1) de 3 variables
#' 2. Metodologia Box-Jenkins para series multivariadas
#'  2.1. Identificación
#'  2.2. Estimación
#'  2.3. Validación de supuestos
#'  2.4. Uso del modelo: pronóstico y funciones Impulso respuesta (IRF)


# Nota: Tips prácticos en R
## Para limpiar el entorno de trabajo se puede correr el comando: " rm(list = ls()) "
## Para cerrar todas las gráficas actualmente abiertas se puede correr el comando: " dev.off() "
## Para resetear R se puede usar las teclas: Ctrl + Shift + F10
 

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

# Importación de funciones auxiliares de graficación del
# script auxiliar "funciones_auxiliares_graficacion_VAR.R"
source("codigo/R/funciones_auxiliares_graficacion_VAR.R", encoding = "UTF-8")

# Nota: "source" permite importar de manera manual scripts construidos
#       por uno mismo. En éste caso, importa todas las funciones auxiliares
#       de graficación que se encuentran en el script 
#       "funciones_auxiliares_graficacion_VAR.R"


# ===
# 1. Simulación de un proceso VAR(1) con 3 variables ====
# ===

# 1.1 Especificación de las condiciones de la simulación ----

# Fijamos la semilla para que siempre dé el mismo resultado
semilla_simulacion = 82901
set.seed(semilla_simulacion) 

# Determinamos un tamaño de muestra de 5000 observaciones
T = 5000 # Nota: Entre más muestra, mejor se dará la convergencia
         #       de los resultados simulados a los teóricos

# Nombre de las variables y los errores asociados
variables = c("y_1", "y_2", "y_3")
errores = c("u_1", "u_2", "u_3")

# Se va a simular un modelo VAR(1) cuya ecuación está dada por: 
## Y_t = A_0 + A_1 Y_{t-1} + u_t

# Nota: Y_t es una matriz de 3 variables (una variable por columna). 
#       u_t también es una matriz de 3 variables, donde cada columna de la 
#       matriz es el error asociado a cada variable de la matriz Y_t.

# Y_t Se crea como una matriz de 0s. Luego se llena con valores reales, cuando
#     ocurra la simulación del VAR(1)
Y_t <- matrix(0, nrow = T, ncol = length(variables),
              dimnames = list(NULL, variables)); head(Y_t)

# Nota: En este punto, Y_t es una matriz de ceros que se llenará con la simulación

# 1.2 Simulación de los errores en forma reducida " u_t " ----

#===
# Nota: Tenga presente que esta es la parte más importante de toda la simulación 
#       Dado que la forma en la que se simulan dichos errores, determinará
#       todas las características de las series simuladas Y_t. Ésto ocurre, porque
#       a partir de u_t, usando la fórmula Y_t = A_0 + A_1 Y_{t-1} + u_t es que
#       se construyen la serie Y_t. Por tanto, lo crucial de la simulación, 
#       es simular de manera correcta la distribución de los errores en forma 
#       reducida "u_t".
#===

# En esta simulación se quiere que los errores estén correlacionados y que,
# además, tengan desviaciones estándar diferentes. Para ello simulamos:
#
#   u_t ~ N_3(0, Sigma_u) ;  Distribución Normal Trivariada
#
# La matriz P_chol es triangular inferior. Esto permite construir una matriz de
# varianzas y covarianzas:
#
#   Sigma_u = P_chol P_chol' ; Donde P_chol es la matriz de la descomposicion de Cholesky
#
# Esta construcción es coherente con una identificación recursiva tipo Cholesky:
# y_1 es contemporáneamente más exógena que y_2 y y_3, mientras que y_2 es más
# exógena que y_3. Los errores reducidos u_t estarán correlacionados, pero los
# errores estructurales e_t que los generan son ortogonales.
#
# Nota: La correlación entre errores no garantiza por sí sola un orden de
# exogeneidad. El orden se impone mediante la estructura triangular de P_chol
# para la relación contemporánea. La matriz A_1, definida más abajo, gobierna
# los efectos rezagados y no necesita ser triangular para usar 
# la Descomposición de Cholesky.


# 1.2.1 Construcción de los errores " u_t " usando descomposición de Cholesky ----

# Acá en la simulación partimos al revés de la metodología de Box Jenkins. 
# Inicialmente, definimos la matriz de la descomposición de Cholesky "P_chol" 
# porque ella cumple dos roles importantes: 
  # 1. Determina el orden de exogenidad de las variables: y1, y2 y y3. 
  # 2. Determina la estructura de correlación de la matriz de varianzas y covarianzas
  #    de los errores (i.e. de la matriz Sigma_u_teorica, definida abajo.)

P_chol = matrix(c(0.70, 0.00, 0.00,
                  0.35, 1.10, 0.00,
                  0.25, 0.55, 1.60),
                nrow = 3, byrow = TRUE,
                dimnames = list(errores, c("eps_1", "eps_2", "eps_3")))

# Matriz de varianzas-covarianzas teórica de la distribución normal multivariada
Sigma_u_teorica = P_chol %*% t(P_chol); Sigma_u_teorica

# Matriz de Correlaciones teórica
cor_u_teorica = cov2cor(Sigma_u_teorica); cor_u_teorica

# Desviaciones estándar teóricas de los errores en forma reducida
desv_u_teoricas = sqrt(diag(Sigma_u_teorica)); desv_u_teoricas


# Nota: Existen dos formas de simular una distribución normal multivariada 
# (e.g. u_t ~ N_3(0, Sigma_u)) en R: 
  # MASS::mvrnorm() 
  # mvtnorm::rmvnorm()

# La media de los errores en forma reducida será el vector de ceros
media_u = rep(0, length(errores))
names(media_u) = errores

# Errores en forma reducida " u_t " simulados de una normal trivariada
# usando el paquete MASS
u_t_mass = MASS::mvrnorm(n = T, mu = media_u, Sigma = Sigma_u_teorica)
colnames(u_t_mass) = errores

# Errores en forma reducida " u_t " simulados de una normal trivariada
# usando el paquete mvtnorm
u_t_mvtnorm = mvtnorm::rmvnorm(n = T, mean = media_u, sigma = Sigma_u_teorica,
                               method = "chol")
colnames(u_t_mvtnorm) = errores

# Nota: Para la simulación del moderlo VAR, se usaran los errores que se obtienen
#       de la simulación de una distribución normal trivariada del paquete "mvtnorm"
u_t = u_t_mvtnorm

# 1.2.2 Propiedades de los errores en forma reducida " u_t " ----

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

# Verificación gráfica de la normalidad de los errores " u_t " simulados
graficos_errores = graficar_diagnostico_errores(u_t = u_t,
                                                errores = errores,
                                                cor_u_muestral = cor_u_muestral)

x11();grid.arrange(graficos_errores$series,
                   graficos_errores$histograma, ncol = 2)
x11();grid.arrange(graficos_errores$qq,
                   graficos_errores$correlacion, ncol = 2)

# 1.3 Simulación del VAR(1) de 3 variables ----

# Nota: Recuerde que Se va a simular un modelo VAR(1) cuya ecuación está dada por: 
## Y_t = A_0 + A_1 Y_{t-1} + u_t

# Definimos el vector constante A_0
A_0 = c(0.5, 0.2, -0.1); A_0 

# Definimos la matriz de coeficientes autorregresivos.
A_1 = matrix(c(0.35, 0.08, 0.04,
               0.25, 0.30, 0.06,
               0.15, 0.20, 0.25),
             nrow = 3, byrow = TRUE,
             dimnames = list(variables, paste0("L1.", variables))); A_1 # Matriz 3x3

# La matriz A_1 no es triangular inferior. Por tanto, la simulación permite
# efectos rezagados cruzados entre las tres variables. Esto separa claramente
# la dinámica del VAR de la identificación contemporánea de Cholesky: el orden
# recursivo y_1, y_2, y_3 se mantiene por el orden de las columnas de Y_t y por
# la estructura triangular de P_chol, no porque A_1 sea triangular.

# Nota: La idea de la simulación es: 
#       Dado que conocemos A_0, A_1 y ya simulamos los errores u_t de una 
#       distribución normal trivariada cuya matriz de varianza y varianzas 
#       está dada por: "Sigma_u_teorica = P_chol %*% t(P_chol)", procedemos
#       a llenar fila a fila la matriz Y_t usando al ecuación dinámica del VAR: 
#       Y_t = A_0 + A_1 Y_{t-1} + u_t

# Función que permite simular el VAR(1) de manera recursiva mediante un loop: 

sim_VAR1 = function(Y_t, A_0, A_1, u_t, T){
  for (i in 2:T) {
    # Se usa la fórmula de un VAR(1): Y_t = A_0 + A_1 Y_{t-1} + u_t
    # Para llenar cada una de las filas de Y_t
    Y_t[i,] = as.numeric(A_0 + A_1 %*% Y_t[i-1,] + u_t[i,]) # Y_t = A_0 + A_1 Y_{t-1} + u_t
  }  
  return(Y_t)
}

# Nota: La función sim_VAR1 lo que busca es llenar mediante un ciclo, cada una 
#       de las filas (iteración por iteración) de Y_t. La matriz Y_t pasa de ser una
#       matriz de ceros, a una matriz que contendrás los valores de las series 
#       simuladas. Note que, como se construyo la matriz de varianzas y covarianzas
#       "Sigma_u_teorica = P_chol %*% t(P_chol)" usando descomposición de Cholesky, 
#       existe un orden natural de las variables a simular, a saber: y1 es la variable 
#       más exogena, luego le sigue y_2 y por último la variable menos exógena 
#       (o más endógena) es y_3
Y_t = sim_VAR1(Y_t, A_0, A_1, u_t, T) 

# Convertimos la serie en un objeto de serie de tiempo "ts"
Y_t = ts(Y_t, start=c(1900,1), frequency=4)

# Gráficas de la series simuladas: 
y1 = graficar_ts(Y_t[,"y_1"], titulo = "Variable y_1", color = "lightblue")
y2 = graficar_ts(Y_t[,"y_2"], titulo = "Variable y_2", color = "royalblue")
y3 = graficar_ts(Y_t[,"y_3"], titulo = "Variable y_3", color = "darkorange")

x11();grid.arrange(y1,y2,y3,ncol=3)

# Nota: Recuerden que los modelos VAR requieren de series estacionarias. Por tanto, 
#       empleamos Test ADF para verificar la estacionariedad de las series. 

adf1= ur.df(Y_t[,"y_1"], lags=3,selectlags = "AIC",type="none")
summary(adf1) # Rechazo H0, la serie es I(0)

adf2= ur.df(Y_t[,"y_2"], lags=3,selectlags = "AIC",type="none")
summary(adf2) # Rechazo H0, la serie es I(0)

adf3= ur.df(Y_t[,"y_3"], lags=3,selectlags = "AIC",type="none")
summary(adf3) # Rechazo H0, la serie es I(0)

# ===
# 2. Metodología Box-Jenkins para series multivariadas ====
# ===

# ===
# 2.1. Identificación ====
# ===

# Ya tenemos las series simuladas en la matriz Y_t, por lo que ya es posible aplicar
# la metodología Box-Jenkins en las series simuladas que se encuentran en Y_t. 

# Beamos que rezago nos recomienda la función VARselect() 

# Selección de rezagos para un VAR con tendencia e intercepto.
VARselect(Y_t, lag.max=6,type = "both", season = NULL)

# Selección de rezagos para un VAR con sólo intercepto.
VARselect(Y_t, lag.max=6,type = "const", season = NULL)

# Selección de rezagos para un VAR sin términos determinísticos.
VARselect(Y_t, lag.max=6,type = "none", season = NULL)

# Como el proceso generador de datos es un VAR(1), esperamos que los criterios
# de información favorezcan rezagos bajos, especialmente p = 1.

# ===
# 2.2. Estimación ====
# ===

# Para seleccionar el VAR(1), verificamos si tiene intercepto y deriva:

# VAR con tendencia e intercepto
V.tr = VAR(Y_t, p=1, type="both", season=NULL)
summary(V.tr) # La tendencia no es significativa, analizamos const

# VAR con intercepto.
V.dr= VAR(Y_t, p=1, type="const", season=NULL) 
summary(V.dr) # El intercepto es significativo en dos de las 3 ecuaciones

# VAR sin términos determinísticos.
V.no = VAR(Y_t, p=1, type="none", season=NULL)  
summary(V.no)

# Elegimos el modelo con constante, pues se ha visto que tienen constante signi-
# ficativa.

# Estabilidad del VAR(1):

# Nota: Los valores propios de la matriz A_1 deben ser en valor absoluto menores 
# a 1 para que el VAR(1) sea estacionario, con ello se garantiza que sus inversos 
# múltiplicativos, que son las raíces del polinomio característico asociado al 
# proceso VAR, sean en valor absoluto mayores a 1 y se pueda garantizar la 
# estabilidad del proceso.

roots(V.dr) #El proceso es estable.

# Coeficientes estimados del VAR(1):

# Podemos ver todos los coeficientes con Bcoef
Bcoef(V.dr) 

# Matriz teóricas "A1" vs matriz estimada "Acoef(V.dr)"

A_1 # Matriz teórica usada en la simulación.
A_1_sim = Acoef(V.dr); A_1_sim # Las estimaciones deberían ser cercanas a A_1.

# Matriz de varianzas y covarianzas de los residuales teórica vs estimada

# Matriz de varianzas y covarianzas téorica
Sigma_u_teorica

# Matriz de varianzas y covarianzas estimada
Sigma.est = summary(V.dr)$covres; Sigma.est 

# Análisis de todos el modelo VAR(1) estimado
summary(V.dr)

# ===
# 2.3. Validación de supuestos ====
# ===

# No autocorrelación serial ===

# PT.asymptotic es para muestra grande y "PT.adjusted" para muestra pequeña.
P.75=serial.test(V.dr, lags.pt = 75, type = "PT.asymptotic");P.75 # No rechazo
P.30=serial.test(V.dr, lags.pt = 30, type = "PT.asymptotic");P.30 # No rechazo
P.20=serial.test(V.dr, lags.pt = 20, type = "PT.asymptotic");P.20 # No rechazo
P.10=serial.test(V.dr, lags.pt = 10, type = "PT.asymptotic");P.10 # No rechazo


# Graficamos los resultados del test usando 20 residuos: Se grafican los residuales, 
# su distribución, la ACF y PACF de los residuales y a ACF y PACF de los 
# residuales al cuadrado (proxy para heterocedasticidad)

x11()
plot(P.20, names = "y_1") # Residuales de la primera serie
plot(P.20, names = "y_2") # Residuales de la segunda serie
plot(P.20, names = "y_3") # Residuales de la tercera serie

# Nota: Se cumple el supuesto de no autocorrelación serial en los residuales

# Homocedasticidad ===

# Test tipo ARCH multivariado
arch.test(V.dr, lags.multi = 24, multivariate.only = TRUE) # No rechazo
arch.test(V.dr, lags.multi = 12, multivariate.only = TRUE) # No rechazo

# Nota: Se cumple el supuesto de homocedasticidad

# Normalidad ===

# Jarque-Bera para series multivariadas.  
normality.test(V.dr) # No rechazo, se cumple el supuesto.

# Nota: Se cumple el supuesto de normalidad

# ===
# 2.4. Uso del modelo: pronóstico y funciones Impulso respuesta (IRF) ====
# ===

# Pronóstico ===


pronostico_var = predict(V.dr, n.ahead = 12,ci=0.95) 
pronostico_var

# Graficas pronóstico
x11()
graficar_pronostico_var(pronostico_var) 

# Versión Fanchart
fanchart(predict(V.dr, n.ahead = 12), colors = c("blue","lightblue"))

# Funciones de impulso respuesta no ortogonalizadas ===

# Nota: Recuede que para poder calcular las IRF de un modelo VAR
#       este debe tener su representación como VMA(infinito). 
#       Es decir, pasamos del VAR(1) --> VMA(infinito)

# IRFs no ortogonalizadas: 
Phi(V.dr, nstep=10) # Esta función nos calcula la matriz de coeficientes de 
                    # las IRFs no ortogonalizadas "n pasos adelante"

# Graficación de las IRFs

# Definimos el número pasos adelante
pasos_adelante = 0:18
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_bootstrap_irf = 100

# La funcion graficar_grilla_irf() calcula el objeto irf() una sola vez
# y luego crea cada panel con programacion funcional.
# IRF de las variables del sistema ante distintos choques exogenos.
irf_no_ortog = graficar_grilla_irf(V.dr, variables, pasos_adelante,
                                   ortog = FALSE, int_conf = int_conf_irf,
                                   prefijo_titulo = "Impulso",
                                   semilla = semilla_irf,
                                   runs = repeticiones_bootstrap_irf)

x11()
# Grilla de IRF: columnas = impulsos; filas = respuestas.
grid.arrange(grobs = irf_no_ortog$graficas,
             layout_matrix = matrix(seq_along(irf_no_ortog$graficas),
                                    nrow = length(variables), byrow = TRUE))

# Funciones de impulso respuesta ortogonalizadas ===

# IRF Ortogonalizadas. 
# Cuando ortog = TRUE, la función irf() usa la descomposición de Cholesky de la
# matriz de varianzas y covarianzas de los residuales. En este script el orden
# de las variables es y_1, y_2, y_3; por tanto, la identificación recursiva
# interpreta a y_1 como la variable contemporáneamente más exógena, luego y_2 y
# finalmente y_3 (como la más endógena). Esta es una restricción de identificación: 
# los errores en forma reducida " u_t " pueden estar correlacionados, pero los choques 
# ortogonalizados son los que se interpretan como innovaciones estructurales recursivas.

# IRFs ortogonalizadas: 
Psi(V.dr, nstep=10) # Esta función nos calcula la matriz de coeficientes de 
                    # las IRFs ortogonalizadas "n pasos adelante"  

# Graficación de las IRFs

# Usamos los mismos pasos adelante, intervalo de confianza y semilla.
# IRFs ortogonalizadas de las variables del sistema ante distintos choques exogenos.
irf_ortog = graficar_grilla_irf(V.dr, variables, pasos_adelante,
                                ortog = TRUE, int_conf = int_conf_irf,
                                prefijo_titulo = "Impulso ortogonal",
                                semilla = semilla_irf,
                                runs = repeticiones_bootstrap_irf)

x11()
# Grilla de OIRF: columnas = impulsos; filas = respuestas.
grid.arrange(grobs = irf_ortog$graficas,
             layout_matrix = matrix(seq_along(irf_ortog$graficas),
                                    nrow = length(variables), byrow = TRUE))


# Descomposición de varianza del error de pronóstico ===

# La descomposición de varianza del error de pronóstico (FEVD) da la proporción de la 
# varianza de error de pronóstico de cada variable explicada por las variables 
# dentro del sistema

x11()
fevd(V.dr, n.ahead = 18)
plot(fevd(V.dr, n.ahead = 18),col=c("orange3", "firebrick4", "royalblue4"))
