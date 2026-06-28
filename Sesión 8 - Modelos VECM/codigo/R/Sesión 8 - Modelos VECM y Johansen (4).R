#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                           UNIVERSIDAD NACIONAL DE COLOMBIA
#                   Facultad de Ciencias Económicas | 2024 - 02
#                            Econometría II | Monitoría 
#
#                                     Sesión 8   
#                        Cointegración y metodologia Johansen
#                                  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Limpieza de entorno 
rm(list = ls())
dev.off()


#______________________________________________________________________________________________#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### 0. Instalación de paquetes ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

library(pacman)

pacman::p_load(vars,       # Para usar modelos VAR
               urca,       # Para hacer pruebas Dickey Fuller
               ggplot2,    # Para hacer gráficos con ggplot
               ggfortify,  # Para hacer gráficos de series de tiempo
               gridExtra,  # Para generar gráficos multiples
               dplyr,      # Para manipulación de dataframes
               tidyr,      # Para manipulación de dataframes
               readxl,     # Para leer archivos xlsx.
               tsDyn,      # Para usar metodología Johansen
               VAR.etp     # Para estimar modelos VAR
               )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### 1. Introducción a metodología Johansen ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Aspectos generales de la metodología de Johansen ----

# Consiste en un procedimiento en 4 etapas: 

## Etapa 1: Verificación preliminar de las variables a trabajar (Orden de 
#            integración y gráficas) e Identificación del número de rezagos del 
#            VECM mediante criterios de información sobre el VAR en niveles 
## Etapa 2: Determinación del rango de la matriz Pi (es decir del número de 
#            relaciones de cointegración) y estimación del VECM
## Etapa 3: Análisis de la matriz beta (matriz que contiene el vector de 
#            cointegración) y matriz alpha (matriz que contiene los parámetros 
#            de velocidad de ajuste)
## Etapa 4: Validación de supuestos y usos del modelo 


# Ejemplo 1: Precio de referencia Brent y WTI

# Vamos a utilizar una serie del precio spot del petróleo de referencia Brent y 
# una serie del precio spot del petróleo de referencia WTI. Las series tienen 
# frecuencia mensual y comprenden el periodo de Enero del 2000 a Diciembre de 2020.

#Vamos a cargar las series
Data = read_excel(file.choose())
attach(Data)

#Creamos los objetos ts
P.Brent = ts(Brent, start=c(2000,1), frequency=12)
P.WTI = ts(WTI,start=c(2000,1), frequency=12)


#~~ Gráficas de las series ~~#
autoplot(cbind(P.Brent, P.WTI), facets = F, main="Precios spot del petróleo", xlab="", ylab="", size=2)


#~~ Una gráfica más guapa ~~#
autoplot(cbind(P.Brent, P.WTI), facets = F, size = 1 ) + # PLot como tal, 
                                                         # Con facets = T, salen las
                                                         # series por separado.
  
  labs(title = "Precios spot del petróleo",
       subtitle = "(Petróleo Brent & WTI)", x = "", y = "",
       color = "Petróleos") + # Títulos y Ejes. Con color fijamos las leyendas
  
  scale_color_manual(
    values = c("P.Brent" = "lightblue", "P.WTI" = "coral"),  # Colores personalizados
    labels = c("P.Brent" = "Brent", "P.WTI" = "WTI")) +
  
  theme(
    legend.title = " ", #
    legend.position = "bottom"  # Posición de la leyenda
        ) +
  
  guides(color = guide_legend(override.aes = list(size = 2))) +  # Ajuste de la leyenda

  theme_classic() 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 1.1 Identificación del orden de integración de las series ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Procedemos a hacer las pruebas de raíz unitaria (Para identificar el orden de 
# integración de las dos series)

#~~ Referencia Brent 
summary(ur.df(P.Brent, lags = 12, type="trend")) # Tendencia no significativa
summary(ur.df(P.Brent, lags = 12, type="drift")) # Deriva no sinificativa
summary(ur.df(P.Brent, lags = 12, type="none"))  # Serie no estacionaria.

#~~ Referencia WTI
summary(ur.df(P.WTI, lags=12,  type="trend")) # Tendenncia no significativa
summary(ur.df(P.WTI, lags=12,  type="drift")) # Deriva no significativa
summary(ur.df(P.WTI, lags=12,  type="none"))  # Serie no estacionaria.

# Aplicamos diferenciación

#~~ Referencia Brent 
summary(ur.df(diff(P.Brent), lags=12, type="none")) # I(1)

#~~ Referencia WTI
summary(ur.df(diff(P.WTI), lags=12, type="none")) # I(1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 1.2 Modelo VAR en niveles ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Posteriormente, estimaremos un VAR en niveles para determinar el número rezagos del VECM

# Ojo: Se analizaran los criterios de información sobre el VAR en niveles 

# Se va a construir la matriz con las series

Y = cbind(P.Brent, P.WTI) 

# Selección
VARselect(Y, lag.max = 6, type="both", season=NULL) # Selección
 summary(VAR(Y, p=2, type="both", season=NULL))  # Estimación, tendencia no significativa

VARselect(Y, lag.max = 6, type="const", season=NULL) # Selección
 summary(VAR(Y, p=2, type="const", season=NULL))  # Estimación, intercepto significativo

# No tiene sentido analizar sin constante ya que este último modelo resulta no
# significativo.

# Elegimos VAR(2) en niveles 
 
VAR2 = VAR(Y, p = 2, type="const", season=NULL)

# Vamos a analizar el comportamiento de los residuales. Dada que es una serie 
# mensual, analicemos su comportamiento en puntos críticos.

P.12=serial.test(VAR2, lags.pt = 12, type = "PT.asymptotic");P.12 
  #Rechazo, se viola el supuesto
P.24=serial.test(VAR2, lags.pt = 24, type = "PT.asymptotic");P.24 
  #No rechazo, se cumple el supuesto
P.36=serial.test(VAR2, lags.pt = 36, type = "PT.asymptotic");P.36 
  #No rechazo, se cumple el supuesto

# A medida que se alejan los periodos, se cumple el supuesto.
# Efecto desvanecimiento. Es normal que ocurra esto, por lo que en general, 
# validaremos el cumplimiento del supuesto.

# Graficamos los residuales para 20 lags: 

x11(); plot(P.12, names = "P.Brent") # Bien comportados salvo por heterocedasticidad
x11(); plot(P.12, names = "P.WTI")   # Bien comportados salvo por heterocedasticidad

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### 2. Determinación del rango de la matriz Pi ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# La función ca.jo nos permitirá analizar el Test de Johansen.

# Argumentos del test de Johansen

# Para ver todos los argumentos del test de Johansen 

?ca.jo       # Acostumbrense a ver que hace una función siempre en R. 
args("ca.jo") 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 2.1 Test de Johansen - Sin intercepto ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~ Criterio del valor propio máximo ~~#

# Generalmente es la prueba preferida y la  más robusta. 
# El procedimiento que se analiza es:

# H0: r=0 vs H1: r=1, luego H0: r=1 vs H1: r=2, y así sucesivamente. Aquí k=2

eigen_none = ca.jo(Y, ecdet = "none", type = "eigen", K = 2, spec = "transitory")
summary(eigen_none) # Al 5% de confianza las series están cointegradas.

#~~ Criterio de la traza ~~#

# Es un procedimiento secuencial en donde se contrasta
# H0: r=0 vs H1: r>=1, luego H0: r<=1 vs H1: r>1, y así sucesivamente. Aquí k=2

trace_none= ca.jo(Y, ecdet = "none", type = "trace", K = 2, spec = "transitory")
summary(trace_none) #Al 5% de confianza las series están cointegradas.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 2.2 Test de Johansen - Con intercepto ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Criterio del Valor propio
eigen_const = ca.jo(Y, ecdet = "const", type = "eigen", K = 2, spec = "transitory")
summary(eigen_const) #Al 5% de confianza las series están cointegradas.

# Criterio de la traza
trace_const = ca.jo(Y, ecdet = "const", type = "trace", K = 2, spec = "transitory")
summary(trace_const) #Al 5% de confianza las series están cointegradas.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 2.3 Estimaciones de los modelos vistos ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~ Sin constante ~~#

#La función cajorls permite estimar el modelo VEC
VEC_none = cajorls(eigen_none, r=1)  # r=1 ya que encontramos una relación de Coint.
VEC_none

# Con esta función obtenemos el vector de cointegración normalizado
coefB(VEC_none)

# Con esta función obtenemos los coeficientes de velocidad de ajuste
coefA(VEC_none)

#~~ Con constante ~~#

#La función cajorls permite estimar el modelo VEC
VEC_const = cajorls(eigen_const, r=1) 
VEC_const

# Con esta función obtenemos el vector de cointegración normalizado
coefB(VEC_const)

#Con esta función obtenemos los coeficientes de velocidad de ajuste
coefA(VEC_const)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 2.5 Test para determinar Tendencia lineal en el modelo ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Tenemos la función lttest del paquete urca, veamos que hace.

?lttest

# Es decir 

# H0 : No existencia de tendencia lineal.
# H1 : Existencia de tendencia lineal.

lttest(eigen_const, r=1) 

# No rechazo la hipótesis nula, por lo que no se debe incluir tendencia 
# lineal en el modelo. 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### 4. Validación de supuestos y usos del modelo ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 4.1 Reparametrización del VECM como un VAR en niveles ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Nota: Dado los resultados de la prueba lttest, se usara el modelo VEC con 
# constante en el vector de cointegración

VAR.oil = vec2var(eigen_const, r = 1) 
# VAR.oil va a ser la reparametrización del modelo VEC anterior como VAR en niveles

VAR.oil
class(VAR.oil) # Notar que la clase del objeto ahora es vec2var. Esto es importante, 
               # dado que se necesita el modelo VEC reparametrizado como una VAR 
               # en niveles para poder validar los supuetos y poder hacer uso del modelo

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 4.2 Validación de supuestos de VECM --> VAR ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Autocorrelación: PT.asymptotic es para muestra grande y "PT.adjusted" es 
# corrección para muestra pequeña.

P.12_V=serial.test(VAR.oil, lags.pt = 12, type = "PT.asymptotic");P.12_V 
P.24_V=serial.test(VAR.oil, lags.pt = 24, type = "PT.asymptotic");P.24_V
P.36_V=serial.test(VAR.oil, lags.pt = 36, type = "PT.asymptotic");P.36_V


# Homocedasticidad: Test tipo ARCH multivariado 

arch.test(VAR.oil, lags.multi = 24, multivariate.only = TRUE) 
arch.test(VAR.oil, lags.multi = 12, multivariate.only = TRUE)


# Normalidad: Test Jarque-Bera multivariado
normality.test(VAR.oil) 


# Nota: Como se violan los supuestos de heterocedasticidad y normalidad, hay que 
#       calcular los intevalos de confianza mediante bootstrapping para poder hacer 
#       inferencia estadística correcta (tanto en los pronósticos como en las OIRF)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### 4.3 Pronóstico del VECM reparamterizado ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Recuerden que debido al incumplimiento de normalidad, los intervalos de confianza 
# deben computarse por bootstrapping.

prono_VECM = predict(VAR.oil, n.ahead=12, boots = T); prono_VECM
x11(); plot(prono_VECM)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### 4.4 Funciones impulso respuesta para VECM ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Función:

impulso_respuesta = function(var, impulso, respuesta, pasos_adelantes, ortog,
                             int_conf, titulo){
  # Cálculo de la función impulso respuesta
  total_pasos_futuros = length(pasos_adelantes) - 1
  IRF = irf(var, impulse=impulso, response=respuesta, n.ahead = total_pasos_futuros,
            ortho=ortog, ci = int_conf)
  IRF_data_frame = data.frame(IRF$irf,IRF$Lower,IRF$Upper, pasos_adelantes)
  # Gráfica de la función impulso respuesta
  graph = IRF_data_frame %>% 
    ggplot(aes(x=IRF_data_frame[,4], y=IRF_data_frame[,1], ymin=IRF_data_frame[,2],
               ymax=IRF_data_frame[,3] )) +
    geom_hline(yintercept = 0, color="blue") +
    geom_ribbon(fill="grey", alpha=0.2) +
    geom_line() +
    theme_light() +
    ggtitle(titulo)+
    ylab("")+
    xlab("Pasos Adelante") +
    theme(plot.title = element_text(size = 11, hjust=0.5),
          axis.title.y = element_text(size=11))    
  return(graph)
}

#Definimos el número pasos adelante
lags=c(0:12)

#IRF de las variables del sistema ante distintos choques exógenos.

y1.y1 = impulso_respuesta(VAR.oil,"P.WTI","P.WTI",pasos_adelantes=c(0:18),
                           ortog = T,int_conf = 0.95,titulo = "Impulso de P.WTI - Respuesta de P.WTI")
y1.y2 = impulso_respuesta(VAR.oil,"P.WTI","P.Brent",pasos_adelantes = c(0:18),
                           ortog = T,int_conf = 0.95,titulo = "Impulso de P.WTI - Respuesta de P.Brent")
y2.y1 = impulso_respuesta(VAR.oil,"P.Brent","P.WTI",pasos_adelantes = c(0:18),
                           ortog = T,int_conf = 0.95,titulo = "Impulso de P.Brent - Respuesta de P.WTI")
y2.y2 = impulso_respuesta(VAR.oil,"P.Brent","P.Brent",pasos_adelantes = c(0:18),
                           ortog = T,int_conf = 0.95,titulo = "Impulso de P.Brent - Respuesta de P.Brent")


x11();grid.arrange(y1.y1,y1.y2,y2.y1,y2.y2,ncol=2)

# Según esto, los impulsos de WTI no son significativos en las respuestas de Brent
# ni el mismo WTI.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### 5. Un comportamiento llamativo ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Cuando creamos el vector de VAR en niveles, fijamos en la primera varible el 
# precio del petróleo WTI. Veamos que sucede si hacemos un modesto cambio.

Y = cbind(P.WTI, P.Brent) 

# Estimemos de misma forma todo el modelo y veamos hasta donde llegamos con los
# impulso respuesta.

VAR2 = VAR(Y, p = 2, type="const", season=NULL)

P.12=serial.test(VAR2, lags.pt = 12, type = "PT.asymptotic");P.12 
P.24=serial.test(VAR2, lags.pt = 24, type = "PT.asymptotic");P.24 
P.36=serial.test(VAR2, lags.pt = 36, type = "PT.asymptotic");P.36 

# Los supuestos no se alteran significativamente.

# Estimación VECM

# Criterio del Valor propio y constante
eigen_const = ca.jo(Y, ecdet = "const", type = "eigen", K = 2, spec = "transitory")
summary(eigen_const) # Se mantiene la conclusión

# Reparametrización.
VAR.oil = vec2var(eigen_const, r = 1) 

P.12_V=serial.test(VAR.oil, lags.pt = 12, type = "PT.asymptotic");P.12_V 
P.24_V=serial.test(VAR.oil, lags.pt = 24, type = "PT.asymptotic");P.24_V
P.36_V=serial.test(VAR.oil, lags.pt = 36, type = "PT.asymptotic");P.36_V

# Los supuestos del VAR reparametrizado se mantienen similares.

#~~ Impulso Respuesta ~#

# Ahora, tenemos el momento decisivo. Veamos que ocurre con las OIRF.

y1.y1 = impulso_respuesta(VAR.oil,"P.WTI","P.WTI",pasos_adelantes=c(0:18),
                          ortog = T,int_conf = 0.95,titulo = "Impulso de P.WTI - Respuesta de P.WTI")
y1.y2 = impulso_respuesta(VAR.oil,"P.WTI","P.Brent",pasos_adelantes = c(0:18),
                          ortog = T,int_conf = 0.95,titulo = "Impulso de P.WTI - Respuesta de P.Brent")
y2.y1 = impulso_respuesta(VAR.oil,"P.Brent","P.WTI",pasos_adelantes = c(0:18),
                          ortog = T,int_conf = 0.95,titulo = "Impulso de P.Brent - Respuesta de P.WTI")
y2.y2 = impulso_respuesta(VAR.oil,"P.Brent","P.Brent",pasos_adelantes = c(0:18),
                          ortog = T,int_conf = 0.95,titulo = "Impulso de P.Brent - Respuesta de P.Brent")

x11();grid.arrange(y1.y1,y1.y2,y2.y1,y2.y2,ncol=2)

# Los impuslo respuesta cambian significativamente. ¿La noción de esto? Depende
# del contexto económico.

# Lo que muestra la conclusión teórica es que la variable que se coloca primero 
# es la más exogena de las dos. Por lo que, cuando construyan su VAR, coloquen 
# la más exogena arriba.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                          FIN DEL CÓDIGO                                   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#