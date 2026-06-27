# Sesión 7: Modelos VAR en R

Este documento presenta la versión en R de la sesión de monitoría de Econometría II sobre modelos de vectores autorregresivos, conocidos como modelos VAR. Está dirigido a estudiantes universitarios que quieren entender la lógica general de los scripts antes de ejecutarlos o revisarlos línea por línea.

Los scripts muestran cómo se construye, estima, valida e interpreta un modelo VAR en dos escenarios complementarios:

1. Un escenario controlado, donde las series son simuladas y conocemos el verdadero proceso generador de datos.
2. Un escenario aplicado, donde se trabaja con datos macroeconómicos reales tomados del ejemplo de Enders.

La idea central de la sesión es estudiar sistemas dinámicos en los que varias variables se explican conjuntamente por sus propios rezagos y por los rezagos de las demás variables. En un VAR(p), el sistema puede escribirse como:

$$
Y_t = A_0 + A_1Y_{t-1} + A_2Y_{t-2} + \cdots + A_pY_{t-p} + u_t,
$$

donde \(Y_t\) es un vector de variables endógenas, \(A_0\) es el vector de constantes, \(A_1,\ldots,A_p\) son matrices de coeficientes y \(u_t\) es el vector de errores en forma reducida.

## Ubicación del material en R

Este README se encuentra dentro de la carpeta de código en R:

```text
codigo/R/
```

Desde la raíz del proyecto, esa es la ruta donde están los tres scripts documentados. Desde la ubicación de este README, los scripts están en la misma carpeta:

| Archivo | Rol dentro de la sesión |
|---|---|
| `Modelos_VAR_Series_Simuladas.R` | Simula un VAR(1) trivariado y luego aplica la metodología Box-Jenkins multivariada sobre las series simuladas. |
| `Modelos_VAR_ejemplo_Enders.R` | Estima un VAR con datos macroeconómicos trimestrales de Estados Unidos siguiendo el ejemplo de Enders. |
| `funciones_auxiliares_graficacion_VAR.R` | Contiene funciones auxiliares de graficación y extracción de resultados usadas por los dos scripts principales. |

## Cómo leer los scripts

La ruta recomendada de estudio es la siguiente:

1. Empezar con `Modelos_VAR_Series_Simuladas.R`, porque allí se conoce el verdadero modelo que genera las series. Esto permite comparar teoría, simulación y estimación.
2. Continuar con `Modelos_VAR_ejemplo_Enders.R`, donde la metodología se aplica a datos reales.
3. Usar `funciones_auxiliares_graficacion_VAR.R` solo como apoyo. No es necesario entenderlo en detalle para seguir la clase.

Para ejecutar los scripts principales, lo más conveniente es abrir el proyecto de R desde la raíz del repositorio:

```text
Sesión 7 - Modelos VAR/
```

Esto es importante porque los scripts usan rutas relativas del proyecto para cargar datos y funciones auxiliares. En particular, `Modelos_VAR_Series_Simuladas.R` carga el script auxiliar con:

```r
source("codigo/R/funciones_auxiliares_graficacion_VAR.R", encoding = "UTF-8")
```

Por tanto, aunque este README esté dentro de `codigo/R/`, se recomienda ejecutar los scripts con la raíz del proyecto como directorio de trabajo.

## Script auxiliar de graficación

El archivo `funciones_auxiliares_graficacion_VAR.R` no es un script conceptual de la clase. Es un archivo auxiliar que contiene funciones para facilitar la graficación y evitar repetir código.

Este archivo incluye, entre otras, funciones para:

- graficar series de tiempo;
- graficar diagnósticos de errores simulados;
- graficar pronósticos de modelos VAR;
- extraer y graficar funciones impulso-respuesta;
- construir grillas de IRF y OIRF para todas las combinaciones de impulso y respuesta.

Como estudiante, no es necesario modificar este archivo ni entender cada línea interna. Basta con saber que los otros dos scripts lo importan para producir gráficas más limpias y organizadas. En términos prácticos, este archivo es una "caja de herramientas" para la visualización.

## `Modelos_VAR_Series_Simuladas.R`

Este script construye un laboratorio econométrico. En lugar de comenzar con datos reales, primero se define un modelo VAR conocido y luego se generan artificialmente las series. Esto permite ver con claridad qué debería recuperar la estimación cuando el modelo está correctamente especificado.

### Objetivo del script

El objetivo es simular un proceso VAR(1) con tres variables:

$$
Y_t =
\begin{pmatrix}
y_{1t} \\
y_{2t} \\
y_{3t}
\end{pmatrix},
$$

y después estimar un modelo VAR sobre esas series simuladas. El proceso generador de datos usado en el script es:

$$
Y_t = A_0 + A_1Y_{t-1} + u_t.
$$

Como se conoce \(A_0\), \(A_1\) y la matriz de varianzas y covarianzas de \(u_t\), el estudiante puede comparar los parámetros teóricos con los parámetros estimados por R.

### Detalles importantes de la simulación

El script fija una semilla de simulación para que todos obtengan los mismos resultados:

```r
semilla_simulacion = 82901
set.seed(semilla_simulacion)
```

Luego define un tamaño muestral grande:

```r
T = 5000
```

La muestra amplia ayuda a que los resultados estimados se acerquen a los valores teóricos del modelo simulado.

Las tres variables simuladas se llaman:

```r
y_1, y_2, y_3
```

y los errores en forma reducida se llaman:

```r
u_1, u_2, u_3
```

La parte más importante de la simulación es la construcción de los errores \(u_t\). El script no simula errores independientes, sino errores correlacionados y con distintas desviaciones estándar. Para ello se parte de una matriz triangular inferior asociada a una descomposición de Cholesky:

$$
\Sigma_u = P_{\text{chol}}P_{\text{chol}}',
$$

donde \(\Sigma_u\) es la matriz de varianzas y covarianzas teórica de los errores reducidos. En el script se simula:

$$
u_t \sim N_3(0,\Sigma_u).
$$

Este punto es clave: la correlación contemporánea entre los errores afecta la interpretación de las funciones impulso-respuesta ortogonalizadas. El orden de las variables, \(y_1\), \(y_2\), \(y_3\), se usa como orden recursivo de Cholesky. Bajo esa identificación, \(y_1\) se interpreta como la variable contemporáneamente más exógena, luego \(y_2\), y finalmente \(y_3\).

Después de simular los errores, el script construye las series de manera recursiva. Para cada periodo se aplica:

$$
Y_t = A_0 + A_1Y_{t-1} + u_t.
$$

Es decir, cada observación actual depende de la constante, de los valores rezagados de las tres variables y del choque contemporáneo.

### Qué se aprende con este script

El script permite estudiar todo el ciclo de trabajo de un VAR:

- Simular errores multivariados normales con matriz de covarianzas conocida.
- Construir series a partir de un VAR(1).
- Verificar gráficamente las series simuladas y los errores.
- Aplicar pruebas ADF para revisar estacionariedad.
- Seleccionar rezagos con `VARselect()`.
- Estimar modelos VAR con tendencia, con constante y sin términos deterministas.
- Evaluar estabilidad mediante las raíces del sistema.
- Comparar la matriz teórica \(A_1\) con la matriz estimada.
- Comparar la matriz teórica \(\Sigma_u\) con la matriz estimada de residuales.
- Validar supuestos sobre los residuales.
- Calcular pronósticos, IRF, OIRF y FEVD.

### Validación del modelo

Una vez estimado el VAR(1), el script revisa tres supuestos importantes:

| Supuesto | Herramienta usada |
|---|---|
| No autocorrelación serial | `serial.test()` |
| Homocedasticidad multivariada | `arch.test()` |
| Normalidad multivariada | `normality.test()` |

La idea es que un buen modelo VAR no solo debe ajustar la dinámica de las variables, sino también dejar residuales que se comporten como innovaciones no predecibles.

### Uso del modelo estimado

Al final, el script usa el modelo para tres tareas centrales:

1. Pronosticar las tres series hacia adelante.
2. Analizar funciones impulso-respuesta no ortogonalizadas.
3. Analizar funciones impulso-respuesta ortogonalizadas y descomposición de varianza del error de pronóstico.

Las IRF no ortogonalizadas muestran respuestas ante innovaciones del sistema en forma reducida. Las OIRF usan Cholesky para transformar esos errores correlacionados en choques ortogonales. Por eso, en las OIRF el orden de las variables es una decisión econométrica importante.

## `Modelos_VAR_ejemplo_Enders.R`

Este script lleva la metodología a un caso aplicado con datos macroeconómicos. A diferencia del script de simulación, aquí no conocemos el verdadero proceso generador de datos. Por tanto, el trabajo consiste en preparar las series, justificar transformaciones, estimar el VAR y validar si el modelo resulta razonable.

### Objetivo del script

El objetivo es estimar un VAR con tres variables macroeconómicas trimestrales de Estados Unidos tomadas de la base de Enders:

- `IPI`: índice de producción industrial.
- `CPI`: índice de precios al consumidor.
- `Unem`: tasa de desempleo.

La muestra original cubre el periodo 1960T1-2012T4.

El modelo no se estima directamente con todas las variables en niveles. El script aplica transformaciones econométricamente razonables:

$$
\Delta \log(IPI_t)
$$

como aproximación al crecimiento de la producción industrial, y

$$
\Delta \log(CPI_t)
$$

como aproximación a la inflación trimestral. La tasa de desempleo se conserva en niveles.

Por tanto, el vector de variables que entra al VAR es:

$$
Y_t =
\begin{pmatrix}
\Delta \log(IPI_t) \\
Unem_t \\
\Delta \log(CPI_t)
\end{pmatrix}.
$$

En el código estas variables aparecen como:

```r
dl.IPI, Unem, dl.CPI
```

### Preparación de los datos

El archivo de datos usado es:

```text
../../datos/ENDERS.xlsx
```

Esa ruta está escrita desde la ubicación de este README. Dentro del script `Modelos_VAR_ejemplo_Enders.R`, la base se carga usando rutas del proyecto mediante `here` y `fs`, por lo que el archivo corresponde a:

```text
datos/ENDERS.xlsx
```

visto desde la raíz del proyecto. El script carga la base, convierte las variables a objetos de serie de tiempo trimestral y calcula diferencias logarítmicas para `IPI` y `CPI`.

Como al diferenciar se pierde la primera observación, la tasa de desempleo se alinea con el periodo común de las series transformadas. Así, todas las variables del VAR quedan sincronizadas temporalmente.

### Estacionariedad

Antes de estimar el VAR, el script revisa estacionariedad usando pruebas ADF. Esta es una etapa fundamental porque los VAR en niveles requieren trabajar con variables estacionarias, salvo que se esté tratando explícitamente un problema de cointegración y VECM.

El script primero revisa las variables originales en niveles y luego revisa las variables que efectivamente entran al VAR:

- crecimiento logarítmico del IPI;
- tasa de desempleo;
- inflación logarítmica del CPI.

Esta sección debe leerse como una justificación previa a la estimación: no basta con correr `VAR()`, primero hay que pensar si las series que entran al sistema tienen propiedades adecuadas.

### Identificación y selección de rezagos

El script usa `VARselect()` para revisar criterios de información bajo tres especificaciones:

- VAR con tendencia e intercepto;
- VAR con intercepto;
- VAR sin términos deterministas.

En el ejemplo se trabaja con:

```r
p_var = 3
```

Es decir, se estima un VAR(3). La decisión sobre el número de rezagos no debe entenderse como un paso mecánico. Los criterios de información ayudan, pero también se debe revisar si los residuales quedan libres de autocorrelación.

### Estimación del VAR

El script estima tres versiones:

```r
V.tr.1 = VAR(Y, p = 3, type = "both")
V.dr.1 = VAR(Y, p = 3, type = "const")
V.no.1 = VAR(Y, p = 3, type = "none")
```

Finalmente se trabaja con el VAR(3) con constante:

```r
VAR_enders = V.dr.1
```

La constante es razonable porque las variables transformadas pueden tener medias distintas de cero. Después de estimar, se revisa la estabilidad del sistema usando `roots()`. Para un VAR(p), R evalúa la estabilidad a través de la matriz compañera. La condición relevante es que los valores propios estén dentro del círculo unitario.

### Orden de las variables e interpretación de Cholesky

El orden del sistema es:

```r
dl.IPI, Unem, dl.CPI
```

Este orden es muy importante cuando se calculan funciones impulso-respuesta ortogonalizadas. Con identificación recursiva de Cholesky, el script interpreta a `dl.IPI` como la variable contemporáneamente más exógena, luego `Unem`, y finalmente `dl.CPI`.

Esto no es una conclusión automática del software. Es un supuesto de identificación. Antes de interpretar las OIRF como choques estructurales, el investigador debe preguntarse si ese orden tiene sentido económico.

### Validación del modelo

Luego de estimar el VAR(3), el script revisa:

| Supuesto | Herramienta usada |
|---|---|
| No autocorrelación serial | `serial.test()` con varios rezagos |
| Homocedasticidad multivariada | `arch.test()` |
| Normalidad multivariada | `normality.test()` |

También se grafican diagnósticos de residuales, ACF, PACF y residuales al cuadrado. Esta parte es esencial: si los residuales aún contienen estructura sistemática, el VAR no está capturando adecuadamente la dinámica del sistema.

### Pronóstico, IRF y FEVD

El script produce pronósticos a 12 trimestres con intervalos de confianza:

```r
predict(VAR_enders, n.ahead = 12, ci = 0.95)
```

También incluye una versión de pronóstico por bootstrap usando `VAR.etp`, con 1000 repeticiones. Este enfoque permite construir pronósticos simulando muchas trayectorias futuras posibles del sistema.

Después se calculan funciones impulso-respuesta:

- IRF no ortogonalizadas, asociadas a innovaciones en forma reducida.
- OIRF, asociadas a choques ortogonalizados mediante Cholesky.

Finalmente se calcula la descomposición de varianza del error de pronóstico, FEVD. La FEVD responde una pregunta muy importante:

> De la incertidumbre en el pronóstico de una variable, ¿qué proporción se explica por choques propios y qué proporción se explica por choques de las demás variables?

## Mensaje central para la sesión

Los modelos VAR son útiles porque permiten estudiar sistemas donde las variables se relacionan dinámicamente entre sí. Sin embargo, no basta con estimar el modelo. Una aplicación responsable debe seguir una secuencia:

1. Preparar y transformar adecuadamente las series.
2. Revisar estacionariedad.
3. Elegir un número razonable de rezagos.
4. Estimar distintas especificaciones determinísticas.
5. Verificar estabilidad.
6. Diagnosticar residuales.
7. Interpretar pronósticos, IRF, OIRF y FEVD con cuidado.

El script de series simuladas muestra esta lógica en un ambiente controlado. El script de Enders muestra cómo aplicar la misma lógica cuando se trabaja con datos reales y decisiones econométricas menos obvias.

## Recomendación final para estudiantes

No intenten memorizar cada comando. La meta es entender la secuencia de razonamiento:

$$
\text{datos} \rightarrow \text{transformaciones} \rightarrow \text{VAR} \rightarrow \text{validación} \rightarrow \text{interpretación}.
$$

Si esa ruta queda clara, los comandos de R se vuelven herramientas para implementar una idea econométrica, no una lista de instrucciones aisladas.
