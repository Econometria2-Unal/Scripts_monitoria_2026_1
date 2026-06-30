# Sesión 8: Modelos VECM y metodología de Johansen en R

Este documento presenta la versión en R de la sesión de monitoría de Econometría II sobre cointegración, modelos de corrección del error vectorial, VECM, y metodología de Johansen. Está dirigido a estudiantes universitarios que quieren entender la lógica general del material antes de ejecutar los scripts o revisarlos línea por línea.

La idea central de la sesión es estudiar qué se puede hacer cuando varias series económicas son no estacionarias, pero se mueven juntas en el largo plazo. En ese caso, diferenciar todas las variables puede eliminar información importante sobre equilibrio de largo plazo. Los modelos VECM permiten conservar esa relación de equilibrio y, al mismo tiempo, modelar los ajustes de corto plazo.

En términos generales, un VAR en niveles puede escribirse como:

$$
Y_t = A_1Y_{t-1} + A_2Y_{t-2} + \cdots + A_pY_{t-p} + u_t,
$$

donde \(Y_t\) es un vector de variables endógenas y \(u_t\) es el vector de innovaciones. Si las variables son \(I(1)\) y están cointegradas, este sistema puede reescribirse como un VECM:

$$
\Delta Y_t
= \Pi Y_{t-1}
+ \Gamma_1 \Delta Y_{t-1}
+ \cdots
+ \Gamma_{p-1}\Delta Y_{t-p+1}
+ u_t.
$$

El punto clave está en la matriz \(\Pi\). Si existe cointegración, entonces:

$$
\Pi = \alpha \beta',
$$

donde \(\beta\) contiene las relaciones de cointegración, es decir, los equilibrios de largo plazo, y \(\alpha\) contiene las velocidades de ajuste, es decir, la forma en que cada variable responde cuando el sistema se aleja de ese equilibrio.

## Ubicación del material en R

Este README se encuentra dentro de la carpeta de código en R:

```text
codigo/R/
```

Desde esta carpeta se documentan los siguientes archivos:

| Archivo | Rol dentro de la sesión |
|---|---|
| `Modelos_VECM_y_Johansen.R` | Script principal de la sesión. Construye, estima, valida e interpreta un modelo VECM usando la metodología de Johansen. |
| `funciones_auxiliares_VECM.R` | Script auxiliar con funciones de graficación y organización de resultados. Se usa como apoyo para el script principal. |

La base de datos usada por el script principal está en:

```text
datos/Petróleo.xlsx
```

Allí se encuentran las series mensuales de precios spot del petróleo Brent y WTI, usadas como ejemplo aplicado.

## Cómo leer los scripts

La ruta recomendada de estudio es:

1. Abrir primero `Modelos_VECM_y_Johansen.R`.
2. Ejecutarlo por secciones, siguiendo la tabla de contenidos del propio script.
3. Usar `funciones_auxiliares_VECM.R` únicamente como apoyo.

El archivo `funciones_auxiliares_VECM.R` no es el centro conceptual de la clase. No es necesario entenderlo por dentro, modificarlo ni memorizar sus funciones. Basta con saber que el script principal lo importa para producir gráficas y salidas más ordenadas.

En otras palabras: `funciones_auxiliares_VECM.R` es una caja de herramientas. Ustedes la usan, pero no tienen que abrirla ni cambiarla para entender la metodología VECM.

Para evitar problemas con rutas relativas, es recomendable abrir el proyecto o la carpeta de trabajo desde el nivel donde se encuentra:

```text
Sesión 8 - Modelos VECM/
```

El script principal usa `here` y `fs` para construir rutas hacia los datos y hacia el archivo auxiliar. Por eso, si el código no encuentra la base `Petróleo.xlsx` o el script auxiliar, lo primero que se debe revisar es el directorio de trabajo del proyecto.

## Objetivo de `Modelos_VECM_y_Johansen.R`

El objetivo de `Modelos_VECM_y_Johansen.R` es mostrar el flujo completo para trabajar con variables no estacionarias que pueden tener una relación de equilibrio de largo plazo.

El ejemplo usa dos precios internacionales del petróleo:

- `P.Brent`: precio spot del petróleo Brent.
- `P.WTI`: precio spot del petróleo WTI.

Las series tienen frecuencia mensual y cubren el periodo de enero de 2000 a diciembre de 2020. El sistema que se analiza es:

$$
Y_t =
\begin{pmatrix}
P.Brent_t \\
P.WTI_t
\end{pmatrix}.
$$

La pregunta econométrica de fondo es la siguiente:

> Si Brent y WTI son series no estacionarias, ¿existe una combinación de ambas que sea estacionaria y represente una relación de largo plazo?

Si la respuesta es sí, entonces las series están cointegradas y el modelo apropiado no es simplemente un VAR en diferencias. En ese caso, el VECM permite modelar simultáneamente:

- la relación de equilibrio de largo plazo;
- los ajustes de corto plazo;
- la velocidad con la que cada precio corrige desviaciones frente al equilibrio;
- pronósticos e impulso-respuesta a partir del sistema reparametrizado.

## Idea general del script principal

El script sigue la metodología de Johansen como una secuencia de trabajo. La lógica general es:

$$
\text{datos}
\rightarrow
\text{orden de integración}
\rightarrow
\text{VAR en niveles}
\rightarrow
\text{rango de cointegración}
\rightarrow
\text{VECM}
\rightarrow
\text{validación}
\rightarrow
\text{pronóstico e impulso-respuesta}.
$$

Esta secuencia es importante porque un VECM no se estima de forma aislada. Primero hay que justificar que las variables son candidatas razonables para cointegración. Después se determina cuántas relaciones de cointegración existen. Finalmente se estima e interpreta el modelo.

## Flujo de trabajo del script

### 1. Paquetes, rutas y funciones auxiliares

El script inicia cargando los paquetes necesarios:

| Paquete | Uso principal |
|---|---|
| `tidyverse` | Manipulación y visualización de datos. |
| `vars` | Estimación de VAR, diagnósticos, pronósticos y objetos `vec2var`. |
| `urca` | Pruebas de raíz unitaria y test de cointegración de Johansen. |
| `tsDyn` | Extracción de matrices \(\alpha\) y \(\beta\) del VECM. |
| `gridExtra` | Organización de varias gráficas en una misma ventana. |
| `readxl` | Lectura de archivos `.xlsx`. |
| `here` y `fs` | Manejo de rutas relativas del proyecto. |

Luego se define la ruta de la base de datos y se importa el archivo auxiliar:

```r
source(ruta_funciones_auxiliares_vecm, encoding = "UTF-8")
```

Esta línea carga las funciones que se usarán después para graficar series, pronósticos e impulso-respuesta. No es una parte que el estudiante deba modificar.

### 2. Carga y preparación de los datos

El script carga la base `Petróleo.xlsx` y transforma las columnas `Brent` y `WTI` en objetos de serie de tiempo mensual:

```r
P.Brent = ts(Data$Brent, start = c(2000, 1), frequency = 12)
P.WTI = ts(Data$WTI, start = c(2000, 1), frequency = 12)
```

Luego se construye la matriz de variables del sistema:

```r
Y = ts.intersect(P.Brent = P.Brent, P.WTI = P.WTI)
```

Esta matriz \(Y_t\) es la entrada principal del modelo. La función `ts.intersect()` garantiza que las series queden alineadas temporalmente.

También se grafican las series en niveles. Esta revisión visual es más importante de lo que parece: antes de aplicar pruebas formales, conviene observar si las series parecen tener tendencia, cambios de nivel, movimientos comunes o episodios de alta volatilidad.

### 3. Introducción a la metodología de Johansen

El script resume la metodología en cuatro etapas:

1. Verificar preliminarmente las variables: gráficos, orden de integración y número de rezagos. El número de rezagos se selecciona a partir de criterios de información sobre el VAR en niveles, pero también revisando que los residuales no presenten autocorrelación serial.
2. Determinar el rango de \(\Pi\), es decir, el número de relaciones de cointegración, y escoger el modelo apropiado según ese rango.
3. Analizar \(\beta\), la matriz de cointegración, y \(\alpha\), la matriz de velocidades de ajuste.
4. Validar supuestos y usar el modelo para pronósticos e impulso-respuesta.

La matriz \(\Pi\) es central porque resume la información de largo plazo. Su rango determina cuántas relaciones de cointegración existen:

- Si \(r = 0\), no hay cointegración.
- Si \(0 < r < k\), hay \(r\) relaciones de cointegración.
- Si \(r = k\), las variables serían estacionarias en niveles.

En esta sesión se trabaja con \(k = 2\), porque el sistema contiene Brent y WTI. Por tanto, el caso relevante es encontrar si existe una relación de cointegración entre ambas series.

### 4. Identificación del orden de integración

Antes de aplicar Johansen, el script revisa si las dos series son \(I(1)\). Para ello usa pruebas ADF con la función `ur.df()` del paquete `urca`.

Primero se prueban las series en niveles con distintas especificaciones:

- con tendencia;
- con deriva o intercepto;
- sin términos determinísticos.

Después se aplican pruebas ADF a las primeras diferencias:

```r
adf_d_brent = urca::ur.df(diff(P.Brent), lags = 12, type = "none")
adf_d_wti = urca::ur.df(diff(P.WTI), lags = 12, type = "none")
```

La conclusión del script es que Brent y WTI son series \(I(1)\): no son estacionarias en niveles, pero sí lo son en primeras diferencias. Esta es una condición natural para estudiar cointegración.

### 5. Selección de rezagos mediante un VAR en niveles

El siguiente paso es estimar un VAR en niveles para elegir el número de rezagos que se usará en el análisis de Johansen.

El script evalúa especificaciones con:

- tendencia e intercepto;
- intercepto;
- distintos criterios de información mediante `VARselect()`.

En el ejemplo se selecciona un VAR(3) en niveles con intercepto. El script fija este orden mediante:

```r
p_var = 3
VAR3 = VAR3_const
```

Este paso es clave porque el número de rezagos afecta el test de Johansen y la dinámica del VECM. No debe elegirse mecánicamente: además de mirar criterios de información, el script revisa autocorrelación serial de residuales usando `serial.test()` en rezagos 12, 16 y 20.

La revisión de estos rezagos tiene sentido porque las series son mensuales y se quiere comprobar si queda estructura serial no explicada en los residuales. En la versión actual del script, para el VAR(3) con intercepto no se rechaza la hipótesis de no autocorrelación serial en esos puntos de revisión.

Un detalle importante es que un VAR(3) en niveles se reparametriza como un VECM(2), porque el VECM trabaja con \(p-1\) rezagos de las primeras diferencias:

$$
VAR(3) \rightarrow VECM(2).
$$

El script enfatiza que, para continuar con la metodología de Johansen, el supuesto más importante en esta etapa es que los residuales del VAR en niveles no tengan correlación serial. Puede haber otros problemas, como heterocedasticidad, pero la ausencia de autocorrelación serial es la condición central para que la dinámica del sistema esté razonablemente capturada.

### 6. Determinación del rango de cointegración

Esta es la parte central de la sesión. El script usa la función `ca.jo()` para aplicar el test de Johansen.

Se revisan dos criterios:

| Criterio | Idea del contraste |
|---|---|
| Valor propio máximo, `type = "eigen"` | Contrasta \(H_0: r = j\) contra \(H_1: r = j + 1\). |
| Traza, `type = "trace"` | Contrasta \(H_0: r \leq j\) contra \(H_1: r > j\). |

También se comparan especificaciones determinísticas. El script explica tres posibilidades para el argumento `ecdet`:

```r
ecdet = "none"
ecdet = "const"
ecdet = "trend"
```

La especificación `ecdet = "none"` evalúa una relación de cointegración sin constante:

$$
P.Brent_t - \beta P.WTI_t = 0.
$$

La especificación `ecdet = "const"` permite una constante dentro de la relación de cointegración:

$$
P.Brent_t - \beta P.WTI_t + c = 0.
$$

La especificación `ecdet = "trend"` permite una tendencia lineal dentro de la relación de cointegración:

$$
P.Brent_t - \beta P.WTI_t + c + \delta t = 0.
$$

Para esta sesión, el caso más importante es `ecdet = "const"`, porque suele ser la especificación práctica más común cuando se permite que la relación de largo plazo tenga intercepto.

El script también fija:

```r
spec = "transitory"
```

Esta opción produce una representación del VECM equivalente a la que se trabaja en el curso. Existe también `spec = "longrun"`, pero esa representación no es la que se usa en esta monitoría.

Otro detalle clave es el argumento `K` de `ca.jo()`. En el script:

```r
K = p_var
```

Como previamente se estimó un VAR(3), entonces `K = 3` en el test de Johansen, aunque la reparametrización asociada sea un VECM(2). En otras palabras, `K` hace referencia al orden del VAR en niveles, no al número de rezagos en diferencias del VECM.

En el ejemplo, tanto el criterio del valor propio máximo como el criterio de la traza indican cointegración al 5%. Como el sistema tiene dos variables, la conclusión práctica es que existe una relación de cointegración entre Brent y WTI:

$$
r = 1.
$$

### 7. Estimación del VECM(2)

Una vez determinado el rango de cointegración, el script estima el VECM(2) usando:

```r
VEC_const = urca::cajorls(eigen_const, r = 1)
```

La función `cajorls()` estima el modelo condicionado al rango \(r = 1\). Luego se extraen dos objetos fundamentales:

```r
coefB(VEC_const)
coefA(VEC_const)
```

La matriz \(\beta\) contiene el vector de cointegración. En términos intuitivos, representa la combinación de Brent y WTI que debería comportarse como una relación estable de largo plazo.

La matriz \(\alpha\) contiene las velocidades de ajuste. Estos coeficientes indican qué variables reaccionan cuando el sistema se aleja del equilibrio de largo plazo. Si un coeficiente de \(\alpha\) es relevante, esa variable participa en la corrección del desequilibrio.

Una forma sencilla de leer esta parte es:

$$
\beta'Y_{t-1}
$$

representa el error de equilibrio del periodo anterior, mientras que \(\alpha\) indica cómo ese error entra en la dinámica de \(\Delta Y_t\).

### 8. Tendencia lineal en el modelo

El script usa `lttest()` para evaluar si hay evidencia de una tendencia lineal determinística en el VAR en niveles asociado a la reparametrización del VECM:

```r
urca::lttest(eigen_const, r = 1)
```

La hipótesis nula es que no existe tendencia lineal. En el ejemplo no se rechaza esa hipótesis, por lo que no se incluye tendencia lineal en el VAR en niveles asociado al VECM reparametrizado.

Esta decisión es importante porque los términos determinísticos cambian la interpretación del equilibrio de largo plazo. Incluir una tendencia cuando no corresponde puede distorsionar el análisis de cointegración.

### 9. Reparametrización del VECM como VAR en niveles

Para validar supuestos, pronosticar y calcular funciones impulso-respuesta, el script convierte el VECM a una representación compatible con las herramientas de `vars`:

```r
VAR.oil = vars::vec2var(eigen_const, r = 1)
```

El objeto resultante tiene clase `vec2var`. Esta reparametrización no significa que se ignore la cointegración. Al contrario: permite usar herramientas de VAR manteniendo la estructura de largo plazo estimada por Johansen.

Esta parte es muy importante para estudiantes: el VECM se estima como modelo de corrección del error, pero muchas herramientas prácticas, como diagnósticos, pronósticos e impulso-respuesta, se calculan sobre el objeto reparametrizado.

### 10. Validación de supuestos

El script valida tres aspectos del modelo reparametrizado:

| Supuesto | Herramienta usada |
|---|---|
| No autocorrelación serial | `serial.test()` |
| Homocedasticidad multivariada | `arch.test()` |
| Normalidad multivariada | `normality.test()` |

La validación no es un paso decorativo. Un modelo puede encontrar cointegración y aun así tener residuales problemáticos. Por eso se revisa si los errores se comportan razonablemente como innovaciones.

En el ejemplo se cumple el supuesto más importante para esta aplicación: no se detecta autocorrelación serial en los residuales del modelo reparametrizado. Sin embargo, sí se rechazan los supuestos de homocedasticidad y normalidad. Por eso, la recomendación del script es usar intervalos de confianza por bootstrap para realizar inferencia más cuidadosa en pronósticos e impulso-respuesta.

### 11. Pronóstico del VECM

Después de reparametrizar el modelo, el script produce pronósticos a 12 meses:

```r
prono_VECM = predict(
  VAR.oil,
  n.ahead = horizonte_pronostico,
  ci = int_conf_pronostico
)
```

El horizonte de pronóstico es:

```r
horizonte_pronostico = 12
```

La interpretación debe hacerse recordando que el modelo no solo extrapola la dinámica reciente. También incorpora la relación de largo plazo estimada entre Brent y WTI. Esa es una ventaja importante del VECM frente a un VAR en diferencias cuando las variables están cointegradas.

### 12. Funciones impulso-respuesta

El script calcula funciones impulso-respuesta ortogonalizadas usando una grilla para todas las combinaciones de impulso y respuesta:

```r
irf_ortog_vecm = graficar_grilla_irf(
  VAR.oil,
  variables,
  pasos_adelante,
  ortog = TRUE,
  int_conf = int_conf_irf,
  prefijo_titulo = "Impulso ortogonal",
  semilla = semilla_irf,
  runs = repeticiones_bootstrap_irf
)
```

Las funciones impulso-respuesta muestran cómo responde una variable del sistema ante un choque en otra variable. En este caso, permiten estudiar preguntas como:

- ¿Cómo responde Brent ante un choque en WTI?
- ¿Cómo responde WTI ante un choque en Brent?
- ¿La respuesta es estadísticamente significativa?
- ¿El efecto se disipa o permanece en el tiempo?

Como se usan respuestas ortogonalizadas, el orden de las variables importa. El script define inicialmente:

```r
variables = c("P.Brent", "P.WTI")
```

Bajo una identificación recursiva tipo Cholesky, la primera variable se interpreta como más contemporáneamente exógena que la segunda. En el orden original, Brent se trata como la variable más exógena y WTI como la más endógena.

Con este orden, el script muestra una interpretación sustantiva de las OIRF: un choque estructural en Brent afecta tanto al precio Brent como al precio WTI, mientras que un choque estructural en WTI no tiene un efecto relevante sobre Brent y su efecto sobre WTI tiende a disiparse hacia cero en el largo plazo.

### 13. Orden de las variables

La última sección del script muestra qué ocurre si se cambia el orden del sistema:

```r
variables_alt = c("P.WTI", "P.Brent")
```

El resultado central es que las conclusiones básicas de cointegración y diagnóstico pueden mantenerse similares, pero las funciones impulso-respuesta ortogonalizadas pueden cambiar de forma importante.

Este punto es fundamental: el orden de las variables no es un detalle técnico menor. En OIRF, el orden representa un supuesto económico sobre qué variable puede reaccionar contemporáneamente a cuál.

Por eso, antes de interpretar impulso-respuesta como evidencia económica, se debe justificar el orden elegido. En el script, poner Brent primero equivale a tratarlo como la variable más exógena dentro del sistema. Al invertir el orden y poner WTI primero, las OIRF cambian sustancialmente: los choques estructurales de WTI pasan a verse más significativos y persistentes.

La conclusión pedagógica es fuerte: el orden de las variables en una identificación por Cholesky no es trivial. Si el orden de exogeneidad no está claro por teoría económica o por evidencia estadística, las conclusiones de impulso-respuesta pueden depender demasiado del supuesto de identificación. El script menciona que una alternativa más estructural para enfrentar este problema son los S-VECM, donde se imponen restricciones económicas explícitas para identificar choques estructurales.

## Papel de `funciones_auxiliares_VECM.R`

El archivo `funciones_auxiliares_VECM.R` contiene funciones auxiliares para facilitar el trabajo visual y evitar repetir código. Incluye, entre otras, funciones para:

- graficar series de tiempo usadas en el VECM;
- graficar pronósticos del modelo reparametrizado;
- extraer datos de funciones impulso-respuesta;
- graficar funciones impulso-respuesta individuales;
- construir grillas de impulso-respuesta para todas las combinaciones de variables.

Este archivo debe entenderse como apoyo operativo. No es necesario que los estudiantes entiendan cada función interna para seguir la clase. Tampoco se recomienda modificarlo mientras se estudia el script principal, porque cambiarlo puede alterar las gráficas o producir errores difíciles de rastrear.

La forma correcta de usarlo es sencilla:

```r
source(ruta_funciones_auxiliares_vecm, encoding = "UTF-8")
```

Después de cargarlo, sus funciones quedan disponibles para el resto del script.

## Qué deberían aprender al finalizar

Al terminar esta sesión, la meta no es memorizar todos los comandos de R. La meta es entender la secuencia de razonamiento econométrico:

1. No toda serie no estacionaria debe diferenciarse sin pensar.
2. Si varias series son \(I(1)\), puede existir una relación estable de largo plazo.
3. La metodología de Johansen permite determinar el número de relaciones de cointegración.
4. Un VECM combina dinámica de corto plazo con equilibrio de largo plazo.
5. Las matrices \(\alpha\) y \(\beta\) son el corazón de la interpretación.
6. La validación de residuales sigue siendo necesaria.
7. Los pronósticos y las funciones impulso-respuesta deben interpretarse con cuidado.
8. En OIRF, el orden de las variables es un supuesto económico, no una decisión automática de R.

## Recomendación final para estudiantes

Lean `Modelos_VECM_y_Johansen.R` como una historia econométrica. Primero se observa el comportamiento de los datos, luego se verifica su orden de integración, después se pregunta si existe cointegración, y finalmente se estima un modelo que combina largo plazo y corto plazo.

La ruta mental de la sesión puede resumirse así:

$$
I(1) + \text{cointegración}
\rightarrow
\text{VECM}
\rightarrow
\alpha,\beta
\rightarrow
\text{validación}
\rightarrow
\text{interpretación económica}.
$$

Si esa idea queda clara, los comandos de R se vuelven herramientas para implementar una metodología, no una lista aislada de instrucciones.
