# Sesión 7: Modelos VAR en Python

Este documento presenta la versión en Python de la sesión de monitoría de Econometría II sobre modelos de vectores autorregresivos, conocidos como modelos VAR. Está dirigido a estudiantes universitarios que quieren entender la lógica general de los scripts antes de ejecutarlos o revisarlos línea por línea.

Los scripts muestran cómo se construye, estima, valida e interpreta un modelo VAR en dos escenarios complementarios:

1. Un escenario controlado, donde las series son simuladas y conocemos el verdadero proceso generador de datos.
2. Un escenario aplicado, donde se trabaja con datos macroeconómicos reales tomados del ejemplo de Enders.

La idea central de la sesión es estudiar sistemas dinámicos en los que varias variables se explican conjuntamente por sus propios rezagos y por los rezagos de las demás variables. En un VAR(p), el sistema puede escribirse como:

$$
Y_t = A_0 + A_1Y_{t-1} + A_2Y_{t-2} + \cdots + A_pY_{t-p} + u_t,
$$

donde \(Y_t\) es un vector de variables endógenas, \(A_0\) es el vector de constantes, \(A_1,\ldots,A_p\) son matrices de coeficientes y \(u_t\) es el vector de errores en forma reducida.

## Ubicación del material en Python

Este README se encuentra dentro de la carpeta de código en Python:

```text
codigo/python/
```

Desde la raíz del proyecto, esa es la ruta donde están los tres scripts documentados. Desde la ubicación de este README, los scripts están en la misma carpeta:

| Archivo | Rol dentro de la sesión |
|---|---|
| `Modelos_VAR_Series_Simuladas.py` | Simula un VAR(1) trivariado y luego aplica la metodología Box-Jenkins multivariada sobre las series simuladas. |
| `Modelos_VAR_ejemplo_Enders.py` | Estima un VAR con datos macroeconómicos trimestrales de Estados Unidos siguiendo el ejemplo de Enders. |
| `funciones_auxiliares_graficacion_VAR.py` | Contiene funciones auxiliares de graficación, diagnóstico, IRF, pronóstico y FEVD usadas por los dos scripts principales. |

También puede aparecer una carpeta `__pycache__/`. Esa carpeta es generada automáticamente por Python al ejecutar módulos y no hace parte del contenido conceptual de la sesión.

## Cómo leer los scripts

La ruta recomendada de estudio es la siguiente:

1. Empezar con `Modelos_VAR_Series_Simuladas.py`, porque allí se conoce el verdadero modelo que genera las series. Esto permite comparar teoría, simulación y estimación.
2. Continuar con `Modelos_VAR_ejemplo_Enders.py`, donde la metodología se aplica a datos reales.
3. Usar `funciones_auxiliares_graficacion_VAR.py` solo como apoyo. No es necesario entenderlo en detalle para seguir la clase.

Los scripts están escritos con bloques `# %%`, lo que permite ejecutarlos por celdas en editores como VS Code o Spyder. Esta estructura es útil para una sesión de monitoría porque permite avanzar paso a paso: importar paquetes, preparar datos, estimar, diagnosticar y luego interpretar resultados.

La implementación de Python usa rutas construidas con `pathlib`. Por eso, los scripts pueden ubicar tanto el archivo auxiliar como la base de datos de Enders de forma relativamente robusta. Aun así, lo más ordenado es abrir el proyecto desde la raíz:

```text
Sesión 7 - Modelos VAR/
```

Desde la ubicación de este README, la base de datos usada en el ejemplo aplicado está en:

```text
../../datos/ENDERS.xlsx
```

Dentro del script `Modelos_VAR_ejemplo_Enders.py`, esa ruta se construye mediante:

```python
directorio_codigo_python = ruta_script
directorio_sesion_var = directorio_codigo_python.parent.parent
directorio_datos = directorio_sesion_var / "datos"
ruta_enders = directorio_datos / "ENDERS.xlsx"
```

## Script auxiliar de graficación

El archivo `funciones_auxiliares_graficacion_VAR.py` no es un script conceptual de la clase. Es un archivo auxiliar que contiene funciones para facilitar la graficación, organizar resultados y evitar repetir código.

Este archivo incluye, entre otras, funciones para:

- graficar series de tiempo con `matplotlib` y `seaborn`;
- graficar diagnósticos de errores simulados;
- graficar diagnósticos de residuales de modelos VAR;
- graficar pronósticos con bandas de confianza;
- extraer y graficar funciones impulso-respuesta;
- construir grillas de IRF y OIRF para todas las combinaciones de impulso y respuesta;
- calcular bandas bootstrap para IRF;
- graficar la descomposición de varianza del error de pronóstico.

Como estudiante, no es necesario modificar este archivo ni entender cada línea interna. Basta con saber que los otros dos scripts lo importan para producir gráficas más limpias y organizadas. En términos prácticos, este archivo es una "caja de herramientas" para la visualización y presentación de resultados.

## Paquetes principales usados

La versión en Python usa principalmente:

| Paquete | Uso principal |
|---|---|
| `numpy` | Simulación numérica, álgebra matricial y generación de errores aleatorios. |
| `pandas` | Organización de datos, series temporales e índices. |
| `statsmodels` | Estimación de modelos VAR, selección de rezagos, pronósticos, IRF, FEVD y pruebas de diagnóstico. |
| `scipy` | Pruebas estadísticas como Jarque-Bera. |
| `matplotlib` y `seaborn` | Graficación de series, diagnósticos, pronósticos e IRF. |
| `pathlib` y `sys` | Manejo de rutas e importación del script auxiliar. |

## `Modelos_VAR_Series_Simuladas.py`

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

Como se conoce \(A_0\), \(A_1\) y la matriz de varianzas y covarianzas de \(u_t\), el estudiante puede comparar los parámetros teóricos con los parámetros estimados por Python.

### Detalles importantes de la simulación

El script fija una semilla de simulación para que todos obtengan los mismos resultados:

```python
semilla_simulacion = 82901
generador = np.random.default_rng(semilla_simulacion)
```

Luego define un tamaño muestral grande:

```python
T = 5000
```

La muestra amplia ayuda a que los resultados estimados se acerquen a los valores teóricos del modelo simulado.

Las tres variables simuladas se llaman:

```python
variables = ["y_1", "y_2", "y_3"]
```

y los errores en forma reducida se llaman:

```python
errores = ["u_1", "u_2", "u_3"]
```

La parte más importante de la simulación es la construcción de los errores \(u_t\). El script no simula errores independientes, sino errores correlacionados y con distintas desviaciones estándar. Para ello se parte de una matriz triangular inferior asociada a una descomposición de Cholesky:

$$
\Sigma_u = P_{\text{chol}}P_{\text{chol}}',
$$

donde \(\Sigma_u\) es la matriz de varianzas y covarianzas teórica de los errores reducidos. En Python se generan primero errores estructurales ortogonales:

```python
eps_t = generador.normal(loc=0, scale=1, size=(T, len(errores)))
```

y luego se construyen los errores reducidos mediante:

```python
u_t = eps_t @ P_chol.T
```

Con esta construcción se obtiene:

$$
u_t \sim N_3(0,\Sigma_u).
$$

Este punto es clave: la correlación contemporánea entre los errores afecta la interpretación de las funciones impulso-respuesta ortogonalizadas. El orden de las variables, \(y_1\), \(y_2\), \(y_3\), se usa como orden recursivo de Cholesky. Bajo esa identificación, \(y_1\) se interpreta como la variable contemporáneamente más exógena, luego \(y_2\), y finalmente \(y_3\).

Después de simular los errores, el script construye las series de manera recursiva. Para cada periodo se aplica:

$$
Y_t = A_0 + A_1Y_{t-1} + u_t.
$$

En Python esto se implementa con una función `sim_VAR1()` que llena la matriz `Y_t` fila por fila.

### Detalle propio de Python: índices de tiempo

En la versión de R, las series se convierten en objetos `ts`. En Python, el script usa un índice numérico trimestral con `pandas`:

```python
tiempo = pd.Index(1900 + np.arange(T) / 4, name="tiempo")
```

Esto se hace porque 5000 trimestres desde 1900 exceden el rango de fechas tipo `Timestamp` de `pandas`. Para la estimación con `statsmodels`, el script usa después un `RangeIndex`:

```python
Y_t_modelo = Y_t.reset_index(drop=True)
modelo = VAR(Y_t_modelo)
```

Esta decisión es técnica, no econométrica: preserva las gráficas con escala temporal y evita problemas de compatibilidad al estimar el VAR.

### Qué se aprende con este script

El script permite estudiar todo el ciclo de trabajo de un VAR:

- Simular errores multivariados normales con matriz de covarianzas conocida.
- Construir series a partir de un VAR(1).
- Verificar gráficamente las series simuladas y los errores.
- Aplicar pruebas ADF con `adfuller()` para revisar estacionariedad.
- Seleccionar rezagos con `modelo.select_order()`.
- Estimar modelos VAR con tendencia, con constante y sin términos deterministas.
- Evaluar estabilidad mediante `roots` e `is_stable()`.
- Comparar la matriz teórica \(A_1\) con la matriz estimada.
- Comparar la matriz teórica \(\Sigma_u\) con la matriz estimada de residuales.
- Validar supuestos sobre los residuales.
- Calcular pronósticos, IRF, OIRF y FEVD.

### Validación del modelo

Una vez estimado el VAR(1), el script revisa tres aspectos importantes:

| Supuesto o diagnóstico | Herramienta usada en Python |
|---|---|
| No autocorrelación serial | `test_whiteness()` de `statsmodels` |
| Heterocedasticidad tipo ARCH | `het_arch()` por ecuación |
| Normalidad multivariada | `test_normality()` de `statsmodels` |
| Normalidad univariada | `scipy.stats.jarque_bera()` |

Aquí hay una diferencia importante frente a R. En `statsmodels` no hay un equivalente directo al `arch.test()` multivariado de `vars`. Por eso, el script aplica pruebas ARCH univariadas por ecuación como aproximación docente.

### Uso del modelo estimado

Al final, el script usa el modelo para tres tareas centrales:

1. Pronosticar las tres series hacia adelante.
2. Analizar funciones impulso-respuesta no ortogonalizadas.
3. Analizar funciones impulso-respuesta ortogonalizadas y descomposición de varianza del error de pronóstico.

En `statsmodels`, las IRF se obtienen mediante:

```python
V_dr.irf(10).irfs
```

y las OIRF mediante:

```python
V_dr.irf(10).orth_irfs
```

Las IRF no ortogonalizadas muestran respuestas ante innovaciones del sistema en forma reducida. Las OIRF usan Cholesky para transformar esos errores correlacionados en choques ortogonales. Por eso, en las OIRF el orden de las variables es una decisión econométrica importante.

## `Modelos_VAR_ejemplo_Enders.py`

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

```python
variables = ["dl.IPI", "Unem", "dl.CPI"]
```

### Preparación de los datos

El archivo de datos usado es:

```text
../../datos/ENDERS.xlsx
```

Esa ruta está escrita desde la ubicación de este README. Dentro del script `Modelos_VAR_ejemplo_Enders.py`, la base se carga construyendo la ruta con `pathlib`:

```python
ruta_enders = directorio_datos / "ENDERS.xlsx"
Base = pd.read_excel(ruta_enders)
```

El script carga la base, convierte las variables a objetos `Series` de `pandas` con un índice trimestral numérico y calcula diferencias logarítmicas para `IPI` y `CPI`:

```python
dl_IPI = np.log(IPI).diff().dropna()
dl_CPI = np.log(CPI).diff().dropna()
```

Como al diferenciar se pierde la primera observación, la tasa de desempleo se alinea con el periodo común de las series transformadas:

```python
Unem = UNEM.loc[dl_IPI.index]
```

Así, todas las variables del VAR quedan sincronizadas temporalmente.

### Estacionariedad

Antes de estimar el VAR, el script revisa estacionariedad usando pruebas ADF con `adfuller()`. Esta es una etapa fundamental porque los VAR en niveles requieren trabajar con variables estacionarias, salvo que se esté tratando explícitamente un problema de cointegración y VECM.

El script primero revisa las variables originales en niveles y luego revisa las variables que efectivamente entran al VAR:

- crecimiento logarítmico del IPI;
- tasa de desempleo;
- inflación logarítmica del CPI.

Esta sección debe leerse como una justificación previa a la estimación: no basta con correr `VAR()`, primero hay que pensar si las series que entran al sistema tienen propiedades adecuadas.

### Identificación y selección de rezagos

El script usa `select_order()` de `statsmodels` para revisar criterios de información bajo tres especificaciones:

- VAR con tendencia e intercepto: `trend="ct"`;
- VAR con intercepto: `trend="c"`;
- VAR sin términos deterministas: `trend="n"`.

En el ejemplo se trabaja con:

```python
p_var = 3
```

Es decir, se estima un VAR(3). La decisión sobre el número de rezagos no debe entenderse como un paso mecánico. Los criterios de información ayudan, pero también se debe revisar si los residuales quedan libres de autocorrelación.

### Estimación del VAR

El script estima tres versiones:

```python
V_tr_1 = modelo_enders.fit(p_var, trend="ct")
V_dr_1 = modelo_enders.fit(p_var, trend="c")
V_no_1 = modelo_enders.fit(p_var, trend="n")
```

Finalmente se trabaja con el VAR(3) con constante:

```python
VAR_enders = V_dr_1
```

La constante es razonable porque las variables transformadas pueden tener medias distintas de cero. Después de estimar, se revisa la estabilidad del sistema con dos herramientas de `statsmodels`:

```python
VAR_enders.roots
VAR_enders.is_stable(verbose=True)
```

En `statsmodels`, las raíces reportadas por `roots` deben quedar por fuera del círculo unitario. La función `is_stable()` revisa internamente la estabilidad del sistema usando la matriz compañera.

### Orden de las variables e interpretación de Cholesky

El orden del sistema es:

```python
["dl.IPI", "Unem", "dl.CPI"]
```

Este orden es muy importante cuando se calculan funciones impulso-respuesta ortogonalizadas. Con identificación recursiva de Cholesky, el script interpreta a `dl.IPI` como la variable contemporáneamente más exógena, luego `Unem`, y finalmente `dl.CPI`.

Esto no es una conclusión automática del software. Es un supuesto de identificación. Antes de interpretar las OIRF como choques estructurales, el investigador debe preguntarse si ese orden tiene sentido económico.

### Validación del modelo

Luego de estimar el VAR(3), el script revisa:

| Supuesto o diagnóstico | Herramienta usada en Python |
|---|---|
| No autocorrelación serial | `VAR_enders.test_whiteness()` |
| Heterocedasticidad tipo ARCH | `het_arch()` por ecuación |
| Normalidad multivariada | `VAR_enders.test_normality()` |
| Normalidad univariada | `jarque_bera()` por ecuación |

También se grafican diagnósticos de residuales, ACF y PACF con funciones auxiliares. Esta parte es esencial: si los residuales aún contienen estructura sistemática, el VAR no está capturando adecuadamente la dinámica del sistema.

### Pronóstico, IRF y FEVD

El script produce pronósticos a 12 trimestres con intervalos de confianza usando:

```python
VAR_enders.forecast_interval(
    y=ultimos_valores,
    steps=horizonte_pronostico,
    alpha=alpha_pronostico,
)
```

También incluye una versión de pronóstico por bootstrap. En R se usaba `VAR.etp`; en Python el script implementa una función propia:

```python
pronostico_bootstrap_var()
```

Esta función remuestrea residuales del VAR estimado, propaga la dinámica del modelo y genera muchas trayectorias futuras posibles. En el script se usan 1000 repeticiones y la semilla `202601`.

Después se calculan funciones impulso-respuesta:

- IRF no ortogonalizadas, asociadas a innovaciones en forma reducida.
- OIRF, asociadas a choques ortogonalizados mediante Cholesky.

La función auxiliar `graficar_grilla_irf()` calcula el objeto `irf()` una sola vez y organiza todas las combinaciones de impulso y respuesta en una grilla. Por defecto, las bandas de confianza se calculan mediante bootstrap dentro del script auxiliar.

Finalmente se calcula la descomposición de varianza del error de pronóstico, FEVD:

```python
fevd_enders = VAR_enders.fevd(periods=24)
```

La FEVD responde una pregunta muy importante:

> De la incertidumbre en el pronóstico de una variable, ¿qué proporción se explica por choques propios y qué proporción se explica por choques de las demás variables?

## Diferencias prácticas frente a la versión en R

La lógica econométrica es la misma, pero hay diferencias de implementación importantes:

| Tema | En R | En Python |
|---|---|---|
| Estimación VAR | `vars::VAR()` | `statsmodels.tsa.api.VAR().fit()` |
| Selección de rezagos | `VARselect()` | `select_order()` |
| Estabilidad | `roots()` | `roots` e `is_stable()` |
| Autocorrelación residual | `serial.test()` | `test_whiteness()` |
| ARCH multivariado | `arch.test()` | `het_arch()` por ecuación |
| Normalidad | `normality.test()` | `test_normality()` y `jarque_bera()` |
| Pronóstico | `predict()` | `forecast()` y `forecast_interval()` |
| IRF | `irf()`, `Phi()`, `Psi()` | `irf().irfs`, `irf().orth_irfs` |
| FEVD | `fevd()` | `fevd()` |
| Bootstrap de pronóstico | `VAR.etp::VAR.BPR()` | función propia `pronostico_bootstrap_var()` |

Estas diferencias no cambian la idea econométrica. Solo reflejan que R y Python organizan sus herramientas de manera distinta.

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

Si esa ruta queda clara, los comandos de Python se vuelven herramientas para implementar una idea econométrica, no una lista de instrucciones aisladas.
