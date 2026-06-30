# Sesión 8: Modelos VECM y metodología de Johansen en Python

Este documento presenta la versión en Python de la sesión de monitoría de Econometría II sobre cointegración, modelos de corrección del error vectorial, VECM, y metodología de Johansen. Está dirigido a estudiantes universitarios que quieren entender la lógica general del material antes de ejecutar los scripts o revisarlos línea por línea.

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

## Ubicación del material en Python

Este README se encuentra dentro de la carpeta de código en Python:

```text
codigo/python/
```

Desde esta carpeta se documentan los siguientes archivos:

| Archivo | Rol dentro de la sesión |
|---|---|
| `Modelos_VECM_y_Johansen.py` | Script principal de la sesión. Construye, estima, valida e interpreta un modelo VECM usando `statsmodels` y la metodología de Johansen. |
| `funciones_auxiliares_VECM.py` | Script auxiliar con funciones de impresión, diagnóstico, pronóstico, graficación e impulso-respuesta. Se usa como apoyo para el script principal. |

La base de datos usada por el script principal está en:

```text
datos/Petróleo.xlsx
```

Allí se encuentran las series mensuales de precios spot del petróleo Brent y WTI, usadas como ejemplo aplicado.

## Cómo leer los scripts

La ruta recomendada de estudio es:

1. Abrir primero `Modelos_VECM_y_Johansen.py`.
2. Ejecutarlo por bloques o celdas, aprovechando las marcas `# %%`.
3. Usar `funciones_auxiliares_VECM.py` únicamente como apoyo.

El archivo `funciones_auxiliares_VECM.py` no es el centro conceptual de la clase. No es necesario entenderlo por dentro, modificarlo ni memorizar sus funciones. Basta con saber que el script principal lo importa para producir salidas más ordenadas, diagnósticos y gráficas.

En términos prácticos, `funciones_auxiliares_VECM.py` es una caja de herramientas. Ustedes la usan, pero no tienen que abrirla ni cambiarla para entender la metodología VECM.

Para evitar problemas con rutas relativas, es recomendable ejecutar el script manteniendo la estructura del proyecto:

```text
Sesión 8 - Modelos VECM/
```

El script principal construye rutas con `pathlib`:

```python
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "datos"
ruta_petroleo = DATA_DIR / "Petróleo.xlsx"
```

Por eso, si Python no encuentra la base `Petróleo.xlsx`, lo primero que se debe revisar es que el archivo se esté ejecutando desde su ubicación original dentro del repositorio.

## Objetivo de `Modelos_VECM_y_Johansen.py`

El objetivo de `Modelos_VECM_y_Johansen.py` es mostrar el flujo completo para trabajar en Python con variables no estacionarias que pueden tener una relación de equilibrio de largo plazo.

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

La pregunta econométrica de fondo es:

> Si Brent y WTI son series no estacionarias, ¿existe una combinación de ambas que sea estacionaria y represente una relación de largo plazo?

Si la respuesta es sí, entonces las series están cointegradas y el modelo apropiado no es simplemente un VAR en diferencias. En ese caso, el VECM permite modelar simultáneamente:

- la relación de equilibrio de largo plazo;
- los ajustes de corto plazo;
- la velocidad con la que cada precio corrige desviaciones frente al equilibrio;
- pronósticos e impulso-respuesta a partir del sistema estimado.

## Idea general del script principal

El script sigue la metodología de Johansen como una secuencia de trabajo:

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

El script inicia cargando herramientas de Python:

| Paquete o módulo | Uso principal |
|---|---|
| `pathlib` | Manejo de rutas relativas del proyecto. |
| `numpy` | Cálculo numérico y arreglos. |
| `pandas` | Manejo de datos y series de tiempo. |
| `statsmodels.tsa.api.VAR` | Estimación y selección de rezagos del VAR en niveles. |
| `statsmodels.tsa.stattools.adfuller` | Pruebas ADF de raíz unitaria. |
| `statsmodels.tsa.vector_ar.vecm.VECM` | Estimación del modelo VECM. |
| `statsmodels.tsa.vector_ar.vecm.coint_johansen` | Test de cointegración de Johansen. |
| `funciones_auxiliares_VECM.py` | Funciones de apoyo para imprimir, graficar y organizar resultados. |

Luego se importan funciones auxiliares como:

```python
from funciones_auxiliares_VECM import (
    configurar_entorno_graficas,
    graficar_series_vecm,
    imprimir_adf,
    imprimir_tabla_johansen,
    predecir_vecm,
    graficar_grilla_irf,
)
```

Estas funciones permiten que el script principal sea más legible. La idea es que el estudiante se concentre en la metodología econométrica y no en detalles internos de programación para armar tablas o gráficas.

### 2. Carga y preparación de los datos

El script carga la base `Petróleo.xlsx` con `pandas`:

```python
Data = pd.read_excel(ruta_petroleo)
```

Luego crea un índice temporal mensual con `PeriodIndex`:

```python
tiempo = pd.period_range(
    start="2000-01",
    periods=len(Data),
    freq="M",
    name="tiempo",
)
```

Este punto es importante porque en Python una serie de tiempo no es solamente un vector de datos. En la práctica se necesita:

$$
\text{serie de tiempo en Python}
=
\text{datos}
+
\text{índice temporal}.
$$

Después se construyen dos objetos `pandas.Series`:

```python
P_Brent = pd.Series(Data["Brent"].to_numpy(), index=tiempo, name="P.Brent")
P_WTI = pd.Series(Data["WTI"].to_numpy(), index=tiempo, name="P.WTI")
```

y luego se arma la matriz del sistema:

```python
Y = pd.concat([P_Brent, P_WTI], axis=1)
Y.columns = ["P.Brent", "P.WTI"]
```

Para estimar modelos con `statsmodels`, el script también crea una versión sin índice temporal especial:

```python
Y_modelo = Y.reset_index(drop=True)
```

La razón es práctica: el índice mensual se conserva en `Y` para tablas y gráficas, mientras que `Y_modelo` se usa como entrada limpia para las rutinas de `statsmodels`.

### 3. Introducción a la metodología de Johansen

El script resume la metodología en cuatro etapas:

1. Verificar preliminarmente las variables: gráficos, orden de integración y número de rezagos.
2. Determinar el rango de \(\Pi\), es decir, el número de relaciones de cointegración.
3. Analizar \(\beta\), la matriz de cointegración, y \(\alpha\), la matriz de velocidades de ajuste.
4. Validar supuestos y usar el modelo para pronósticos e impulso-respuesta.

La matriz \(\Pi\) es central porque resume la información de largo plazo. Su rango determina cuántas relaciones de cointegración existen:

- Si \(r = 0\), no hay cointegración.
- Si \(0 < r < k\), hay \(r\) relaciones de cointegración.
- Si \(r = k\), las variables serían estacionarias en niveles.

En esta sesión se trabaja con \(k = 2\), porque el sistema contiene Brent y WTI. Por tanto, el caso relevante es encontrar si existe una relación de cointegración entre ambas series.

### 4. Identificación del orden de integración

Antes de aplicar Johansen, el script revisa si las dos series son \(I(1)\). Para ello usa la prueba ADF con `adfuller()`.

En `statsmodels`, el argumento `regression` controla los términos determinísticos:

| Opción | Interpretación |
|---|---|
| `regression="ct"` | ADF con constante y tendencia. |
| `regression="c"` | ADF con constante. |
| `regression="n"` | ADF sin términos determinísticos. |

Primero se prueban las series en niveles. Después se prueban las primeras diferencias:

```python
d_P_Brent = P_Brent.diff().dropna()
d_P_WTI = P_WTI.diff().dropna()
```

La conclusión del script es que Brent y WTI son series \(I(1)\): no son estacionarias en niveles, pero sí lo son en primeras diferencias. Esta es una condición natural para estudiar cointegración.

### 5. Selección de rezagos mediante un VAR en niveles

El siguiente paso es estimar un VAR en niveles para elegir el número de rezagos que se usará en el análisis de Johansen.

En Python se crea primero el objeto:

```python
modelo_petroleo = VAR(Y_modelo)
```

Luego se evalúan especificaciones con:

- tendencia e intercepto, `trend="ct"`;
- intercepto, `trend="c"`;
- sin términos determinísticos, `trend="n"`.

El script usa `select_order()` para revisar criterios de información. En el ejemplo se trabaja con:

```python
p_var = 3
VAR3_const = modelo_petroleo.fit(p_var, trend="c")
VAR3 = VAR3_const
```

Es decir, se selecciona un VAR(3) en niveles con intercepto. Como el VAR en niveles tiene \(p = 3\), su reparametrización como VECM tiene \(p-1 = 2\) rezagos en diferencias:

$$
VAR(3) \rightarrow VECM(2).
$$

El script valida preliminarmente los residuales del VAR usando `test_whiteness()` en 12, 16 y 20 rezagos:

```python
VAR3.test_whiteness(nlags=12, adjusted=False)
VAR3.test_whiteness(nlags=16, adjusted=False)
VAR3.test_whiteness(nlags=20, adjusted=False)
```

El punto importante es que, para continuar con la metodología de Johansen, los residuales del VAR en niveles no deben presentar autocorrelación serial. Puede haber otros problemas, como heterocedasticidad, pero la ausencia de autocorrelación serial es el supuesto clave para que la dinámica del sistema esté razonablemente capturada.

### 6. Determinación del rango de cointegración

Esta es la parte central de la sesión. En Python se usa:

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen
```

La función `coint_johansen()` recibe dos argumentos especialmente importantes:

```python
det_order = ...
k_ar_diff = ...
```

El argumento `det_order` controla la especificación determinística del test:

| En R con `ca.jo()` | En Python con `coint_johansen()` | Interpretación |
|---|---|---|
| `ecdet = "none"` | `det_order = -1` | Relación de cointegración sin constante. |
| `ecdet = "const"` | `det_order = 0` | Relación de cointegración con constante. |
| `ecdet = "trend"` | `det_order = 1` | Relación de cointegración con tendencia lineal. |

Las tres formas pueden pensarse así:

$$
P.Brent_t - \beta P.WTI_t = 0,
$$

$$
P.Brent_t - \beta P.WTI_t + c = 0,
$$

$$
P.Brent_t - \beta P.WTI_t + c + \delta t = 0.
$$

El argumento `k_ar_diff` indica el número de rezagos en diferencias del VECM. Como el script estimó previamente un VAR(3), se usa:

```python
k_ar_diff = p_var - 1
```

Por tanto:

```python
k_ar_diff = 2
```

Este es un punto donde Python y R se escriben distinto. En R, con `ca.jo()`, se usa `K = p_var`, porque `K` se refiere al orden del VAR en niveles. En Python, con `coint_johansen()`, se usa `k_ar_diff = p_var - 1`, porque `k_ar_diff` se refiere al orden del VECM en diferencias.

### 7. Cuidado: `coint_johansen` y `ca.jo` no son completamente equivalentes

Este punto es muy importante para la lectura del script.

Aunque `coint_johansen()` de `statsmodels` y `ca.jo()` de `urca` implementan pruebas de Johansen, sus resultados pueden diferir. No necesariamente van a producir los mismos estadísticos ni los mismos valores críticos.

En particular:

- los valores críticos pueden cambiar entre implementaciones;
- los estadísticos pueden no coincidir exactamente;
- las especificaciones determinísticas no son idénticas en todos los detalles;
- la forma de indicar los rezagos es distinta: `K` en R frente a `k_ar_diff` en Python;
- en Python, un mismo objeto `JohansenTestResult` guarda tanto el estadístico de traza como el de valor propio máximo.

Por eso, no se debe leer `coint_johansen()` como si fuera una copia exacta de `ca.jo()`.

En el script ocurre algo pedagógicamente valioso: con `det_order = -1`, es decir, sin constante en la relación de cointegración, Python puede sugerir que no hay cointegración, mientras que la versión de R con `ca.jo()` sí puede llevar a una conclusión distinta. Con `det_order = 0`, que permite constante en la relación de cointegración, los resultados de Python son más cercanos a los de R y se conserva la conclusión de una relación de cointegración.

La recomendación práctica del script es trabajar principalmente con:

```python
det_order = 0
```

en Python, y con:

```r
ecdet = "const"
```

en R, porque son las especificaciones más parecidas entre ambos entornos y suelen ser las más usadas en aplicaciones.

### 8. Test de Johansen en Python

El script calcula dos versiones principales del test:

```python
johansen_test_none = coint_johansen(
    Y_modelo,
    det_order=-1,
    k_ar_diff=k_ar_diff,
)
```

y:

```python
johansen_test_const = coint_johansen(
    Y_modelo,
    det_order=0,
    k_ar_diff=k_ar_diff,
)
```

En `statsmodels`, el resultado de `coint_johansen()` contiene tanto el criterio del valor propio máximo como el criterio de la traza. El script usa la función auxiliar `imprimir_tabla_johansen()` para mostrar ambos de forma ordenada.

Con dos variables, el procedimiento secuencial se resume así:

- primero se contrasta \(H_0: r = 0\);
- luego se contrasta \(H_0: r = 1\);
- si se rechaza \(r = 0\) pero no se rechaza \(r = 1\), se concluye que hay una relación de cointegración.

En el caso con constante, el script concluye que existe una relación de cointegración entre Brent y WTI:

$$
r = 1.
$$

### 9. Estimación del VECM(2)

Una vez determinado el rango de cointegración, el script estima el VECM con la clase `VECM` de `statsmodels`.

En `statsmodels`, la especificación determinística del VECM se controla con el argumento `deterministic`. Algunas opciones importantes son:

| Opción | Interpretación |
|---|---|
| `"n"` | Sin términos determinísticos. |
| `"co"` | Constante fuera de la relación de cointegración. |
| `"ci"` | Constante dentro de la relación de cointegración. |
| `"lo"` | Tendencia lineal fuera de la relación de cointegración. |
| `"li"` | Tendencia lineal dentro de la relación de cointegración. |

El script estima una especificación de referencia y luego se concentra en el modelo principal de la sesión: el VECM con constante dentro de la relación de cointegración.

```python
VEC_const = VECM(
    Y_modelo,
    k_ar_diff=k_ar_diff,
    coint_rank=1,
    deterministic="ci",
)

VEC_const_fit = VEC_const.fit()
```

Aquí:

- `k_ar_diff=2` indica que se estima un VECM(2);
- `coint_rank=1` indica que hay una relación de cointegración;
- `deterministic="ci"` indica que la constante está dentro de la relación de cointegración.

La función auxiliar `extraer_matrices_vecm()` organiza los objetos más importantes:

```python
matrices_vecm_const = extraer_matrices_vecm(
    VEC_const_fit,
    variables=variables,
)
```

La matriz \(\beta\) contiene el vector de cointegración. En términos intuitivos, representa la combinación de Brent y WTI que debería comportarse como una relación estable de largo plazo.

La matriz \(\alpha\) contiene las velocidades de ajuste. Estos coeficientes indican qué variables reaccionan cuando el sistema se aleja del equilibrio de largo plazo.

La relación:

$$
\beta'Y_{t-1}
$$

representa el error de equilibrio del periodo anterior, mientras que \(\alpha\) indica cómo ese error entra en la dinámica de \(\Delta Y_t\).

### 10. Tendencia lineal

En R se usa `lttest()` del paquete `urca` para evaluar tendencia lineal en el VAR en niveles asociado a la reparametrización del VECM. En Python, `statsmodels` no ofrece un equivalente directo de `lttest()`.

Por eso, el script hace una revisión práctica estimando una especificación con tendencia:

```python
VEC_tendencia = VECM(
    Y_modelo,
    k_ar_diff=k_ar_diff,
    coint_rank=1,
    deterministic="cilo",
)
```

La conclusión del script es que no se encuentra evidencia práctica para incluir tendencia lineal. Por tanto, el modelo principal se mantiene como VECM con constante en la relación de cointegración, sin tendencia lineal.

### 11. Reparametrización del VECM como VAR en niveles

En R, después de estimar el VECM, se puede usar `vec2var()` para obtener un objeto VAR en niveles. En Python no se crea un objeto equivalente a `vec2var`.

En `statsmodels`, la representación VAR en niveles queda guardada dentro del resultado estimado, específicamente en:

```python
VEC_const_fit.var_rep
```

El script usa:

```python
imprimir_matrices_var_reparametrizado(
    VEC_const_fit,
    variables=variables,
)
```

para mostrar las matrices \(A_i\) del VAR en niveles asociado al VECM. Esto es importante porque muchos usos prácticos del modelo, como diagnósticos, pronósticos e impulso-respuesta, se entienden mejor a partir de esta representación.

### 12. Validación de supuestos

El script valida tres aspectos del modelo:

| Supuesto | Herramienta usada en Python |
|---|---|
| No autocorrelación serial | `VEC_const_fit.test_whiteness()` |
| Heterocedasticidad tipo ARCH | `prueba_arch_por_ecuacion()` |
| Normalidad multivariada | `VEC_const_fit.test_normality()` |
| Normalidad por ecuación | `prueba_normalidad_por_ecuacion()` |

La validación no es un paso decorativo. Un modelo puede encontrar cointegración y aun así tener residuales problemáticos. Por eso se revisa si los errores se comportan razonablemente como innovaciones.

En el ejemplo se cumple el supuesto más importante para esta aplicación: no se detecta autocorrelación serial en los residuales. Sin embargo, sí se rechazan los supuestos de homocedasticidad y normalidad. Por eso, la recomendación del script es usar intervalos de confianza por bootstrap para realizar inferencia más cuidadosa en pronósticos e impulso-respuesta.

Un detalle de implementación: `statsmodels` no tiene un equivalente directo al `arch.test()` multivariado de R para objetos VAR/VECM. Por eso, el script auxiliar construye pruebas ARCH univariadas por ecuación.

### 13. Pronóstico del VECM

Después de estimar el VECM, el script produce pronósticos a 12 meses:

```python
horizonte_pronostico = 12
int_conf_pronostico = 0.95
```

La predicción se calcula con la función auxiliar:

```python
pronostico_VECM = predecir_vecm(
    VEC_const_fit,
    n_ahead=horizonte_pronostico,
    ci=int_conf_pronostico,
    indice=Y.index,
    variables=variables,
)
```

Luego se grafican los pronósticos y una versión tipo fanchart. La interpretación debe hacerse recordando que el modelo no solo extrapola la dinámica reciente. También incorpora la relación de largo plazo estimada entre Brent y WTI. Esa es una ventaja importante del VECM frente a un VAR en diferencias cuando las variables están cointegradas.

### 14. Funciones impulso-respuesta

El script calcula funciones impulso-respuesta ortogonalizadas con:

```python
irf_ortog_vecm = graficar_grilla_irf(
    VEC_const_fit,
    variables,
    pasos_adelante,
    ortog=True,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso ortogonal",
    semilla=semilla_irf,
    runs=repeticiones_bootstrap_irf,
)
```

Las funciones impulso-respuesta muestran cómo responde una variable del sistema ante un choque en otra variable. En este caso, permiten estudiar preguntas como:

- ¿Cómo responde Brent ante un choque en WTI?
- ¿Cómo responde WTI ante un choque en Brent?
- ¿La respuesta es estadísticamente significativa?
- ¿El efecto se disipa o permanece en el tiempo?

El script usa:

```python
pasos_adelante = np.arange(0, 19)
repeticiones_bootstrap_irf = 100
```

La función auxiliar calcula bandas de confianza por bootstrap para las IRF. Esto es importante porque los supuestos de normalidad y homocedasticidad no se cumplen perfectamente en los residuales.

Como se usan respuestas ortogonalizadas, el orden de las variables importa. El script define inicialmente:

```python
variables = ["P.Brent", "P.WTI"]
```

Bajo una identificación recursiva tipo Cholesky, la primera variable se interpreta como más contemporáneamente exógena que la segunda. En el orden original, Brent se trata como la variable más exógena y WTI como la más endógena.

Con este orden, el script muestra una interpretación sustantiva de las OIRF: un choque estructural en Brent afecta tanto al precio Brent como al precio WTI, mientras que un choque estructural en WTI no tiene un efecto relevante sobre Brent y su efecto sobre WTI tiende a disiparse hacia cero en el largo plazo.

### 15. Orden de las variables

La última sección del script muestra qué ocurre si se cambia el orden del sistema:

```python
variables_alt = ["P.WTI", "P.Brent"]
Y_alt = Y[variables_alt]
```

Después se repite la metodología con el orden alternativo:

- se estima un VAR(3);
- se verifica no autocorrelación serial;
- se aplica `coint_johansen()` con `det_order = 0`;
- se estima un VECM(2) con `deterministic="ci"`;
- se calculan OIRF.

El resultado central es que las conclusiones básicas de cointegración y diagnóstico pueden mantenerse similares, pero las funciones impulso-respuesta ortogonalizadas pueden cambiar de forma importante.

Este punto es fundamental: el orden de las variables no es un detalle técnico menor. En OIRF, el orden representa un supuesto económico sobre qué variable puede reaccionar contemporáneamente a cuál.

Por eso, antes de interpretar impulso-respuesta como evidencia económica, se debe justificar el orden elegido. En el script, poner Brent primero equivale a tratarlo como la variable más exógena dentro del sistema. Al invertir el orden y poner WTI primero, las OIRF cambian sustancialmente: los choques estructurales de WTI pasan a verse más significativos y persistentes.

La conclusión pedagógica es fuerte: el orden de las variables en una identificación por Cholesky no es trivial. Si el orden de exogeneidad no está claro por teoría económica o por evidencia estadística, las conclusiones de impulso-respuesta pueden depender demasiado del supuesto de identificación. El script menciona que una alternativa más estructural para enfrentar este problema son los S-VECM, donde se imponen restricciones económicas explícitas para identificar choques estructurales.

## Papel de `funciones_auxiliares_VECM.py`

El archivo `funciones_auxiliares_VECM.py` contiene funciones auxiliares para facilitar el trabajo visual, organizar salidas y evitar repetir código. Incluye, entre otras, funciones para:

- configurar la visualización de tablas y gráficas;
- imprimir resultados de pruebas ADF;
- imprimir tablas de selección de rezagos;
- organizar resultados del test de Johansen;
- extraer matrices \(\alpha\) y \(\beta\) del VECM;
- imprimir matrices del VAR reparametrizado;
- graficar series de tiempo;
- graficar diagnósticos de residuales;
- calcular pronósticos del VECM;
- graficar pronósticos y fancharts;
- calcular y graficar funciones impulso-respuesta;
- construir bandas bootstrap para las IRF.

Este archivo debe entenderse como apoyo operativo. No es necesario que los estudiantes entiendan cada función interna para seguir la clase. Tampoco se recomienda modificarlo mientras se estudia el script principal, porque cambiarlo puede alterar las gráficas o producir errores difíciles de rastrear.

La forma correcta de usarlo es simplemente importarlo desde el script principal:

```python
from funciones_auxiliares_VECM import ...
```

Después de importarlo, sus funciones quedan disponibles para el resto del análisis.

## Qué deberían aprender al finalizar

Al terminar esta sesión, la meta no es memorizar todos los comandos de Python. La meta es entender la secuencia de razonamiento econométrico:

1. No toda serie no estacionaria debe diferenciarse sin pensar.
2. Si varias series son \(I(1)\), puede existir una relación estable de largo plazo.
3. La metodología de Johansen permite determinar el número de relaciones de cointegración.
4. Un VECM combina dinámica de corto plazo con equilibrio de largo plazo.
5. Las matrices \(\alpha\) y \(\beta\) son el corazón de la interpretación.
6. En Python, `k_ar_diff` se refiere al número de rezagos en diferencias del VECM.
7. `coint_johansen()` y `ca.jo()` no son completamente equivalentes, por lo que sus resultados deben leerse con cuidado.
8. La validación de residuales sigue siendo necesaria.
9. Los pronósticos y las funciones impulso-respuesta deben interpretarse con cuidado.
10. En OIRF, el orden de las variables es un supuesto económico, no una decisión automática de Python.

## Recomendación final para estudiantes

Lean `Modelos_VECM_y_Johansen.py` como una historia econométrica. Primero se observa el comportamiento de los datos, luego se verifica su orden de integración, después se pregunta si existe cointegración, y finalmente se estima un modelo que combina largo plazo y corto plazo.

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

Si esa idea queda clara, los comandos de Python se vuelven herramientas para implementar una metodología, no una lista aislada de instrucciones.
