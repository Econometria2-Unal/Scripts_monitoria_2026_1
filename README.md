# Scripts - Monitoría de Econometría II

Repositorio oficial de la monitoría de **Econometría II** para el semestre 2026-1.

Aquí encontrarán los códigos, datos y materiales de apoyo que usaremos durante el curso. El repositorio está organizado por sesiones temáticas y contiene material tanto en **R** como en **Python**.

> La referencia operativa del curso será el **main branch**. Esa es la versión que deben consultar para trabajar con los códigos actualizados.

---

## Herramientas de trabajo

Durante el curso trabajaremos con dos ecosistemas de programación:

| Lenguaje | Herramientas principales | Uso en el curso |
|---|---|---|
| **R** | [R](https://www.r-project.org/) + [RStudio](https://posit.co/download/rstudio-desktop/) | Análisis econométrico, series de tiempo, modelos VAR/VECM y scripts docentes en R. |
| **Python** | [Anaconda](https://www.anaconda.com/download) + [Visual Studio Code](https://code.visualstudio.com/) | Ejecución de scripts `.py`, trabajo con datos, simulaciones, gráficos y modelos econométricos. |

En Python trabajaremos dentro de **ambientes virtuales de conda**. Esto permite que todos usemos una configuración de paquetes similar y reduce errores al ejecutar los scripts del repositorio.

---

## Estructura general del repositorio

La organización principal del repositorio es la siguiente:

```text
.
|-- econometria2.yml
|-- README.md
|-- Sesión 1 - Introducción a python/
|-- Sesión 2 - Simulación AR, MA y ARMA/
|-- Sesión 3 - Metodología Box Jenkins/
|-- Sesión 6 - Distribux Normal Multi/
|-- Sesión 7 - Modelos VAR/
`-- Sesión 8 - Modelos VECM/
```

Algunas carpetas pueden contener archivos generados automáticamente por RStudio o Python, como `.Rproj.user`, `.Rhistory`, `.RData` o `__pycache__`. Estos archivos no son el foco conceptual del curso; los materiales importantes son los scripts, datos, README internos, archivos `.Rproj`, archivos `.py`, archivos `.R` y documentos complementarios.

---

## Contenido por subdirectorio

| Carpeta | Tema general | Contenido principal |
|---|---|---|
| `Sesión 1 - Introducción a python/` | Introducción al trabajo con Python. | Carpeta reservada para material introductorio. En el `main branch` actual no contiene archivos visibles. |
| `Sesión 2 - Simulación AR, MA y ARMA/` | Simulación de procesos univariados de series de tiempo. | Dos scripts en Python que simulan ruido blanco, procesos AR, MA, ARMA y casos ARIMA estacionarios/no estacionarios. Se trabajan gráficos de series, FAC, FACP, diferenciación y uso de `statsmodels`. |
| `Sesión 3 - Metodología Box Jenkins/` | Modelación ARIMA aplicada con metodología Box-Jenkins. | Proyecto de RStudio, scripts en `codigo/R`, scripts en `codigo/python` y bases en `datos`. Incluye aplicaciones con exportaciones tradicionales y precio internacional del té. |
| `Sesión 6 - Distribux Normal Multi/` | Simulación de la distribución normal multivariada. | Material en R y Python para construir una normal multivariada a partir de normales estándar independientes. Incluye scripts principales, funciones auxiliares, README internos, visualizaciones HTML y un PDF teórico complementario. |
| `Sesión 7 - Modelos VAR/` | Modelos de vectores autorregresivos, VAR. | Scripts en R y Python, funciones auxiliares, README internos y datos `ENDERS.xlsx`. Se estudian VAR con series simuladas y un ejemplo aplicado con variables macroeconómicas. |
| `Sesión 8 - Modelos VECM/` | Cointegración, metodología de Johansen y modelos VECM. | Scripts en R y Python, funciones auxiliares, README internos y datos `Petróleo.xlsx`. Se trabaja con precios Brent y WTI, pruebas de raíz unitaria, cointegración y modelos de corrección del error. |

---

## Guía de lectura por sesiones

### Sesión 1 - Introducción a Python

Esta carpeta está creada como espacio de trabajo para material introductorio. En la versión actual del `main branch` no contiene scripts ni datos.

### Sesión 2 - Simulación AR, MA y ARMA

Esta sesión introduce la simulación de procesos de series de tiempo. Los scripts muestran cómo generar procesos como:

- ruido blanco;
- procesos autorregresivos AR;
- procesos de medias móviles MA;
- procesos ARMA;
- procesos ARIMA no estacionarios y sus diferencias.

La idea central es conectar la teoría de procesos estocásticos con su comportamiento visual y estadístico. Por eso los scripts enfatizan gráficos de series, funciones de autocorrelación, funciones de autocorrelación parcial y comparaciones entre procesos estacionarios y no estacionarios.

### Sesión 3 - Metodología Box-Jenkins

Esta carpeta contiene versiones en R y Python de ejercicios aplicados de modelación ARIMA. La metodología Box-Jenkins se trabaja como una secuencia:

```text
identificación -> estimación -> diagnóstico -> pronóstico
```

Los datos principales están en `datos/`:

| Archivo | Contenido |
|---|---|
| `Expotradicionales1990-2017.csv` | Serie de exportaciones tradicionales. |
| `PTEAUSDM2005-202506.csv` | Serie del precio internacional del té. |

Los scripts replican el flujo típico de trabajo: cargar datos, construir índices temporales, revisar gráficos, analizar FAC/FACP, aplicar pruebas de estacionariedad, estimar modelos ARIMA/SARIMAX y diagnosticar residuales.

### Sesión 6 - Distribución normal multivariada

Esta sesión muestra cómo simular una normal multivariada usando álgebra matricial. La idea estadística central es partir de una normal estándar no correlacionada,

```math
Z \sim N_p(0, I_p),
```

y transformarla para obtener una variable aleatoria con media y matriz de covarianzas deseadas:

```math
U = ZP' + \mu, \qquad PP' = \Sigma.
```

La carpeta contiene:

| Ruta | Contenido |
|---|---|
| `codigo/python/` | Script principal, funciones auxiliares, README interno y gráficos HTML interactivos. |
| `codigo/R/` | Versión en R del mismo ejercicio y visualizaciones HTML. |
| `codigo/simulax_normal_multivariada.pdf` | Documento teórico complementario sobre la construcción matricial de la simulación. |

Se comparan distintas formas de obtener la matriz \(P\), como descomposición espectral, SVD y Cholesky.

### Sesión 7 - Modelos VAR

Esta sesión estudia modelos VAR, útiles para analizar sistemas donde varias variables se explican conjuntamente por sus propios rezagos y por los rezagos de las demás variables. En términos generales, un VAR(\(p\)) puede escribirse como:

```math
Y_t = A_0 + A_1Y_{t-1} + A_2Y_{t-2} + \cdots + A_pY_{t-p} + u_t.
```

La carpeta contiene:

| Ruta | Contenido |
|---|---|
| `codigo/python/` | Scripts de series simuladas, ejemplo aplicado de Enders, funciones auxiliares y README interno. |
| `codigo/R/` | Versión en R de los modelos VAR y README interno. |
| `datos/ENDERS.xlsx` | Base de datos macroeconómica usada en el ejemplo aplicado. |

La sesión combina un laboratorio controlado con series simuladas y un caso aplicado con datos macroeconómicos. Se trabajan selección de rezagos, estabilidad, diagnóstico de residuales, pronósticos, funciones impulso-respuesta y descomposición de varianza del error de pronóstico.

### Sesión 8 - Modelos VECM

Esta sesión aborda cointegración, metodología de Johansen y modelos de corrección del error vectorial. El punto de partida es que algunas series pueden ser no estacionarias, pero tener una relación estable de largo plazo.

Un VECM puede escribirse como:

```math
\begin{aligned}
\Delta Y_t
&= \Pi Y_{t-1}
+ \Gamma_1 \Delta Y_{t-1}
+ \cdots \\
&\quad + \Gamma_{p-1}\Delta Y_{t-p+1}
+ u_t.
\end{aligned}
```

Cuando existe cointegración,

```math
\Pi = \alpha\beta',
```

donde \(\beta\) contiene las relaciones de equilibrio de largo plazo y \(\alpha\) contiene las velocidades de ajuste.

La carpeta contiene:

| Ruta | Contenido |
|---|---|
| `codigo/python/` | Script principal de VECM/Johansen, funciones auxiliares y README interno. |
| `codigo/R/` | Versión en R del ejercicio y README interno. |
| `datos/Petróleo.xlsx` | Base de precios mensuales del petróleo Brent y WTI. |

El objetivo es aprender a revisar orden de integración, aplicar la prueba de Johansen, estimar un VECM, validar residuales, generar pronósticos e interpretar funciones impulso-respuesta.

---

## Consejos para trabajar en Python

### 1. Usar el ambiente de conda del repositorio

El repositorio incluye el archivo:

```text
econometria2.yml
```

Este archivo define un ambiente virtual de conda llamado `econometria2`. Su objetivo es instalar los paquetes necesarios para ejecutar los scripts de Python del curso.

Contenido declarado en el YAML:

```yaml
name: econometria2
channels:
  - defaults
dependencies:
  - python=3.14
  - matplotlib
  - openpyxl
  - numpy
  - plotly
  - seaborn
  - pandas
  - nbformat
  - statsmodels
  - ipykernel
prefix: C:\Users\gcrp9\anaconda3\envs\econometria2
```

> Nota: Conda puede instalar dependencias adicionales que no aparecen explícitamente en el YAML. Además, la línea `prefix` refleja la ruta local donde fue creado originalmente el ambiente; en otros computadores puede cambiar sin problema.

### 2. Paquetes incluidos en `econometria2.yml`

| Paquete | Para que se usa en el curso | Enlace oficial |
|---|---|---|
| `python=3.14` | Lenguaje base para ejecutar todos los scripts `.py`. | [python.org](https://www.python.org/) |
| `matplotlib` | Gráficos estáticos de series de tiempo, residuales, FAC/FACP, pronósticos e IRF. | [matplotlib.org](https://matplotlib.org/) |
| `openpyxl` | Lectura y escritura de archivos Excel `.xlsx`, usados en bases como `ENDERS.xlsx` o `Petróleo.xlsx`. | [openpyxl.readthedocs.io](https://openpyxl.readthedocs.io/) |
| `numpy` | Cálculo numérico, simulación, álgebra lineal, vectores, matrices y generación de datos aleatorios. | [numpy.org](https://numpy.org/) |
| `plotly` | Gráficos interactivos, especialmente visualizaciones HTML en 2D y 3D. | [plotly.com/python](https://plotly.com/python/) |
| `seaborn` | Gráficos estadísticos con una sintaxis cómoda y estilos visuales limpios. | [seaborn.pydata.org](https://seaborn.pydata.org/) |
| `pandas` | Manejo de bases de datos, series de tiempo, índices temporales, tablas y transformaciones. | [pandas.pydata.org](https://pandas.pydata.org/) |
| `nbformat` | Soporte para trabajar con notebooks y formatos usados por Jupyter. | [nbformat.readthedocs.io](https://nbformat.readthedocs.io/) |
| `statsmodels` | Modelos econométricos y de series de tiempo: ARIMA, SARIMAX, VAR, VECM, pruebas ADF, diagnósticos e IRF. | [statsmodels.org](https://www.statsmodels.org/) |
| `ipykernel` | Permite registrar el ambiente de conda como kernel de Python para Jupyter y VSCode. | [ipykernel.readthedocs.io](https://ipykernel.readthedocs.io/) |

### 3. Importar el ambiente `econometria2`

La forma recomendada es usar **Anaconda Prompt**.

Primero, ubíquense en la carpeta raíz del repositorio:

```bash
cd ruta\al\repositorio
```

Luego creen el ambiente desde el archivo YAML:

```bash
conda env create -f econometria2.yml
```

Activen el ambiente:

```bash
conda activate econometria2
```

Verifiquen que el ambiente quedó instalado:

```bash
conda env list
```

Opcionalmente, registren el ambiente como kernel para Jupyter/VSCode:

```bash
python -m ipykernel install --user --name econometria2 --display-name "Python (econometria2)"
```

Si necesitan actualizar el ambiente después de cambios en el YAML:

```bash
conda env update -f econometria2.yml --prune
```

### 4. Configurar Python en VSCode

Guía completa: [Configurar Python en VSCode](https://germankux.notion.site/Configurar-Python-en-VSCode-2fdb3071946680f59a65ce539b225a7f).

Resumen recomendado para el curso:

1. Abran VSCode e inicien sesión con su usuario de GitHub.
2. Instalen las extensiones **Python** y **Jupyter** desde el panel de extensiones de VSCode.
3. Abran en VSCode la carpeta del repositorio o la carpeta de trabajo donde guardarán sus scripts.
4. Usen la paleta de comandos con `Ctrl + Shift + P`.
5. Busquen `Python: Select Interpreter`.
6. Seleccionen el intérprete asociado al ambiente de conda `econometria2`.
7. Para ejecutar código por bloques o selección de líneas, activen el modo interactivo de Jupyter en VSCode.
8. En la configuración de VSCode, busquen `Jupyter Interactive Window` y activen la opción relacionada con ejecutar selección en la ventana interactiva.
9. Guarden los archivos `.py` dentro de la carpeta abierta en VSCode.
10. Ejecuten selecciones de código con `Ctrl + Shift + Enter`.

La idea es que VSCode use el intérprete correcto. Si VSCode está apuntando a otro Python, es posible que algunos paquetes no aparezcan instalados aunque ya existan dentro de `econometria2`.

### 5. Exportar e importar ambientes de conda

Guía completa: [Exportación e Importación de Ambientes Virtuales de Conda](https://germankux.notion.site/Exportaci-n-e-Importaci-n-de-Ambientes-Virtuales-de-Conda-38eb307194668092af1ceb9c4750836d).

Resumen de la lógica de trabajo:

```bash
conda env list
conda activate myenv
```

Para exportar un ambiente existen tres opciones comunes:

```bash
conda env export --from-history > myenv.yml
```

Más portable, pero menos reproducible. Guarda principalmente los paquetes instalados de forma explícita.

```bash
conda env export --no-builds > myenv.yml
```

Punto intermedio entre portabilidad y reproducibilidad.

```bash
conda env export > myenv.yml
```

Más reproducible, pero menos portable entre computadores o sistemas operativos diferentes.

Para importar un ambiente desde un archivo YAML:

```bash
conda env create -f ruta_directorio\myenv.yml
conda activate myenv
```

En este curso, el caso concreto es:

```bash
conda env create -f econometria2.yml
conda activate econometria2
```

---

## Material de YouTube

Playlist recomendada: [Git, GitHub, Anaconda y Python en VSCode](https://www.youtube.com/playlist?list=PL6saL77GH5h2wqRghNGMup6uhk09AkR5r).

Esta playlist está pensada para que aprendan a usar **Git** y **GitHub**, y también para que puedan configurar **Anaconda** y **Python en VSCode**. Les servirá especialmente si quieren clonar el repositorio, mantenerlo actualizado, trabajar de forma ordenada y evitar problemas con ambientes de Python.

---

## Recomendaciones finales

- Revisen siempre el contenido del **main branch**.
- Lean primero los README internos cuando existan dentro de cada sesión.
- Ejecuten los scripts por bloques, especialmente los archivos `.py` que usan celdas `# %%`.
- No modifiquen funciones auxiliares si apenas están estudiando la sesión; úsenlas como apoyo.
- Mantengan activo el ambiente `econometria2` antes de correr scripts de Python.
- En R, abran los proyectos `.Rproj` con RStudio cuando la carpeta de la sesión los incluya.

La meta del repositorio no es solo correr código. La meta es que cada script sirva como puente entre la teoría econométrica y la práctica computacional.
