# Simulación de la distribución normal multivariada en R

## Introducción

Este directorio contiene el material en R de la sesión dedicada a la **distribución normal multivariada**, con énfasis en la simulación de una normal bivariada correlacionada.

La idea central del repositorio es mostrar, paso a paso, cómo se puede pasar de variables normales estándar independientes a una distribución normal multivariada con una estructura de varianzas, covarianzas y correlaciones definida por el usuario.

En términos matemáticos, el objetivo principal es construir una variable aleatoria bivariada:

$$
U \sim N_2(\mu,\Sigma),
$$

a partir de una normal estándar bivariada no correlacionada:

$$
Z \sim N_2(0,I_2).
$$

Para lograrlo, el script busca una matriz \(P\) tal que:

$$
P P' = \Sigma.
$$

Luego transforma las simulaciones estándar mediante:

$$
U = ZP' + \mu.
$$

Esta transformación es el corazón conceptual del ejercicio: la matriz \(P\) introduce la escala, la covarianza y la correlación deseadas.

## Archivos principales

En esta carpeta encontrarán dos scripts importantes:

| Archivo | Rol dentro del repositorio |
|---|---|
| `simulacion_normal_multivariada.R` | Script principal. Es el archivo que deben leer, ejecutar y estudiar con cuidado. Allí se desarrolla la simulación de la normal multivariada y se generan las comparaciones y visualizaciones. |
| `funciones_auxiliares_distribux_normal_multivariada.R` | Script auxiliar. Contiene funciones de apoyo para comparar resultados, construir gráficas y exportar archivos HTML. No es necesario que lo entiendan en detalle ni que lo modifiquen. Solo deben usarlo. |

También encontrarán carpetas con archivos HTML generados por el script. Estas contienen visualizaciones interactivas de las distribuciones simuladas.

## Documento teórico complementario

Además de los scripts, el material cuenta con el archivo:

```text
../simulax_normal_multivariada.pdf
```

Este PDF complementa teóricamente los archivos computacionales. Su propósito es explicar, desde el álgebra matricial, por qué la simulación de una normal multivariada correlacionada puede construirse a partir de normales estándar independientes.

La idea principal del documento es la siguiente. Si se parte de un vector aleatorio estándar:

$$
\varepsilon_t \sim N_p(0,I_p),
$$

y se define una transformación lineal:

$$
U_t = P\varepsilon_t,
$$

entonces:

$$
\operatorname{Var}(U_t)
= \operatorname{Var}(P\varepsilon_t)
= P\operatorname{Var}(\varepsilon_t)P'
= PI_pP'
= PP'.
$$

Por tanto, para simular una normal multivariada con matriz de covarianzas \(\Sigma_u\), necesitamos encontrar una matriz \(P\) tal que:

$$
PP' = \Sigma_u.
$$

El PDF enfatiza que \(P\) puede entenderse como una **raíz matricial** de \(\Sigma_u\). Esa matriz contiene la estructura de varianzas, covarianzas y correlaciones que se quiere imponer sobre los errores simulados. Las descomposiciones de Cholesky, espectral y SVD son tres formas distintas de construir esa matriz \(P\).

## Objetivo del script `simulacion_normal_multivariada.R`

El objetivo de `simulacion_normal_multivariada.R` es explicar cómo simular una distribución normal bivariada correlacionada usando tres formas distintas de descomponer la matriz de varianzas y covarianzas:

1. Descomposición espectral.
2. Descomposición en valores singulares, SVD.
3. Descomposición de Cholesky.

Todas estas técnicas tienen la misma finalidad dentro del script: encontrar una matriz que permita transformar una normal estándar no correlacionada en una normal multivariada con matriz de covarianzas \(\Sigma\).

La pregunta de fondo es:

> ¿Cómo podemos generar datos simulados que tengan medias, varianzas y correlaciones específicas?

El script responde esa pregunta construyendo primero datos simples, independientes y estandarizados, y después aplicando una transformación lineal que les impone la estructura de dependencia deseada.

## Idea general de la simulación

El flujo conceptual del script puede resumirse así:

1. Se simulan dos variables normales estándar independientes.
2. Esas variables se organizan en una matriz \(Z\), donde cada fila representa una observación de una normal bivariada estándar.
3. Se define un vector de medias \(\mu\).
4. Se construye una matriz de varianzas y covarianzas \(\Sigma\).
5. Se obtiene una matriz de transformación a partir de \(\Sigma\).
6. Se transforma \(Z\) para obtener una normal bivariada correlacionada.
7. Se compara la simulación manual con la simulación producida por `mvtnorm::rmvnorm()`.
8. Se generan gráficas interactivas en 2D y 3D.
9. Se exportan las gráficas como archivos HTML.

La lógica estadística es la siguiente:

$$
Z \sim N_2(0,I_2)
$$

y si existe una matriz \(P\) tal que:

$$
P P' = \Sigma,
$$

entonces:

$$
U = ZP' + \mu
$$

tiene distribución:

$$
U \sim N_2(\mu,\Sigma).
$$

## Parámetros principales de la simulación

El script fija una semilla para que los resultados sean reproducibles:

```r
set.seed(82901)
```

Luego define el número de observaciones:

```r
n_observaciones = 100000
```

Usar una muestra grande permite que las medias, varianzas, covarianzas y correlaciones muestrales se acerquen bastante a sus valores teóricos.

El vector de medias usado es:

```r
media = c(u_1 = 0, u_2 = 0)
```

Por tanto, la distribución objetivo está centrada en cero:

$$
\mu =
\begin{pmatrix}
0 \\
0
\end{pmatrix}.
$$

Después se definen las desviaciones estándar:

```r
desv_estandar = c(u_1 = 1.0, u_2 = 1.5)
```

y la correlación teórica:

```r
rho = 0.90
```

Esto significa que se quiere simular una normal bivariada donde \(u_1\) y \(u_2\) tienen una relación lineal positiva fuerte.

## Construcción de la matriz \(\Sigma\)

La matriz de varianzas y covarianzas se construye como:

$$
\Sigma =
\begin{pmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2 \\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}.
$$

En el script:

$$
\sigma_1 = 1.0, \qquad \sigma_2 = 1.5, \qquad \rho = 0.90.
$$

Por tanto:

$$
\Sigma =
\begin{pmatrix}
1.00 & 1.35 \\
1.35 & 2.25
\end{pmatrix}.
$$

Esta matriz resume toda la estructura de dispersión y dependencia que se quiere imponer sobre la distribución simulada.

## Simulación de la normal estándar bivariada

El primer paso práctico consiste en generar normales estándar independientes:

$$
z_1 \sim N(0,1),
\qquad
z_2 \sim N(0,1).
$$

Luego se juntan en una matriz:

$$
Z =
\begin{pmatrix}
z_{11} & z_{12} \\
z_{21} & z_{22} \\
\vdots & \vdots \\
z_{n1} & z_{n2}
\end{pmatrix}.
$$

Cada fila de \(Z\) representa una observación simulada de una normal bivariada estándar no correlacionada:

$$
Z \sim N_2(0,I_2).
$$

Esta parte es clave porque sirve como punto de partida: antes de crear correlación, el script genera datos que todavía no tienen la estructura de dependencia deseada.

## Métodos de descomposición usados

El script muestra tres maneras de obtener una matriz de transformación a partir de \(\Sigma\). La idea teórica, desarrollada en `simulax_normal_multivariada.pdf`, es que todas buscan una matriz \(P\) que cumpla:

$$
PP' = \Sigma.
$$

Por eso, aunque las matrices \(P\) se construyan de formas distintas, las tres descomposiciones permiten simular la misma distribución objetivo:

$$
U \sim N_2(\mu,\Sigma).
$$

### 1. Descomposición espectral

La descomposición espectral se basa en valores y vectores propios. Para una matriz simétrica como \(\Sigma\), se puede escribir:

$$
\Sigma = Q\Lambda Q',
$$

donde \(Q\) contiene los vectores propios y \(\Lambda\) contiene los valores propios.

El PDF interpreta esta descomposición como un cambio de base. La matriz \(Q\) es ortogonal, es decir:

$$
Q^{-1} = Q',
$$

y sus columnas forman una base de vectores propios. La matriz \(\Lambda\) es diagonal y guarda los valores propios de \(\Sigma\). En esa base, la matriz de covarianzas queda representada de forma diagonal, lo que facilita tomar su raíz.

La matriz de transformación se obtiene usando la raíz cuadrada de los valores propios:

$$
P = Q\Lambda^{1/2}.
$$

Conceptualmente, este método rota los datos hacia la base de vectores propios, aplica las escalas dadas por \(\Lambda^{1/2}\), y con ello reproduce la estructura de covarianzas deseada.

### 2. Descomposición SVD

La descomposición en valores singulares escribe una matriz como:

$$
\Sigma = UDV'.
$$

Aquí \(D\) es una matriz diagonal cuyos elementos son los valores singulares, mientras que \(U\) y \(V\) son matrices ortogonales asociadas a las direcciones singulares izquierda y derecha.

En este caso, también se usa una raíz de la matriz para construir una transformación que reproduzca la matriz de covarianzas objetivo:

$$
P = UD^{1/2}.
$$

Para matrices de covarianzas simétricas y semidefinidas positivas, la SVD está muy relacionada con la descomposición espectral. En la práctica, el script la incluye para mostrar que hay más de una manera computacional de llegar a la misma normal multivariada.

### 3. Descomposición de Cholesky

La descomposición de Cholesky expresa la matriz de covarianzas como el producto de una matriz triangular por su transpuesta:

$$
\Sigma = LL'.
$$

En este caso, la matriz de transformación puede tomarse como:

$$
P = L.
$$

Esta es una técnica muy usada en simulación porque permite generar variables correlacionadas de manera eficiente. El PDF la destaca como especialmente importante por su interpretación práctica: \(L\) es una matriz triangular, y el orden de las variables puede importar. En contextos econométricos, como modelos VAR estructurales, ese orden puede representar restricciones o supuestos de identificación.

Por eso, Cholesky suele ser la descomposición más operativa para implementar simulaciones o imponer una estructura recursiva, mientras que la descomposición espectral y la SVD ayudan a entender la geometría matricial detrás del problema.

## Simulación manual frente a `mvtnorm`

Una de las partes más importantes del script es la comparación entre:

- La simulación manual hecha paso a paso.
- La simulación realizada con `mvtnorm::rmvnorm()`.

El paquete `mvtnorm` es una herramienta especializada para trabajar con distribuciones normales multivariadas. El script lo usa como referencia computacional.

Para que la comparación sea justa, el código hace que `mvtnorm::rmvnorm()` utilice las mismas normales estándar que se generaron manualmente. Así se puede comparar observación por observación y no solo de forma aproximada.

La idea es verificar que:

$$
U_{\text{manual}} \approx U_{\text{mvtnorm}}.
$$

Cuando la diferencia máxima entre ambos resultados es prácticamente cero, se confirma que la simulación manual reproduce correctamente lo que hace internamente una función especializada.

## Qué deben observar en los resultados

Al ejecutar el script, conviene prestar atención a tres tipos de resultados.

### 1. Correlación de la normal estándar

Al inicio, la matriz \(Z\) debe tener una correlación muestral cercana a cero entre sus columnas. Esto confirma que se generaron dos normales estándar independientes.

La matriz de correlación esperada es aproximadamente:

$$
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}.
$$

### 2. Matriz \(\Sigma\) y correlación teórica

Después, el script imprime la matriz de covarianzas teórica y la matriz de correlaciones asociada. Allí debe observarse que la correlación objetivo entre \(u_1\) y \(u_2\) es \(0.90\).

### 3. Comparación manual vs. `mvtnorm`

La tabla de comparación muestra si los resultados manuales y los de `mvtnorm` coinciden numéricamente para cada método:

- Descomposición espectral.
- SVD.
- Cholesky.

Esta comparación es el chequeo computacional más importante del script.

## Visualizaciones generadas

El script produce gráficas interactivas en dos formatos.

### Gráficas 2D

Las gráficas 2D muestran nubes de puntos simulados.

La normal estándar bivariada no correlacionada debe verse aproximadamente circular, porque ambas variables tienen varianza uno y correlación cercana a cero.

La normal bivariada correlacionada debe verse como una nube inclinada y alargada. Esa inclinación aparece porque:

$$
\rho = 0.90.
$$

Es decir, valores altos de \(u_1\) tienden a venir acompañados por valores altos de \(u_2\).

### Gráficas 3D

Las gráficas 3D muestran la función de densidad de la normal bivariada.

En estas figuras, la altura representa la densidad:

$$
f(u_1,u_2).
$$

La normal estándar no correlacionada tiene una forma simétrica alrededor del cero. En cambio, la normal correlacionada tiene una base más alargada, de acuerdo con la estructura de \(\Sigma\).

## Archivos HTML

Al final, el script exporta las gráficas a archivos HTML dentro de la carpeta `html_nm`.

Estos archivos pueden abrirse directamente desde un navegador. Esto es útil porque permite explorar las figuras de manera interactiva, incluso después de cerrar R o RStudio.

Los archivos generados son:

| Archivo | Contenido |
|---|---|
| `01_std_2d.html` | Normal estándar bivariada en 2D. |
| `02_eig_2d.html` | Normal correlacionada simulada con descomposición espectral en 2D. |
| `03_svd_2d.html` | Normal correlacionada simulada con SVD en 2D. |
| `04_chol_2d.html` | Normal correlacionada simulada con Cholesky en 2D. |
| `05_std_3d.html` | Densidad 3D de la normal estándar bivariada. |
| `06_eig_3d.html` | Densidad 3D asociada a la simulación por descomposición espectral. |
| `07_svd_3d.html` | Densidad 3D asociada a la simulación por SVD. |
| `08_chol_3d.html` | Densidad 3D asociada a la simulación por Cholesky. |

## Sobre el script auxiliar

El archivo `funciones_auxiliares_distribux_normal_multivariada.R` contiene funciones que ayudan a:

- Comparar la simulación manual con `mvtnorm`.
- Imprimir resúmenes explicativos en consola.
- Preparar muestras para graficar.
- Construir gráficas 2D con `plotly`.
- Construir superficies de densidad 3D.
- Exportar las gráficas como archivos HTML.

Este archivo está pensado como soporte técnico del script principal. No es el foco conceptual de la sesión.

Para esta clase, la recomendación es:

> No modificar `funciones_auxiliares_distribux_normal_multivariada.R`.

Los estudiantes solo necesitan saber que el script principal lo carga mediante `source()` para usar funciones ya preparadas. Esto permite que `simulacion_normal_multivariada.R` sea más legible y se concentre en la explicación estadística.

## Recomendación de lectura para estudiantes

Para estudiar este material, se recomienda seguir este orden:

1. Abrir `../simulax_normal_multivariada.pdf` para revisar la intuición teórica.
2. Abrir `simulacion_normal_multivariada.R`.
3. Leer primero los comentarios generales y la tabla de contenidos.
4. Ejecutar el script por bloques, no todo de una vez al comienzo.
5. Revisar la matriz \(Z\) y su correlación muestral.
6. Revisar cómo se construye \(\Sigma\).
7. Entender la transformación \(U = ZP' + \mu\).
8. Comparar los tres métodos de descomposición.
9. Observar la tabla manual vs. `mvtnorm`.
10. Abrir los archivos HTML generados en `html_nm`.

La parte más importante no es memorizar cada línea de código, sino entender la idea estadística detrás de la simulación:

> una normal multivariada correlacionada puede construirse transformando linealmente una normal estándar no correlacionada.

## Conceptos clave de la sesión

Al finalizar la revisión del script, deberían quedar claros los siguientes conceptos:

- Qué significa simular una normal bivariada.
- Qué representa el vector de medias \(\mu\).
- Qué representa la matriz de covarianzas \(\Sigma\).
- Cómo se relacionan varianza, covarianza y correlación.
- Por qué una matriz de covarianzas define la geometría de una distribución multivariada.
- Cómo una transformación lineal puede introducir correlación entre variables.
- Para qué sirven la descomposición espectral, la SVD y Cholesky en simulación.
- Cómo validar una simulación comparándola con una función especializada.
- Cómo interpretar nubes de puntos y superficies de densidad en distribuciones normales bivariadas.

## Cierre

Este material busca conectar la teoría de la distribución normal multivariada con una implementación computacional concreta en R.

El script `simulacion_normal_multivariada.R` debe leerse como una guía pedagógica: primero construye el caso más simple, luego introduce la matriz de covarianzas, después aplica distintas descomposiciones matriciales y finalmente verifica y visualiza los resultados.

La lección principal es que la dependencia entre variables aleatorias puede modelarse y simularse de forma ordenada usando álgebra matricial. En este caso, toda la estructura de la distribución queda resumida en:

$$
U \sim N_2(\mu,\Sigma).
$$
