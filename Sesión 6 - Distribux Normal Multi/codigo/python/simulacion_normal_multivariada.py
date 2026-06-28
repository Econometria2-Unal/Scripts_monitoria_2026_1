# %% Importacion de paquetes ---

#' Universidad Nacional de Colombia
#' Facultad de Ciencias Economicas
#'
#' Econometria II | Monitoria
#' Sesion 7: Modelos de vectores autorregresivos
#'
#' Simulacion de una normal multivariada (bivariada) correlacionada
#' usando descomposicion espectral, SVD y Cholesky.

# NOTA MUY IMPORTANTE: Para entender el Script, se sugiere que se lea a la par tanto
#                      el Script, como las notas teoricas en PDF que acompanan el script

# ===
# Tabla de contenidos ===
# ===

#' Simulacion:
  #' 1. Simulacion de una normal estandar bivariada no correlacionada
  #' 2. Simulacion de una normal bivariada correlacionada
    #' 2.1 Generacion de la matriz de varianzas-covarianzas de la distribucion normal bivariada correlacionada
    #' 2.2 Simulacion de una normal bivariada correlacionada de manera manual y por NumPy
      #' 2.2.1 Simulacion manual: descomposicion espectral, SVD y Cholesky
      #' 2.2.2 Simulacion con NumPy: descomposicion espectral, SVD y Cholesky
#'
#' Comparacion de los resultados de la simulacion manual vs simulacion con NumPy
#'Graficacion de la distribuciones normales multivariadas obtenidas:
  #'  Graficas 2D interactivas con plotly
  #'  Graficas 3D interactivas de densidad con plotly
  #'  Exportacion de graficas a archivos HTML


# Paquetes necesarios para este script:
# - numpy: simulacion de normales multivariadas y algebra lineal.
# - pandas: tablas con nombres de filas y columnas.
# - scipy: densidad de la normal multivariada.
# - plotly: graficas interactivas.

from importlib.util import find_spec
from pathlib import Path
import sys
import warnings


paquetes_necesarios = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "plotly": "plotly",
}

paquetes_faltantes = [
    paquete for paquete, modulo in paquetes_necesarios.items()
    if find_spec(modulo) is None
]

if len(paquetes_faltantes) > 0:
    paquetes_pip = " ".join(paquetes_faltantes)
    raise ImportError(
        "Faltan paquetes por instalar: "
        + ", ".join(paquetes_faltantes)
        + f"\nInstalalos con: pip install {paquetes_pip}"
    )

import numpy as np
import pandas as pd


# %% Manejo de rutas relativas ---

def encontrar_directorio_proyecto(ruta_script_relativa):
    candidatos = []

    try:
        ruta_script_actual = Path(__file__).resolve()
        candidatos.append(ruta_script_actual.parent)
        candidatos.extend(ruta_script_actual.parents)
    except NameError:
        pass

    candidatos.append(Path.cwd())
    candidatos.extend(Path.cwd().parents)

    for candidato in candidatos:
        if (candidato / ruta_script_relativa).exists():
            return candidato

    raise FileNotFoundError(
        "No se pudo ubicar la raiz del proyecto. Revisa la ruta relativa "
        "del script actual."
    )


# Fijar la ruta del archivo actual como referencia para las rutas relativas.
ruta_script_relativa = Path(
    "Sesión 6 - Distribux Normal Multi"
) / "codigo" / "python" / "simulacion_normal_multivariada.py"

# Directorios principales del proyecto.
directorio_proyecto = encontrar_directorio_proyecto(ruta_script_relativa)
directorio_sesion_normal_multivariada = Path(
    "Sesión 6 - Distribux Normal Multi"
)
directorio_codigo_python = (
    directorio_sesion_normal_multivariada
    / "codigo"
    / "python"
)

# Importacion de funciones auxiliares para comparacion, graficacion y exportacion.
ruta_funciones_auxiliares_normal_multivariada = (
    directorio_proyecto
    / directorio_codigo_python
    / "funciones_auxiliares_distribux_normal_multivariada.py"
)

if str(ruta_funciones_auxiliares_normal_multivariada.parent) not in sys.path:
    sys.path.insert(0, str(ruta_funciones_auxiliares_normal_multivariada.parent))

from funciones_auxiliares_distribux_normal_multivariada import (
    configurar_rutas_exportacion,
    crear_graficas_2d_normal_multivariada,
    crear_graficas_3d_normal_multivariada,
    crear_tabla_comparacion_numpy,
    definir_directorio_graficas,
    exportar_graficas_html,
    imprimir_descripcion_exportacion_html,
    imprimir_descripcion_graficas_2d,
    imprimir_descripcion_graficas_3d,
    imprimir_resumen_comparacion_numpy,
    mostrar_graficas_interactivo,
)

configurar_rutas_exportacion(
    directorio_proyecto=directorio_proyecto,
    directorio_codigo_python=directorio_codigo_python,
)


# Fijamos la semilla para que la simulacion sea reproducible.
semilla_simulacion = 82901
generador_simulacion = np.random.default_rng(semilla_simulacion)

# Numero de observaciones
n_observaciones = 100000 # Entre mas datos sean simulados, los resultados
                         # computacionales convergeran mejor a los teoricos

# Vector de medias de la distribucion normal bivariada correlacionada.
media = pd.Series([0.0, 0.0], index=["u_1", "u_2"], name="media")


def matriz_con_nombres(matriz, nombres):
    return pd.DataFrame(matriz, index=nombres, columns=nombres)


def cov2cor(Sigma):
    Sigma = np.asarray(Sigma, dtype=float)
    desviaciones = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(desviaciones, desviaciones)


# ===
# Simulacion ====
# ===

# %% 1. Simulacion de una normal estandar bivariada no correlacionada ----

# El primer paso para simular una distribucion normal bivariada correlacionada
# es simular una distribucion normal estandar bivariada no correlacionada

# Note: Tenga en cuenta que simular de una distribucion normal p-variada
#       no correlacionada es equivalente a simular p distribuciones normales
#       univariadas independientes. En distribuciones normales multivariadas
#       que no haya correlacion entre sus V.A. componentes, implica necesiamente
#       que sus V.A. componentes son independientes.

# Primero simulamos dos errores normales estandar univariados independientes:
#
#   z_1 ~ N(0, 1)
#   z_2 ~ N(0, 1)
#
# Nota: Al juntarlos en una matriz (que sera la que guarde la informacion de la
# distribucion normal multivariada) Z = (z_1, z_2), cada fila es una observacion de
# una normal estandar bivariada no correlacionada: Z ~ N_2(0, I_2), es decir,
# las columnas de Z (la distribucion normal bivariada) seran los resultados de
# simular z_1 y z_2, las distribuciones normales univariadas estandar.

# Simulo las dos distribuciones normales estandar independientes univariadas
normales_estandar = generador_simulacion.normal(size=n_observaciones * 2)

# Construyo la matriz Z, que sera la simulacion de la distribucion normal
# bivariada estandar no correlacionada
Z = normales_estandar.reshape(n_observaciones, 2)

# Le doy nombres a las columnas de Z
Z = pd.DataFrame(Z, columns=["z_1", "z_2"])

print("\nCorrelacion muestral de los errores estandar no correlacionados:")
print(matriz_con_nombres(np.corrcoef(Z.to_numpy(), rowvar=False), Z.columns))
# Note que la matriz de varianzas y covarianzas de Z es muy cercana
# a la matriz identidad 2x2


# %% 2. Simulacion de una normal bivariada correlacionada ----

# El segundo paso, es simular la distribucion normal bivariada correlacionada a partir de la
# distribucion normal bivariada estandar no correlacionada que se simulo en el paso anterior.

# ===
# 2.1 Generacion de la matriz de varianzas-covarianzas de la distribucion normal bivariada correlacionada ----
# ===

# Lo primero que se debe hacer para simular la distribucion normal bivariada correlacionada,
# es especificar la matriz de varianzas-covarianzas (Sigma) de esta distribucion

# Usaremos la misma matriz Sigma para las tres descomposiciones que veremos:
  # 1. Descomposicion Espectral
  # 2. SVD
  # 3. Descomposicion de Cholesky

# Las desviaciones estandar de cada una de las V.A. que componen la distribucion
# normal bivariada correlacionada son:
desv_estandar = pd.Series([1.0, 1.5], index=["u_1", "u_2"], name="sd")

# Correlacion teorica entre las V.A. que conforman la distribucion
# normal bivariada correlacionada es:
rho = 0.90

# Construccion de la matriz de varianzas y covarianzas (Sigma) de la distribucion
# normal bivariada correlacionada
Sigma = pd.DataFrame(
    [
        [
            desv_estandar["u_1"] ** 2, # Varianza de u_1
            rho * np.prod(desv_estandar), # Covarianza entre u_1 y u_2, cov(u_1, u_2) = rho * std(u_1) * std(u_2)
        ],
        [
            rho * np.prod(desv_estandar),
            desv_estandar["u_2"] ** 2, # Varianza de u_2
        ],
    ],
    index=media.index,
    columns=media.index,
)

print("\nMatriz de varianzas-covarianzas teorica Sigma:")
print(Sigma)

print("\nMatriz de correlaciones teorica asociada a Sigma:")
print(matriz_con_nombres(cov2cor(Sigma), Sigma.columns))


# ===
# 2.2 Simulacion de una normal bivariada correlacionada de manera manual y por NumPy ----
# ===

# %% 2.2.1 Simulacion manual: Para descomposicion espectral, SVD y Cholesky ----

# Idea de la simulacion manual:
#
# Si Z ~ N_2(0, I_2) y encontramos una matriz P tal que:
#
#   P @ P.T = Sigma, # Todas las descomposiciones de Sigma, consisten
#                    # en encontrar la matriz P, que permita hacer dicha descomposicion
#
# entonces:
#
#   U = Z @ P.T + media # Nota: Se multiplica P.T porque los vectores son filas
#                       # En las notas del PDF, se multiplica por P, porque los vectores son columnas
#
# tiene distribucion normal bivariada con matriz de covarianzas Sigma.
#
# La funcion siguiente replica las matrices de descomposicion P usadas internamente por
# numpy.random.Generator.multivariate_normal() para los metodos "eigh", "svd" y "cholesky".
# Asi la comparacion manual vs. NumPy se puede hacer observacion por observacion.

# 1. Se encuentra la matriz P, que se obtiene la descomposicion de Sigma

# Nota: Para entender estas funcion, leer las notas sobre los diferentes tipos de
#       descomposicion que hay de la matriz Sigma

def descomposicion_sigma(Sigma, metodo):
    Sigma = np.asarray(Sigma, dtype=float)
    tolerancia = np.sqrt(np.finfo(float).eps)

    # Para realizar la Descomposicion Espectral de Sigma
    if metodo == "espectral":

        # Encuentra los valores y vectores propios de Sigma
        valores, vectores = np.linalg.eigh(Sigma)

        if not np.all(valores >= -tolerancia * abs(valores[-1])):
            warnings.warn(
                "Sigma no es positiva semidefinida numericamente.",
                RuntimeWarning,
                stacklevel=2,
            )

        # La matriz P = Q * Lambda^{1/2}
        matriz_P_trans = (vectores * np.sqrt(np.maximum(valores, 0.0))).T

    # Para realizar la SVD
    elif metodo == "svd":

        # Encuentra los valores y vectores singulares a izquierda y derecha de Sigma
        u, valores, vh = np.linalg.svd(Sigma)

        if not np.all(valores >= -tolerancia * abs(valores[0])):
            warnings.warn(
                "Sigma no es positiva semidefinida numericamente.",
                RuntimeWarning,
                stacklevel=2,
            )

        # La matriz P = U * D^{1/2}
        matriz_P_trans = (u * np.sqrt(np.maximum(valores, 0.0))).T

    # Para realizar la Descomposicion de Cholesky
    elif metodo == "cholesky":

        # Realiza la Descomposicion de Cholesky de Sigma
        cholesky = np.linalg.cholesky(Sigma)

        # La matriz P = L, donde L es la matriz triangular inferior de Cholesky
        matriz_P_trans = cholesky.T

    else:
        raise ValueError("Metodo no reconocido. Usa: espectral, svd o cholesky.")

    # La funcion retorna la matriz P.T
    return matriz_P_trans


# 2. Se simula la distribucion normal bivariada correlacionada (U)
#    A partir de la distribucion normal bivariada estandar no correlacionada (Z)
#    y la matriz de descomposicion (matriz_P)

def simulacion_manual_distribucion_normal_multiv_correlac(Z, media, Sigma, metodo):

    # Nota: U sera la matriz que contiene los resultados de simular la distribucion normal
    #       bivariada correlacionada

    # Matriz que se obtiene de la descomposicion de la matriz Sigma
    matriz_P_trans = descomposicion_sigma(Sigma, metodo)

    # Se simula la distribucion normal bivariada correlacionada (U), usando la distribucion
    # normal bivariada estandar no correlacionada (Z) y la matriz de descomposicion (matriz_P)
    U = np.asarray(Z, dtype=float) @ matriz_P_trans

    # Sumarle la media a U
    U = U + media.to_numpy()

    # Se ponen los nombres de las V.A. que conforman la distribucion normal bivariada correlacionada
    U = pd.DataFrame(U, columns=media.index)

    # La funcion retorna la matriz U asociada con la distribucion normal bivariada
    # correlacionada
    return U


# Simulacion manual de la distribucion normal multivariada correlacionada usando:

# 1. Descomposicion Espectral (Sigma = Q * Lambda * Q^{'})
U_manual_espectral = simulacion_manual_distribucion_normal_multiv_correlac(
    Z,
    media,
    Sigma,
    "espectral",
)

# 2. SVD (Sigma = U * D * V^{'})
U_manual_svd = simulacion_manual_distribucion_normal_multiv_correlac(
    Z,
    media,
    Sigma,
    "svd",
)

# 3. Descomposicion de Cholesky (Sigma = L * L^{'})
U_manual_cholesky = simulacion_manual_distribucion_normal_multiv_correlac(
    Z,
    media,
    Sigma,
    "cholesky",
)


# %% 2.2.2 Simulacion con NumPy: descomposicion espectral, SVD y Cholesky ----

# Nota: Para simular, numpy.random.Generator.multivariate_normal() normalmente
# genera sus propias normales estandar. Pero, para efectos de comparacion con
# la simulacion manual, aqui reiniciamos un generador con la misma semilla usada
# para construir la matriz Z. Asi se usa la misma secuencia de normales estandar.

def simulacion_numpy_distribucion_normal_multiv_correlac(n, media, Sigma, metodo):
    metodos_numpy = {
        "espectral": "eigh",
        "svd": "svd",
        "cholesky": "cholesky",
    }

    if metodo not in metodos_numpy:
        raise ValueError("Metodo no reconocido. Usa: espectral, svd o cholesky.")

    generador = np.random.default_rng(semilla_simulacion)
    U = generador.multivariate_normal(
        mean=np.asarray(media, dtype=float),
        cov=np.asarray(Sigma, dtype=float),
        size=n,
        method=metodos_numpy[metodo],
    )

    U = pd.DataFrame(U, columns=media.index)

    return U


# Simulacion usando NumPy de la distribucion normal multivariada
# correlacionada usando:

# 1. Descomposicion Espectral (Sigma = Q * Lambda * Q^{'})
U_numpy_espectral = simulacion_numpy_distribucion_normal_multiv_correlac(
    n=n_observaciones,
    media=media,
    Sigma=Sigma,
    metodo="espectral",
)

# 2. SVD (Sigma = U * D * V^{'})
U_numpy_svd = simulacion_numpy_distribucion_normal_multiv_correlac(
    n=n_observaciones,
    media=media,
    Sigma=Sigma,
    metodo="svd",
)

# 3. Descomposicion de Cholesky (Sigma = L * L^{'})
U_numpy_cholesky = simulacion_numpy_distribucion_normal_multiv_correlac(
    n=n_observaciones,
    media=media,
    Sigma=Sigma,
    metodo="cholesky",
)


# %% 6. Comparacion de los resultados de la simulacion manual vs simulacion con NumPy ----

# Nota: En esta seccion se verifica que la simulacion manual y NumPy producen
# exactamente los mismos datos cuando ambas simulaciones emplean mismas distribuciones
# normales estandar. Este es el chequeo computacional mas importante del
# script: muestra que las diferentes descomposiciones de la matriz Sigma
# realizadas manualmente generan la misma distribucion normal multivariada correlacionada
# que genera la funcion especializada numpy.random.Generator.multivariate_normal().

# Tabla de comparacion entre las diferentes distribuciones multivariadas correlacionadas
# generadas
tabla_comparacion = crear_tabla_comparacion_numpy(
    U_manual_espectral=U_manual_espectral,
    U_numpy_espectral=U_numpy_espectral,
    U_manual_svd=U_manual_svd,
    U_numpy_svd=U_numpy_svd,
    U_manual_cholesky=U_manual_cholesky,
    U_numpy_cholesky=U_numpy_cholesky,
)

# Imprime un resumen comparativo de los resultados de las diferentes simulaciones
imprimir_resumen_comparacion_numpy(
    tabla_comparacion=tabla_comparacion,
    U_manual_espectral=U_manual_espectral,
    Sigma=Sigma,
)


# %% Graficacion de la distribuciones normales multivariadas obtenidas ----

# ===
# Graficas 2D interactivas de plotly ----
# ===

# Nota: Las grafica 2D muestra la nube de puntos de las diferentes observaciones
# o datos simuladoas a partir de las diferentes distribuciones normales multivariadas

# Las graficas 2D permiten ver la geometria de las distribuciones normales multivariadas
# Simuladas. La distribucion normal estandar bivariada no correlacionada debe verse
# proyectada en los plano casi circular; la normal multivariada correlacionada proyecta
# en el plano debe verse como una nube inclinada y alargada. Esa inclinacion
# se debe a la correlacion positiva de 0.9 entre u_1 y u_2.
n_puntos_grafica = 8000
semilla_graficos = 202601

# Imprime una descripcion grafica de los graficos 2D simulados
imprimir_descripcion_graficas_2d(
    n_observaciones=n_observaciones,
    n_puntos_grafica=n_puntos_grafica,
)

# Se generan las graficas 2D de las distribuciones normal multivariada
graficas_2d = crear_graficas_2d_normal_multivariada(
    Z=Z,
    U_manual_espectral=U_manual_espectral,
    U_manual_svd=U_manual_svd,
    U_manual_cholesky=U_manual_cholesky,
    n_puntos=n_puntos_grafica,
    semilla_graficos=semilla_graficos,
)

# Las graficas 2D generadas para los diferentes tipos de distribucion normal multivariadas simuladas:

g_normal_estandar = graficas_2d["normal_estandar"] # Normal multivariada estandar no correlacionada
g_espectral = graficas_2d["espectral"] # Normal multivariada correlacionada (usando desc. espectral)
g_svd = graficas_2d["svd"] # Normal multivariada correlacionada (usando SVD)
g_cholesky = graficas_2d["cholesky"] # Normal multivariada correlacionada (usando Chokesky)


# %% Graficas 3D interactivas de plotly ----

# Nota: Las grafica 3D muestran las funciones de densidad asociadas a
#       las diferentes distribuciones normales multivariadas simuladas

# Las graficas 3D no muestran puntos simulados: muestran la funcion de densidad
# teorica evaluada sobre una grilla. Es otra forma de visualizar y entender la
# distribucion normal multivariada simulada.

# Imprime una descripcion grafica de los graficos 3D simulados
imprimir_descripcion_graficas_3d()

# Se generan las graficas 3D de las distribuciones normal multivariada
graficas_3d = crear_graficas_3d_normal_multivariada(
    media=media,
    Sigma=Sigma,
)

# Las graficas 3D generadas para los diferentes tipos de distribucion normal multivariadas simuladas:

g_normal_estandar_3d = graficas_3d["normal_estandar_3d"] # Normal multivariada estandar no correlacionada
g_espectral_3d = graficas_3d["espectral_3d"] # Normal multivariada correlacionada (usando desc. espectral)
g_svd_3d = graficas_3d["svd_3d"] # Normal multivariada correlacionada (usando SVD)
g_cholesky_3d = graficas_3d["cholesky_3d"] # Normal multivariada correlacionada (usando Chokesky)


# %% Exportacion de graficas a archivos HTML ----

# Nota: Recomendacion, ir a la carpeta "html_nm" y
#       abrir los archivos HTML de las graficas para visualizarlas desde el navegador

# Finalmente, se exportan las graficas a HTML. Asi, las figuras
# pueden abrirse en cualquier navegador y compartirse sin depender de que la
# sesion de Python siga abierta.

# Se define el directorio donde se almacenaran las graficas HTML
directorio_graficas = definir_directorio_graficas(nombre_directorio="html_nm")

# Imprime una descripcion de la exportacion de los HTML
imprimir_descripcion_exportacion_html(
    directorio_graficas=directorio_graficas,
)

# Se genera un diccionario que contiene las graficas que se van a exportar
# a HTML
graficas_html = {
    "01_std_2d.html": g_normal_estandar,
    "02_eig_2d.html": g_espectral,
    "03_svd_2d.html": g_svd,
    "04_chol_2d.html": g_cholesky,
    "05_std_3d.html": g_normal_estandar_3d,
    "06_eig_3d.html": g_espectral_3d,
    "07_svd_3d.html": g_svd_3d,
    "08_chol_3d.html": g_cholesky_3d,
}

# Se exportan las graficas en formato HTML para poder abrir desde el navegador
exportar_graficas_html(
    graficas=graficas_html,
    directorio_graficas=directorio_graficas,
)

# Si se ejecuta el script interactivamente, tambien se muestran las graficas
# en el visor disponible. Al usar python desde consola, se guardan en HTML sin abrir ventanas.
mostrar_graficas_interactivo(graficas_html)

# %%
