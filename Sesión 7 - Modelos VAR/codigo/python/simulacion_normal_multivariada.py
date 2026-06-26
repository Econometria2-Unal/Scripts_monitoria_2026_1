# %% 1. Preparacion del entorno ============================

#' Universidad Nacional de Colombia
#' Facultad de Ciencias Economicas
#'
#' Econometria II | Monitoria
#' Sesion 3: Metodologia Box Jenkins
#'
#' Simulacion de una normal multivariada correlacionada
#' usando descomposicion espectral, SVD y Cholesky.


# ===
# Tabla de contenidos ===
# ===

#' 1. Preparacion del entorno
#' 2. Simulacion de una normal estandar bivariada no correlacionada
#' 3. Matriz de varianzas-covarianzas con alta correlacion
#' 4. Simulacion manual: espectral, SVD y Cholesky
#' 5. Simulacion auxiliar tipo mvtnorm: espectral, SVD y Cholesky
#' 6. Comparacion de resultados manuales y auxiliares
#' 7. Graficas 2D interactivas con plotly
#' 8. Graficas 3D interactivas de densidad con plotly
#' 9. Exportacion de graficas a archivos HTML


# Paquetes necesarios para este script:
# - numpy: simulacion y algebra lineal.
# - pandas: tablas con nombres de filas y columnas.
# - scipy: densidad de la normal multivariada.
# - plotly: graficas interactivas y exportacion a HTML.
from importlib.util import find_spec
from pathlib import Path
import shutil
import sys
import tempfile
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
import plotly.graph_objects as go
from scipy.stats import multivariate_normal


# Fijamos la semilla para que la simulacion sea reproducible.
semilla_simulacion = 82901
generador_simulacion = np.random.default_rng(semilla_simulacion)

# Numero de observaciones solicitado.
n_observaciones = 100000

# Media de la normal bivariada correlacionada.
media = pd.Series([0.0, 0.0], index=["u_1", "u_2"], name="media")


def matriz_con_nombres(matriz, nombres):
    """Convierte una matriz de numpy en DataFrame con nombres."""
    return pd.DataFrame(matriz, index=nombres, columns=nombres)


# %% 2. Simulacion de una normal estandar bivariada no correlacionada ============================

# Primero simulamos dos errores normales estandar univariados independientes:
#
#   z_1 ~ N(0, 1)
#   z_2 ~ N(0, 1)
#
# Al juntarlos en una matriz Z = (z_1, z_2), cada fila es una observacion de
# una normal estandar bivariada no correlacionada: Z ~ N_2(0, I_2).
normales_estandar = generador_simulacion.normal(size=n_observaciones * 2)
Z = normales_estandar.reshape(n_observaciones, 2)
Z = pd.DataFrame(Z, columns=["z_1", "z_2"])

print("\nCorrelacion muestral de los errores estandar no correlacionados:")
print(matriz_con_nombres(np.corrcoef(Z.to_numpy(), rowvar=False), Z.columns))


# %% 3. Matriz de varianzas-covarianzas con alta correlacion ============================

# Usaremos la misma matriz Sigma para las tres descomposiciones. En este
# ejemplo las desviaciones estandar marginales son distintas, pero la
# correlacion teorica entre los errores es alta.
desv_estandar = pd.Series([1.0, 1.5], index=["u_1", "u_2"], name="sd")
rho = 0.90

Sigma = pd.DataFrame(
    [
        [
            desv_estandar["u_1"] ** 2,
            rho * np.prod(desv_estandar),
        ],
        [
            rho * np.prod(desv_estandar),
            desv_estandar["u_2"] ** 2,
        ],
    ],
    index=media.index,
    columns=media.index,
)


def cov2cor(Sigma):
    """Replica cov2cor() de R para matrices de varianzas-covarianzas."""
    Sigma = np.asarray(Sigma, dtype=float)
    desviaciones = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(desviaciones, desviaciones)


print("\nMatriz de varianzas-covarianzas teorica Sigma:")
print(Sigma)

print("\nMatriz de correlaciones teorica asociada a Sigma:")
print(matriz_con_nombres(cov2cor(Sigma), Sigma.columns))


# %% 4. Simulacion manual: espectral, SVD y Cholesky ============================

# Idea de la simulacion manual:
#
# Si Z ~ N_2(0, I_2) y encontramos una matriz R tal que:
#
#   R.T @ R = Sigma,
#
# entonces:
#
#   U = Z @ R + media
#
# tiene distribucion normal bivariada con matriz de covarianzas Sigma.
#
# La funcion siguiente replica la logica de los factores usados por
# mvtnorm::rmvnorm() para los metodos "eigen", "svd" y "chol". Asi la
# comparacion manual vs. una rutina auxiliar se puede hacer observacion por
# observacion.
def obtener_factor(Sigma, metodo):
    Sigma = np.asarray(Sigma, dtype=float)
    tolerancia = np.sqrt(np.finfo(float).eps)

    if metodo == "espectral":
        valores, vectores = np.linalg.eigh(Sigma)
        orden = np.argsort(valores)[::-1]
        valores = valores[orden]
        vectores = vectores[:, orden]

        if not np.all(valores >= -tolerancia * abs(valores[0])):
            warnings.warn(
                "Sigma no es positiva semidefinida numericamente.",
                RuntimeWarning,
                stacklevel=2,
            )

        factor = (
            vectores
            @ (vectores.T * np.sqrt(np.maximum(valores, 0.0))[:, np.newaxis])
        ).T

    elif metodo == "svd":
        u, valores, vh = np.linalg.svd(Sigma)
        v = vh.T

        if not np.all(valores >= -tolerancia * abs(valores[0])):
            warnings.warn(
                "Sigma no es positiva semidefinida numericamente.",
                RuntimeWarning,
                stacklevel=2,
            )

        factor = (
            v
            @ (u.T * np.sqrt(np.maximum(valores, 0.0))[:, np.newaxis])
        ).T

    elif metodo == "cholesky":
        # numpy.linalg.cholesky() devuelve L tal que L @ L.T = Sigma.
        # Para mantener U = Z @ R + media con R.T @ R = Sigma, usamos R = L.T.
        factor = np.linalg.cholesky(Sigma).T

    else:
        raise ValueError("Metodo no reconocido. Usa: espectral, svd o cholesky.")

    return factor


def simular_manual(Z, media, Sigma, metodo):
    factor = obtener_factor(Sigma, metodo)
    U = np.asarray(Z, dtype=float) @ factor
    U = U + media.to_numpy()
    U = pd.DataFrame(U, columns=media.index)
    return U


U_manual_espectral = simular_manual(Z, media, Sigma, "espectral")
U_manual_svd = simular_manual(Z, media, Sigma, "svd")
U_manual_cholesky = simular_manual(Z, media, Sigma, "cholesky")


# %% 5. Simulacion auxiliar tipo mvtnorm: espectral, SVD y Cholesky ============================

# En Python no hay un equivalente directo de mvtnorm::rmvnorm() con argumento
# method = "eigen", "svd" o "chol". Para conservar la comparacion del script
# original, definimos una rutina auxiliar que recibe el mismo vector de
# normales estandar usado para construir Z.
#
# Importante: el vector se reorganiza por filas, igual que en el argumento
# pre0.9_9994 = FALSE de mvtnorm::rmvnorm().
def crear_generador_desde_Z(Z):
    normales = np.asarray(Z, dtype=float).reshape(-1)

    def generador(n):
        if n != len(normales):
            raise ValueError(
                f"El generador fijo esperaba {len(normales)} normales."
            )

        return normales.copy()

    return generador


def simular_auxiliar_tipo_mvtnorm(n, media, Sigma, metodo, rnorm):
    normales = rnorm(n * len(media))
    Z_auxiliar = normales.reshape(n, len(media))
    return simular_manual(Z_auxiliar, media, Sigma, metodo)


U_auxiliar_espectral = simular_auxiliar_tipo_mvtnorm(
    n=n_observaciones,
    media=media,
    Sigma=Sigma,
    metodo="espectral",
    rnorm=crear_generador_desde_Z(Z),
)

U_auxiliar_svd = simular_auxiliar_tipo_mvtnorm(
    n=n_observaciones,
    media=media,
    Sigma=Sigma,
    metodo="svd",
    rnorm=crear_generador_desde_Z(Z),
)

U_auxiliar_cholesky = simular_auxiliar_tipo_mvtnorm(
    n=n_observaciones,
    media=media,
    Sigma=Sigma,
    metodo="cholesky",
    rnorm=crear_generador_desde_Z(Z),
)


# %% 6. Comparacion de resultados manuales y auxiliares ============================

def comparar_manual_auxiliar(metodo, U_manual, U_auxiliar):
    diferencia_maxima = np.max(
        np.abs(U_manual.to_numpy() - U_auxiliar.to_numpy())
    )
    son_iguales = np.allclose(
        U_manual.to_numpy(),
        U_auxiliar.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    resumen = pd.DataFrame(
        {
            "metodo": [metodo],
            "iguales_manual_auxiliar": [son_iguales],
            "diferencia_maxima": [diferencia_maxima],
            "media_u_1": [U_manual["u_1"].mean()],
            "media_u_2": [U_manual["u_2"].mean()],
            "var_u_1": [U_manual["u_1"].var(ddof=1)],
            "cov_u_1_u_2": [U_manual["u_1"].cov(U_manual["u_2"])],
            "var_u_2": [U_manual["u_2"].var(ddof=1)],
            "cor_u_1_u_2": [U_manual["u_1"].corr(U_manual["u_2"])],
        }
    )

    return resumen


tabla_comparacion = pd.concat(
    [
        comparar_manual_auxiliar(
            "Descomposicion espectral",
            U_manual_espectral,
            U_auxiliar_espectral,
        ),
        comparar_manual_auxiliar(
            "SVD",
            U_manual_svd,
            U_auxiliar_svd,
        ),
        comparar_manual_auxiliar(
            "Cholesky",
            U_manual_cholesky,
            U_auxiliar_cholesky,
        ),
    ],
    ignore_index=True,
)

print("\nComparacion manual vs. rutina auxiliar tipo mvtnorm:")
print(tabla_comparacion.to_string(index=False))

print("\nCovarianza muestral - metodo espectral manual:")
print(U_manual_espectral.cov())

print("\nCovarianza teorica objetivo Sigma:")
print(Sigma)


# %% 7. Graficas 2D interactivas con plotly ============================

# Con 100000 puntos, una grafica interactiva puede volverse pesada. Por eso
# simulamos todas las observaciones, pero graficamos una submuestra aleatoria
# suficientemente grande para ver la forma de la distribucion.
n_puntos_grafica = 8000
semilla_graficos = 202601


def preparar_muestra_grafica(U, n_puntos=n_puntos_grafica):
    generador_graficos = np.random.default_rng(semilla_graficos)
    U = np.asarray(U, dtype=float)
    indices = generador_graficos.choice(
        np.arange(U.shape[0]),
        size=min(n_puntos, U.shape[0]),
        replace=False,
    )
    datos = pd.DataFrame(
        {
            "u_1": U[indices, 0],
            "u_2": U[indices, 1],
        }
    )
    return datos


def graficar_normal_bivariada(
    U,
    titulo,
    color_puntos="#1f77b4",
    etiqueta_x="u_1",
    etiqueta_y="u_2",
):
    datos_grafica = preparar_muestra_grafica(U)

    grafica = go.Figure()

    grafica.add_trace(
        go.Histogram2dContour(
            x=datos_grafica["u_1"],
            y=datos_grafica["u_2"],
            colorscale="Viridis",
            reversescale=True,
            showscale=False,
            contours={
                "coloring": "lines",
                "showlabels": True,
            },
            hoverinfo="skip",
            name="Contornos",
        )
    )

    grafica.add_trace(
        go.Scattergl(
            x=datos_grafica["u_1"],
            y=datos_grafica["u_2"],
            mode="markers",
            marker={
                "size": 4,
                "color": color_puntos,
                "opacity": 0.35,
            },
            hovertemplate=(
                f"{etiqueta_x}: %{{x:.3f}}<br>"
                f"{etiqueta_y}: %{{y:.3f}}<extra></extra>"
            ),
            name="Observaciones simuladas",
        )
    )

    grafica.update_layout(
        title={"text": titulo},
        xaxis={"title": etiqueta_x},
        yaxis={"title": etiqueta_y, "scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h", "x": 0, "y": -0.15},
        margin={"l": 60, "r": 30, "b": 80, "t": 70},
    )

    return grafica


# Grafica interactiva de la normal estandar bivariada no correlacionada.
g_normal_estandar = graficar_normal_bivariada(
    Z,
    "Normal estandar bivariada no correlacionada",
    color_puntos="#2a6fbb",
    etiqueta_x="z_1",
    etiqueta_y="z_2",
)

# Graficas interactivas de las normales bivariadas correlacionadas obtenidas
# por cada descomposicion. Se grafica la version manual, porque arriba se
# verifica que coincide con la rutina auxiliar observacion por observacion.
g_espectral = graficar_normal_bivariada(
    U_manual_espectral,
    "Normal bivariada correlacionada - Descomposicion espectral",
    color_puntos="#c03a2b",
)

g_svd = graficar_normal_bivariada(
    U_manual_svd,
    "Normal bivariada correlacionada - SVD",
    color_puntos="#2b8c56",
)

g_cholesky = graficar_normal_bivariada(
    U_manual_cholesky,
    "Normal bivariada correlacionada - Cholesky",
    color_puntos="#6f4bb3",
)


# %% 8. Graficas 3D interactivas de densidad con plotly ============================

# Las graficas 3D muestran la funcion de densidad teorica:
#
#   f(u_1, u_2)
#
# evaluada sobre una grilla. Para la normal estandar usamos I_2; para las
# normales correlacionadas usamos la misma matriz Sigma definida arriba. Como
# los tres metodos producen la misma distribucion teorica, la superficie 3D
# objetivo es la misma, pero se guarda con el nombre de cada descomposicion
# para facilitar la explicacion en clase.
def crear_grilla_densidad(
    media,
    Sigma,
    n_grilla=80,
    multiplicador_sd=3.5,
):
    media = np.asarray(media, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    desv = np.sqrt(np.diag(Sigma))

    eje_x = np.linspace(
        media[0] - multiplicador_sd * desv[0],
        media[0] + multiplicador_sd * desv[0],
        n_grilla,
    )

    eje_y = np.linspace(
        media[1] - multiplicador_sd * desv[1],
        media[1] + multiplicador_sd * desv[1],
        n_grilla,
    )

    grilla_x, grilla_y = np.meshgrid(eje_x, eje_y)
    puntos = np.dstack((grilla_x, grilla_y))
    densidad = multivariate_normal(mean=media, cov=Sigma).pdf(puntos)

    return {
        "x": eje_x,
        "y": eje_y,
        "z": densidad,
    }


def graficar_densidad_3d(
    media,
    Sigma,
    titulo,
    etiqueta_x="u_1",
    etiqueta_y="u_2",
    escala_color="Viridis",
):
    grilla = crear_grilla_densidad(media=media, Sigma=Sigma)

    grafica = go.Figure(
        data=[
            go.Surface(
                x=grilla["x"],
                y=grilla["y"],
                z=grilla["z"],
                colorscale=escala_color,
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "highlightcolor": "#ffffff",
                        "project": {"z": True},
                    }
                },
                hovertemplate=(
                    f"{etiqueta_x}: %{{x:.3f}}<br>"
                    f"{etiqueta_y}: %{{y:.3f}}<br>"
                    "Densidad: %{z:.5f}<extra></extra>"
                ),
            )
        ]
    )

    grafica.update_layout(
        title={"text": titulo},
        scene={
            "xaxis": {"title": etiqueta_x},
            "yaxis": {"title": etiqueta_y},
            "zaxis": {"title": "Densidad"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": -1.65, "z": 1.15}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
    )

    return grafica


g_normal_estandar_3d = graficar_densidad_3d(
    media=np.array([0.0, 0.0]),
    Sigma=np.eye(2),
    titulo="Densidad 3D - Normal estandar bivariada",
    etiqueta_x="z_1",
    etiqueta_y="z_2",
    escala_color="Viridis",
)

g_espectral_3d = graficar_densidad_3d(
    media=media,
    Sigma=Sigma,
    titulo="Densidad 3D - Descomposicion espectral",
    escala_color="YlOrRd",
)

g_svd_3d = graficar_densidad_3d(
    media=media,
    Sigma=Sigma,
    titulo="Densidad 3D - SVD",
    escala_color="Plasma",
)

g_cholesky_3d = graficar_densidad_3d(
    media=media,
    Sigma=Sigma,
    titulo="Densidad 3D - Cholesky",
    escala_color="Portland",
)


# %% 9. Exportacion de graficas a archivos HTML ============================

# Guardamos todas las graficas como archivos HTML autocontenidos. Para evitar
# problemas con rutas largas o con caracteres especiales en Windows, primero se
# crea cada HTML en una carpeta temporal corta y luego se copia a la carpeta
# final del proyecto.
try:
    DIRECTORIO_SCRIPT = Path(__file__).resolve().parent
except NameError:
    DIRECTORIO_SCRIPT = Path.cwd()

directorio_graficas = DIRECTORIO_SCRIPT / "graficas_html_normal_multivariada"
directorio_graficas.mkdir(parents=True, exist_ok=True)


def guardar_grafica_html(grafica, nombre_archivo):
    ruta_archivo = directorio_graficas / nombre_archivo

    with tempfile.TemporaryDirectory() as carpeta_temporal:
        ruta_temporal = Path(carpeta_temporal) / nombre_archivo

        grafica.write_html(
            file=ruta_temporal,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
        )

        shutil.copy2(ruta_temporal, ruta_archivo)

    print(
        "Grafica guardada: "
        + ruta_archivo.resolve().as_posix()
    )


guardar_grafica_html(
    g_normal_estandar,
    "01_normal_estandar_bivariada_2d.html",
)

guardar_grafica_html(
    g_espectral,
    "02_normal_correlacionada_espectral_2d.html",
)

guardar_grafica_html(
    g_svd,
    "03_normal_correlacionada_svd_2d.html",
)

guardar_grafica_html(
    g_cholesky,
    "04_normal_correlacionada_cholesky_2d.html",
)

guardar_grafica_html(
    g_normal_estandar_3d,
    "05_normal_estandar_bivariada_3d.html",
)

guardar_grafica_html(
    g_espectral_3d,
    "06_normal_correlacionada_espectral_3d.html",
)

guardar_grafica_html(
    g_svd_3d,
    "07_normal_correlacionada_svd_3d.html",
)

guardar_grafica_html(
    g_cholesky_3d,
    "08_normal_correlacionada_cholesky_3d.html",
)

print(
    "\nDirectorio con graficas HTML:\n"
    + directorio_graficas.resolve().as_posix()
)

# Si se ejecuta el script interactivamente, tambien se muestran las graficas.
# Al usar python desde consola, se guardan en HTML sin abrir ventanas.
if hasattr(sys, "ps1"):
    g_normal_estandar.show()
    g_espectral.show()
    g_svd.show()
    g_cholesky.show()
    g_normal_estandar_3d.show()
    g_espectral_3d.show()
    g_svd_3d.show()
    g_cholesky_3d.show()
