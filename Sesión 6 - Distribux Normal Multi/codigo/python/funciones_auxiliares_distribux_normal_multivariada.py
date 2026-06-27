#' Universidad Nacional de Colombia
#' Facultad de Ciencias Economicas
#'
#' Econometria II | Monitoria
#' Funciones auxiliares para la simulacion normal multivariada.

# Este archivo contiene funciones de apoyo para las secciones finales del
# script simulacion_normal_multivariada.py. La idea pedagogica es que el script
# principal conserve la lectura conceptual, mientras que los detalles tecnicos
# de comparacion, graficacion y exportacion quedan encapsulados aqui.

from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import multivariate_normal


DIRECTORIO_PROYECTO = None
DIRECTORIO_CODIGO_PYTHON = None


def configurar_rutas_exportacion(directorio_proyecto, directorio_codigo_python):
    global DIRECTORIO_PROYECTO, DIRECTORIO_CODIGO_PYTHON

    DIRECTORIO_PROYECTO = Path(directorio_proyecto)
    DIRECTORIO_CODIGO_PYTHON = Path(directorio_codigo_python)


def imprimir_parrafo(texto, ancho=90):
    print("\n".join(textwrap.wrap(str(texto), width=ancho)))
    print()


def _como_dataframe(U, columnas=("u_1", "u_2")):
    if isinstance(U, pd.DataFrame):
        return U.copy()

    return pd.DataFrame(np.asarray(U, dtype=float), columns=list(columnas))


# ===
# 6. Comparacion de resultados manuales y con NumPy ----
# ===

def comparar_manual_numpy(metodo,
                          U_manual,
                          U_numpy,
                          tolerancia=1e-12):
    U_manual = _como_dataframe(U_manual)
    U_numpy = _como_dataframe(U_numpy)

    diferencia_maxima = np.max(
        np.abs(U_manual.to_numpy() - U_numpy.to_numpy())
    )
    son_iguales = np.allclose(
        U_manual.to_numpy(),
        U_numpy.to_numpy(),
        rtol=0.0,
        atol=tolerancia,
    )

    resumen = pd.DataFrame(
        {
            "metodo": [metodo],
            "iguales_manual_numpy": [son_iguales],
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


def crear_tabla_comparacion_numpy(U_manual_espectral,
                                  U_numpy_espectral,
                                  U_manual_svd,
                                  U_numpy_svd,
                                  U_manual_cholesky,
                                  U_numpy_cholesky):
    tabla_comparacion = pd.concat(
        [
            comparar_manual_numpy(
                "Descomposicion espectral",
                U_manual_espectral,
                U_numpy_espectral,
            ),
            comparar_manual_numpy(
                "SVD",
                U_manual_svd,
                U_numpy_svd,
            ),
            comparar_manual_numpy(
                "Cholesky",
                U_manual_cholesky,
                U_numpy_cholesky,
            ),
        ],
        ignore_index=True,
    )

    return tabla_comparacion


def imprimir_resumen_comparacion_numpy(tabla_comparacion,
                                       U_manual_espectral,
                                       Sigma):
    imprimir_parrafo(
        "Lectura conceptual de la comparacion: la simulacion manual y NumPy "
        "parten de la misma matriz Z de normales estandar. Por eso, si las "
        "matrices raiz de Sigma fueron construidas de la misma manera, ambas "
        "simulaciones deben coincidir observacion por observacion."
    )

    imprimir_parrafo(
        "La columna iguales_manual_numpy indica si las dos matrices simuladas "
        "son numericamente iguales para cada metodo. La columna diferencia_maxima "
        "reporta el mayor error absoluto elemento a elemento. Valores True y "
        "diferencias cercanas a cero muestran que la simulacion manual reproduce "
        "lo que hace NumPy internamente."
    )

    print("\nComparacion manual vs. NumPy:")
    print(tabla_comparacion.to_string(index=False))

    print("\nCovarianza muestral - metodo espectral manual:")
    print(_como_dataframe(U_manual_espectral).cov())

    print("\nCovarianza teorica objetivo Sigma:")
    print(Sigma)

    return tabla_comparacion


# ===
# 7. Graficas 2D interactivas con plotly ----
# ===

def preparar_muestra_grafica(U,
                             n_puntos=8000,
                             semilla_graficos=202601):
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


def graficar_normal_bivariada(U,
                              titulo,
                              color_puntos="#1f77b4",
                              etiqueta_x="u_1",
                              etiqueta_y="u_2",
                              n_puntos=8000,
                              semilla_graficos=202601):
    datos_grafica = preparar_muestra_grafica(
        U=U,
        n_puntos=n_puntos,
        semilla_graficos=semilla_graficos,
    )

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


def crear_graficas_2d_normal_multivariada(Z,
                                          U_manual_espectral,
                                          U_manual_svd,
                                          U_manual_cholesky,
                                          n_puntos=8000,
                                          semilla_graficos=202601):
    graficas = {
        "normal_estandar": graficar_normal_bivariada(
            Z,
            "Normal estandar bivariada no correlacionada",
            color_puntos="#2a6fbb",
            etiqueta_x="z_1",
            etiqueta_y="z_2",
            n_puntos=n_puntos,
            semilla_graficos=semilla_graficos,
        ),
        "espectral": graficar_normal_bivariada(
            U_manual_espectral,
            "Normal bivariada correlacionada - Descomposicion espectral",
            color_puntos="#c03a2b",
            n_puntos=n_puntos,
            semilla_graficos=semilla_graficos,
        ),
        "svd": graficar_normal_bivariada(
            U_manual_svd,
            "Normal bivariada correlacionada - SVD",
            color_puntos="#2b8c56",
            n_puntos=n_puntos,
            semilla_graficos=semilla_graficos,
        ),
        "cholesky": graficar_normal_bivariada(
            U_manual_cholesky,
            "Normal bivariada correlacionada - Cholesky",
            color_puntos="#6f4bb3",
            n_puntos=n_puntos,
            semilla_graficos=semilla_graficos,
        ),
    }

    return graficas


def imprimir_descripcion_graficas_2d(n_observaciones,
                                     n_puntos_grafica):
    imprimir_parrafo(
        "Lectura conceptual de las graficas 2D: se simularon "
        f"{n_observaciones} observaciones, pero se grafica una submuestra de "
        f"{n_puntos_grafica} puntos para que el archivo interactivo sea manejable."
    )

    imprimir_parrafo(
        "La nube de la normal estandar no correlacionada debe verse aproximadamente "
        "circular. En cambio, la normal correlacionada debe verse alargada en "
        "direccion positiva, porque rho = 0.90 implica que valores altos de u_1 "
        "tienden a venir acompanados por valores altos de u_2."
    )

    imprimir_parrafo(
        "Los contornos resumen zonas de mayor y menor concentracion de probabilidad. "
        "Para una normal bivariada correlacionada, esos contornos tienen forma "
        "eliptica y su inclinacion refleja el signo y magnitud de la correlacion."
    )


# ===
# 8. Graficas 3D interactivas de densidad con plotly ----
# ===

def crear_grilla_densidad(media,
                          Sigma,
                          n_grilla=80,
                          multiplicador_sd=3.5):
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


def graficar_densidad_3d(media,
                         Sigma,
                         titulo,
                         etiqueta_x="u_1",
                         etiqueta_y="u_2",
                         escala_color="Viridis",
                         n_grilla=80,
                         multiplicador_sd=3.5):
    grilla = crear_grilla_densidad(
        media=media,
        Sigma=Sigma,
        n_grilla=n_grilla,
        multiplicador_sd=multiplicador_sd,
    )

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


def crear_graficas_3d_normal_multivariada(media,
                                          Sigma,
                                          n_grilla=80,
                                          multiplicador_sd=3.5):
    graficas = {
        "normal_estandar_3d": graficar_densidad_3d(
            media=np.array([0.0, 0.0]),
            Sigma=np.eye(2),
            titulo="Densidad 3D - Normal estandar bivariada",
            etiqueta_x="z_1",
            etiqueta_y="z_2",
            escala_color="Viridis",
            n_grilla=n_grilla,
            multiplicador_sd=multiplicador_sd,
        ),
        "espectral_3d": graficar_densidad_3d(
            media=media,
            Sigma=Sigma,
            titulo="Densidad 3D - Descomposicion espectral",
            escala_color="YlOrRd",
            n_grilla=n_grilla,
            multiplicador_sd=multiplicador_sd,
        ),
        "svd_3d": graficar_densidad_3d(
            media=media,
            Sigma=Sigma,
            titulo="Densidad 3D - SVD",
            escala_color="Plasma",
            n_grilla=n_grilla,
            multiplicador_sd=multiplicador_sd,
        ),
        "cholesky_3d": graficar_densidad_3d(
            media=media,
            Sigma=Sigma,
            titulo="Densidad 3D - Cholesky",
            escala_color="Portland",
            n_grilla=n_grilla,
            multiplicador_sd=multiplicador_sd,
        ),
    }

    return graficas


def imprimir_descripcion_graficas_3d():
    imprimir_parrafo(
        "Lectura conceptual de las graficas 3D: la superficie muestra la funcion "
        "de densidad teorica de la normal bivariada. La altura representa que tan "
        "probable es observar combinaciones cercanas a cada punto (u_1, u_2)."
    )

    imprimir_parrafo(
        "Para la normal estandar no correlacionada, la superficie es simetrica "
        "alrededor de cero. Para la normal correlacionada, la base de la superficie "
        "se estira de acuerdo con Sigma. Los tres metodos de descomposicion generan "
        "la misma distribucion objetivo, por eso sus superficies teoricas son "
        "iguales salvo por el titulo y la escala de color."
    )


# ===
# 9. Exportacion de graficas a archivos HTML ----
# ===

def obtener_directorio_proyecto_exportacion():
    candidatos = []

    if DIRECTORIO_PROYECTO is not None:
        candidatos.append(Path(DIRECTORIO_PROYECTO))

    candidatos.append(Path.cwd())
    candidatos.extend(Path.cwd().parents)

    for candidato in candidatos:
        if DIRECTORIO_CODIGO_PYTHON is None:
            continue

        if (candidato / DIRECTORIO_CODIGO_PYTHON).exists():
            return candidato

    raise FileNotFoundError(
        "No se pudo ubicar la raiz del proyecto para exportar las graficas HTML. "
        "Ejecuta primero el bloque de rutas relativas."
    )


@contextmanager
def usar_directorio_proyecto():
    directorio_trabajo_original = Path.cwd()
    os.chdir(obtener_directorio_proyecto_exportacion())

    try:
        yield
    finally:
        os.chdir(directorio_trabajo_original)


def definir_directorio_graficas(nombre_directorio="html_nm",
                                directorio_base=None):
    if directorio_base is None:
        directorio_base = DIRECTORIO_CODIGO_PYTHON

    if directorio_base is None:
        raise ValueError(
            "No se ha configurado directorio_codigo_python. Usa "
            "configurar_rutas_exportacion() antes de exportar las graficas."
        )

    directorio_graficas = Path(directorio_base) / nombre_directorio
    directorio_proyecto_exportacion = obtener_directorio_proyecto_exportacion()
    directorio_graficas_abs = (
        directorio_graficas
        if directorio_graficas.is_absolute()
        else directorio_proyecto_exportacion / directorio_graficas
    )

    directorio_graficas_abs.mkdir(parents=True, exist_ok=True)

    if not directorio_graficas_abs.exists():
        raise OSError(
            "No se pudo crear el directorio de graficas HTML: "
            + directorio_graficas.as_posix()
        )

    return directorio_graficas

def guardar_grafica_html(grafica,
                         nombre_archivo,
                         directorio_graficas):
    ruta_archivo = Path(directorio_graficas) / nombre_archivo
    directorio_proyecto_exportacion = obtener_directorio_proyecto_exportacion()
    ruta_archivo_abs = (
        ruta_archivo
        if ruta_archivo.is_absolute()
        else directorio_proyecto_exportacion / ruta_archivo
    )
    directorio_graficas_abs = ruta_archivo_abs.parent

    with tempfile.TemporaryDirectory() as carpeta_temporal:
        ruta_temporal = Path(carpeta_temporal) / nombre_archivo

        grafica.write_html(
            file=ruta_temporal,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
        )

        directorio_graficas_abs.mkdir(parents=True, exist_ok=True)

        if ruta_archivo_abs.exists():
            ruta_archivo_abs.unlink()

        try:
            shutil.copy2(ruta_temporal, ruta_archivo_abs)
        except OSError as exc:
            raise OSError(
                "No se pudo copiar la grafica HTML a: "
                + ruta_archivo.as_posix()
            ) from exc

        if not ruta_archivo_abs.exists():
            raise OSError(
                "No se pudo copiar la grafica HTML a: "
                + ruta_archivo.as_posix()
            )

    print("Grafica guardada: " + ruta_archivo.as_posix())

    return ruta_archivo

def exportar_graficas_html(graficas,
                           directorio_graficas=None):
    if not isinstance(graficas, dict) or any(nombre == "" for nombre in graficas):
        raise ValueError(
            "El objeto graficas debe ser un diccionario nombrado: "
            "nombre_archivo.html = grafica."
        )

    if directorio_graficas is None:
        directorio_graficas = definir_directorio_graficas()

    directorio_graficas = Path(directorio_graficas)
    directorio_proyecto_exportacion = obtener_directorio_proyecto_exportacion()
    directorio_graficas_abs = (
        directorio_graficas
        if directorio_graficas.is_absolute()
        else directorio_proyecto_exportacion / directorio_graficas
    )
    directorio_graficas_abs.mkdir(parents=True, exist_ok=True)

    rutas = {}

    for nombre_archivo, grafica in graficas.items():
        rutas[nombre_archivo] = guardar_grafica_html(
            grafica=grafica,
            nombre_archivo=nombre_archivo,
            directorio_graficas=directorio_graficas,
        )

    print(
        "\nDirectorio con graficas HTML:\n"
        + directorio_graficas.as_posix()
    )

    return rutas

def imprimir_descripcion_exportacion_html(directorio_graficas):
    imprimir_parrafo(
        "Lectura conceptual de la exportacion: cada grafica se guarda como HTML "
        "autocontenido para que pueda abrirse en un navegador sin depender de la "
        "sesion de Python. Esto facilita compartir las visualizaciones y revisarlas "
        "despues de ejecutar el script."
    )

    imprimir_parrafo(
        "Los archivos se guardaran en: "
        + Path(directorio_graficas).as_posix()
    )


def mostrar_graficas_interactivo(graficas):
    if hasattr(sys, "ps1"):
        for grafica in graficas.values():
            grafica.show()
