"""
Funciones auxiliares para modelos VECM y metodologia Johansen en Python.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.vector_ar.vecm import VECM


sns.set_theme(style="whitegrid", context="notebook")


class _ResultadoPronosticoVECM(dict):
    def __repr__(self):
        tablas = self.get("tablas", {})
        return "\n\n".join(
            f"${variable}\n{tabla}" for variable, tabla in tablas.items()
        )

    __str__ = __repr__


# 1. Funciones auxiliares generales ----


def configurar_entorno_graficas(
    ancho_visualizacion: int = 140,
    max_columns: int = 20,
):
    """Configura la visualizacion general de tablas y graficas.

    Parametros
    ----------
    ancho_visualizacion : int, default 140
        Ancho usado al imprimir tablas en pantalla.
    max_columns : int, default 20
        Numero maximo de columnas que se muestran en pantalla.

    Retorna
    -------
    None
        La funcion ajusta la visualizacion y cierra graficas abiertas.
    """
    pd.set_option("display.width", ancho_visualizacion)
    pd.set_option("display.max_columns", max_columns)
    plt.close("all")


def mostrar_graficas():
    """Muestra las graficas creadas por el script.

    Retorna
    -------
    None
        La funcion muestra las graficas o las cierra si es necesario.
    """
    if plt.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


def imprimir_adf(resultado, nombre_variable):
    """Imprime los resultados principales de una prueba ADF.

    Parametros
    ----------
    resultado : tuple
        Resultado de la prueba ADF.
    nombre_variable : str
        Nombre de la variable analizada.

    Retorna
    -------
    None
        La funcion imprime el resumen de la prueba en pantalla.
    """
    print(f"\nADF para {nombre_variable}")
    print(f"Estadistico ADF: {resultado[0]:.6f}")
    print(f"p-valor: {resultado[1]:.6f}")
    print(f"Rezagos usados: {resultado[2]}")
    print("Valores criticos:")
    for clave, valor in resultado[4].items():
        print(f"  {clave}: {valor:.6f}")


def imprimir_seleccion_rezagos(resultado, titulo, incluir_rezago_cero=True):
    """Imprime la tabla de seleccion de rezagos de un VAR.

    Parametros
    ----------
    resultado : object
        Resultado de la seleccion de rezagos.
    titulo : str
        Titulo que se imprime antes de la tabla.
    incluir_rezago_cero : bool, default True
        Indica si la tabla incluye una fila para cero rezagos.

    Retorna
    -------
    None
        La funcion imprime la tabla y los rezagos seleccionados.
    """
    print(f"\n{titulo}")
    try:
        print(resultado.summary())
    except IndexError:
        n_filas = len(next(iter(resultado.ics.values())))
        indice = range(0, n_filas) if incluir_rezago_cero else range(1, n_filas + 1)
        tabla = pd.DataFrame(resultado.ics, index=indice)
        columnas = [col for col in ["aic", "bic", "fpe", "hqic"] if col in tabla]
        print(tabla[columnas])
        print("Rezagos seleccionados:")
        print(pd.Series(resultado.selected_orders))


def tabla_johansen(resultado, tipo: str) -> pd.DataFrame:
    """Organiza los resultados de una prueba de Johansen.

    Parametros
    ----------
    resultado : object
        Resultado de la prueba de Johansen.
    tipo : {"eigen", "trace"}
        Tipo de estadistico que se quiere presentar.

    Retorna
    -------
    pandas.DataFrame
        Tabla con estadisticos y valores criticos.
    """
    tipo = tipo.lower()
    n_variables = len(resultado.eig)
    etiquetas_rango = ["r = 0"] + [f"r <= {i}" for i in range(1, n_variables)]

    if tipo == "eigen":
        estadisticos = resultado.lr2
        valores_criticos = resultado.cvm
        nombre_estadistico = "Estadistico valor propio maximo"
    elif tipo == "trace":
        estadisticos = resultado.lr1
        valores_criticos = resultado.cvt
        nombre_estadistico = "Estadistico traza"
    else:
        raise ValueError("tipo debe ser 'eigen' o 'trace'.")

    return pd.DataFrame(
        {
            "r": etiquetas_rango,
            nombre_estadistico: estadisticos,
            "CV 90%": valores_criticos[:, 0],
            "CV 95%": valores_criticos[:, 1],
            "CV 99%": valores_criticos[:, 2],
        }
    )


def imprimir_tabla_johansen(resultado, tipo: str, titulo: str):
    """Imprime una tabla de resultados de Johansen.

    Parametros
    ----------
    resultado : object
        Resultado de la prueba de Johansen.
    tipo : {"eigen", "trace"}
        Tipo de estadistico que se quiere presentar.
    titulo : str
        Titulo que se imprime antes de la tabla.

    Retorna
    -------
    pandas.DataFrame
        Tabla impresa en pantalla.
    """
    tabla = tabla_johansen(resultado, tipo)
    print(f"\n{titulo}")
    print(tabla)
    return tabla


def extraer_matrices_vecm(vecm_resultados, variables=None):
    """Extrae las matrices alpha y beta de un VECM estimado.

    Parametros
    ----------
    vecm_resultados : object
        Resultado del modelo VECM estimado.
    variables : iterable of str, optional
        Nombres de las variables del modelo.

    Retorna
    -------
    dict
        Tablas con el vector de cointegracion y las velocidades de ajuste.
    """
    if variables is None:
        variables = getattr(vecm_resultados, "names", None)
    if variables is None:
        variables = [f"y_{i + 1}" for i in range(vecm_resultados.neqs)]
    variables = list(variables)

    n_relaciones = vecm_resultados.beta.shape[1]
    nombres_relaciones = [f"r_{i + 1}" for i in range(n_relaciones)]

    beta = pd.DataFrame(
        vecm_resultados.beta,
        index=variables,
        columns=nombres_relaciones,
    )
    alpha = pd.DataFrame(
        vecm_resultados.alpha,
        index=variables,
        columns=nombres_relaciones,
    )

    const_coint = np.asarray(vecm_resultados.const_coint)
    if const_coint.size:
        constante = pd.DataFrame(
            const_coint,
            index=nombres_relaciones,
            columns=["constante"],
        )
    else:
        constante = pd.DataFrame(index=nombres_relaciones)

    return {
        "beta": beta,
        "alpha": alpha,
        "constante_cointegracion": constante,
    }


def imprimir_matrices_var_reparametrizado(vecm_resultados, variables=None):
    """Imprime las matrices del VAR en niveles asociado a un VECM.

    Parametros
    ----------
    vecm_resultados : object
        Resultado del modelo VECM estimado. Debe tener el atributo `var_rep`,
        que contiene las matrices de la reparametrizacion como VAR en niveles.
    variables : iterable of str, optional
        Nombres de las variables del modelo. Si no se entrega, se usan los
        nombres guardados en el objeto estimado o etiquetas genericas.

    Retorna
    -------
    dict
        Diccionario con las matrices A_i de la reparametrizacion VAR, una por
        cada rezago del VAR en niveles.
    """
    if variables is None:
        variables = getattr(vecm_resultados, "names", None)
    if variables is None:
        n_variables = np.asarray(vecm_resultados.var_rep).shape[1]
        variables = [f"y_{i + 1}" for i in range(n_variables)]
    variables = list(variables)

    matrices = {}
    for i, matriz in enumerate(vecm_resultados.var_rep, start=1):
        matriz_ai = pd.DataFrame(
            matriz,
            index=variables,
            columns=[f"L{i}.{variable}" for variable in variables],
        )
        nombre_matriz = f"A_{i}"
        matrices[nombre_matriz] = matriz_ai

        print(f"\nMatriz {nombre_matriz} de la reparametrizacion VAR")
        print(matriz_ai)

    return matrices


def prueba_arch_por_ecuacion(residuales: pd.DataFrame, lags: int, variables=None):
    """Aplica pruebas ARCH univariadas a los residuales de un modelo.

    Parametros
    ----------
    residuales : pandas.DataFrame
        Residuales del modelo, con una columna por ecuacion.
    lags : int
        Numero de rezagos usados en la prueba.
    variables : iterable of str, optional
        Variables que se quieren evaluar. Si no se entrega, se usan todas las
        columnas de residuales.

    Retorna
    -------
    pandas.DataFrame
        Tabla con los estadisticos y p-valores de la prueba por ecuacion.
    """
    variables = list(residuales.columns) if variables is None else list(variables)
    resultados = []

    print(f"\nPruebas ARCH univariadas con {lags} rezagos")
    for variable in variables:
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(
            residuales[variable],
            nlags=lags,
        )
        resultados.append(
            {
                "variable": variable,
                "lm_stat": lm_stat,
                "lm_pvalue": lm_pvalue,
                "f_stat": f_stat,
                "f_pvalue": f_pvalue,
            }
        )
        print(
            f"{variable}: LM p-value = {lm_pvalue:.6f}; "
            f"F p-value = {f_pvalue:.6f}"
        )

    return pd.DataFrame(resultados).set_index("variable")


def prueba_normalidad_por_ecuacion(residuales: pd.DataFrame, variables=None):
    """Aplica pruebas Jarque-Bera a los residuales de un modelo.

    Parametros
    ----------
    residuales : pandas.DataFrame
        Residuales del modelo, con una columna por ecuacion.
    variables : iterable of str, optional
        Variables que se quieren evaluar. Si no se entrega, se usan todas las
        columnas de residuales.

    Retorna
    -------
    pandas.DataFrame
        Tabla con el estadistico Jarque-Bera y el p-valor por ecuacion.
    """
    variables = list(residuales.columns) if variables is None else list(variables)
    resultados = []

    print("\nPruebas Jarque-Bera univariadas")
    for variable in variables:
        jb = stats.jarque_bera(residuales[variable])
        resultados.append(
            {
                "variable": variable,
                "jb_stat": jb.statistic,
                "p_value": jb.pvalue,
            }
        )
        print(f"Jarque-Bera {variable} p-value: {jb.pvalue:.6f}")

    return pd.DataFrame(resultados).set_index("variable")


def _asegurar_dataframe(datos, columnas: Iterable[str] | None = None) -> pd.DataFrame:
    """Devuelve los datos como un DataFrame de pandas.

    Parametros
    ----------
    datos : pandas.DataFrame or array-like
        Datos que se quieren usar como tabla.
    columnas : iterable of str, optional
        Nombres de columnas cuando los datos no son un DataFrame.

    Retorna
    -------
    pandas.DataFrame
        Copia de los datos organizada como tabla.
    """
    if isinstance(datos, pd.DataFrame):
        return datos.copy()
    return pd.DataFrame(datos, columns=columnas)


def _nombre_para_titulo(variable: str) -> str:
    """Prepara el nombre de una variable para usarlo en titulos.

    Parametros
    ----------
    variable : str
        Nombre original de la variable.

    Retorna
    -------
    str
        Nombre de la variable sin guiones bajos.
    """
    return variable.replace("_", "")


def _indice_para_grafica(indice):
    """Convierte un indice temporal a valores que se puedan graficar.

    Parametros
    ----------
    indice : pandas.Index
        Indice de una serie o tabla.

    Retorna
    -------
    pandas.Index or array-like
        Indice original o version numerica si el indice esta en periodos.
    """
    if isinstance(indice, pd.PeriodIndex):
        if indice.freqstr.startswith("Q"):
            return indice.year + (indice.quarter - 1) / 4
        if indice.freqstr.startswith("M"):
            return indice.year + (indice.month - 1) / 12
        if indice.freqstr.startswith(("A", "Y")):
            return indice.year
        return indice.astype(str)

    return indice


# 2. Funciones auxiliares de graficacion ----


def graficar_ts(serie, titulo: str, color: str, ax=None):
    """Grafica una serie de tiempo.

    Parametros
    ----------
    serie : array-like or pandas.Series
        Serie que se quiere graficar.
    titulo : str
        Titulo de la grafica.
    color : str
        Color de la linea.
    ax : matplotlib.axes.Axes, optional
        Eje donde se dibuja la grafica. Si no se entrega, se crea uno nuevo.

    Retorna
    -------
    tuple
        Figura y eje que contienen la grafica.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    serie = pd.Series(serie)
    eje_x = _indice_para_grafica(serie.index)
    sns.lineplot(x=eje_x, y=serie.to_numpy(), ax=ax, color=color, linewidth=1)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig, ax


def graficar_series_vecm(
    datos,
    variables=None,
    colores: dict[str, str] | None = None,
    etiquetas: dict[str, str] | None = None,
    titulo: str = "Series de tiempo",
    subtitulo: str | None = None,
    figsize=(12, 4),
):
    """Grafica las series de tiempo de un modelo VECM.

    Parametros
    ----------
    datos : pandas.DataFrame
        Series que se quieren graficar.
    variables : iterable of str, optional
        Variables que se quieren incluir. Si no se entrega, se usan todas.
    colores : dict, optional
        Colores asociados a cada variable.
    etiquetas : dict, optional
        Etiquetas que se muestran en la leyenda.
    titulo : str, default "Series de tiempo"
        Titulo principal de la grafica.
    subtitulo : str, optional
        Subtitulo de la grafica.
    figsize : tuple, default (12, 4)
        Tamano de la figura.

    Retorna
    -------
    tuple
        Figura y eje que contienen la grafica.
    """
    datos = _asegurar_dataframe(datos)
    variables = list(datos.columns) if variables is None else list(variables)
    colores = colores or {}
    etiquetas = etiquetas or {}

    fig, ax = plt.subplots(figsize=figsize)
    eje_x = _indice_para_grafica(datos.index)

    for variable in variables:
        ax.plot(
            eje_x,
            datos[variable].to_numpy(),
            label=etiquetas.get(variable, variable),
            color=colores.get(variable),
            linewidth=1.1,
        )

    ax.set_title(titulo, fontsize=12, pad=18 if subtitulo is not None else 6)
    if subtitulo is not None:
        ax.text(
            0.5,
            1.02,
            subtitulo,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#595959",
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=len(variables))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e0e0e0", linewidth=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92) if subtitulo is not None else None)
    return fig, ax


def graficar_diagnostico_residuales_vecm(
    residuales: pd.DataFrame,
    lags: int = 20,
    incluir_cuadrados: bool = True,
):
    """Grafica diagnosticos de residuales para cada ecuacion del modelo.

    Parametros
    ----------
    residuales : pandas.DataFrame
        Residuales del modelo, con una columna por variable.
    lags : int, default 20
        Numero de rezagos usados en las graficas ACF y PACF.
    incluir_cuadrados : bool, default True
        Si es True, agrega las graficas ACF y PACF de los residuales al
        cuadrado.

    Retorna
    -------
    dict
        Diccionario con una figura de diagnostico por variable.
    """
    figuras = {}

    for nombre, serie in residuales.items():
        n_filas = 3 if incluir_cuadrados else 2
        fig, axes = plt.subplots(
            n_filas,
            2,
            figsize=(12, 8 if incluir_cuadrados else 6),
        )
        fig.suptitle(f"Diagnostico de residuales - {nombre}", fontsize=13)
        serie = serie.dropna()

        eje_x = _indice_para_grafica(serie.index)
        sns.lineplot(x=eje_x, y=serie.to_numpy(), ax=axes[0, 0], color="steelblue")
        axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[0, 0].set_title("Residuales")
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("")

        sns.histplot(serie.to_numpy(), bins=30, ax=axes[0, 1], color="steelblue")
        axes[0, 1].set_title("Distribucion")
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("")

        plot_acf(serie, ax=axes[1, 0], lags=lags, title="ACF residuales")
        plot_pacf(
            serie,
            ax=axes[1, 1],
            lags=lags,
            method="ywm",
            title="PACF residuales",
        )

        if incluir_cuadrados:
            serie_cuadrada = serie**2
            plot_acf(
                serie_cuadrada,
                ax=axes[2, 0],
                lags=lags,
                title="ACF residuales al cuadrado",
            )
            plot_pacf(
                serie_cuadrada,
                ax=axes[2, 1],
                lags=lags,
                method="ywm",
                title="PACF residuales al cuadrado",
            )

        fig.tight_layout()
        figuras[nombre] = fig

    return figuras


# 3. Funciones auxiliares para pronostico ----


def predecir_vecm(
    vecm_resultados,
    n_ahead: int = 12,
    ci: float = 0.95,
    indice=None,
    variables=None,
):
    """Calcula pronosticos de un VECM con intervalos de confianza.

    Parametros
    ----------
    vecm_resultados : object
        Resultado del modelo VECM estimado.
    n_ahead : int, default 12
        Numero de periodos hacia adelante que se quieren pronosticar.
    ci : float, default 0.95
        Nivel de confianza usado para los intervalos.
    indice : pandas.Index, optional
        Indice temporal de la serie original.
    variables : iterable of str, optional
        Nombres de las variables del modelo.

    Retorna
    -------
    dict
        Tablas de pronostico por variable, pronostico puntual y limites del
        intervalo de confianza.
    """
    if not 0 < ci < 1:
        raise ValueError("ci debe estar entre 0 y 1.")

    alpha = 1 - ci
    pronostico_puntual, inferior, superior = vecm_resultados.predict(
        steps=n_ahead,
        alpha=alpha,
    )

    if variables is None:
        variables = getattr(vecm_resultados, "names", None)
    if variables is None:
        variables = [f"y_{i + 1}" for i in range(pronostico_puntual.shape[1])]
    variables = list(variables)

    if isinstance(indice, pd.PeriodIndex):
        fechas_futuras = pd.period_range(
            start=indice[-1] + 1,
            periods=n_ahead,
            freq=indice.freq,
            name=indice.name,
        )
    else:
        fechas_futuras = pd.RangeIndex(
            start=1,
            stop=n_ahead + 1,
            name="paso",
        )

    pronostico_df = pd.DataFrame(
        pronostico_puntual,
        index=fechas_futuras,
        columns=variables,
    )
    inferior_df = pd.DataFrame(inferior, index=fechas_futuras, columns=variables)
    superior_df = pd.DataFrame(superior, index=fechas_futuras, columns=variables)

    tablas = {}
    for variable in variables:
        tablas[variable] = pd.DataFrame(
            {
                "fcst": pronostico_df[variable],
                "lower": inferior_df[variable],
                "upper": superior_df[variable],
                "CI": superior_df[variable] - pronostico_df[variable],
            }
        )

    return _ResultadoPronosticoVECM(
        {
            "tablas": tablas,
            "pronostico": pronostico_df,
            "inferior": inferior_df,
            "superior": superior_df,
        }
    )


def graficar_pronostico_vecm(
    pronostico: pd.DataFrame,
    inferior: pd.DataFrame,
    superior: pd.DataFrame,
):
    """Grafica pronosticos de un VECM con intervalos de confianza.

    Parametros
    ----------
    pronostico : pandas.DataFrame
        Pronosticos puntuales de cada variable.
    inferior : pandas.DataFrame
        Limite inferior del intervalo de confianza.
    superior : pandas.DataFrame
        Limite superior del intervalo de confianza.

    Retorna
    -------
    tuple
        Figura y ejes con las graficas de pronostico.
    """
    variables = list(pronostico.columns)
    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 4))
    axes = np.atleast_1d(axes)

    for ax, variable in zip(axes, variables):
        pasos = np.arange(1, len(pronostico) + 1)
        ax.fill_between(
            pasos,
            inferior[variable].to_numpy(),
            superior[variable].to_numpy(),
            color="grey",
            alpha=0.35,
        )
        sns.lineplot(
            x=pasos,
            y=pronostico[variable].to_numpy(),
            ax=ax,
            color="royalblue",
            linewidth=0.8,
        )
        ax.set_title(variable, fontsize=11)
        ax.set_xlabel("Pasos adelante")
        ax.set_ylabel("")

    fig.suptitle("Pronostico VECM", fontsize=11)
    fig.tight_layout()
    return fig, axes


def graficar_fanchart_vecm(
    datos,
    pronostico_vecm: dict,
    variables=None,
    colores=("blue", "lightblue"),
    figsize=None,
):
    """Grafica un fanchart de pronosticos de un modelo VECM.

    Parametros
    ----------
    datos : pandas.DataFrame
        Series observadas usadas como historia del fanchart.
    pronostico_vecm : dict
        Resultado generado por la funcion predecir_vecm.
    variables : iterable of str, optional
        Variables que se quieren graficar. Si no se entrega, se usan todas.
    colores : tuple, default ("blue", "lightblue")
        Color de la linea de pronostico y color del intervalo.
    figsize : tuple, optional
        Tamano de la figura.

    Retorna
    -------
    tuple
        Figura y ejes con el fanchart de cada variable.
    """
    datos = _asegurar_dataframe(datos)
    pronostico = pronostico_vecm["pronostico"]
    inferior = pronostico_vecm["inferior"]
    superior = pronostico_vecm["superior"]
    variables = list(pronostico.columns) if variables is None else list(variables)

    if figsize is None:
        figsize = (12, 2.8 * len(variables))

    color_pronostico, color_intervalo = colores
    fig, axes = plt.subplots(len(variables), 1, figsize=figsize, sharex=False)
    axes = np.atleast_1d(axes)

    x_historia = np.arange(len(datos))
    x_pronostico = np.arange(len(datos), len(datos) + len(pronostico))

    for ax, variable in zip(axes, variables):
        historia = datos[variable].to_numpy()
        pronostico_variable = pronostico[variable].to_numpy()
        inferior_variable = inferior[variable].to_numpy()
        superior_variable = superior[variable].to_numpy()

        ax.plot(x_historia, historia, color="black", linewidth=0.7)
        ax.fill_between(
            x_pronostico,
            inferior_variable,
            superior_variable,
            color=color_intervalo,
            alpha=0.75,
            linewidth=0,
        )
        ax.plot(
            x_pronostico,
            pronostico_variable,
            color=color_pronostico,
            linewidth=1.2,
        )
        ax.plot(
            [x_historia[-1], x_pronostico[0]],
            [historia[-1], pronostico_variable[0]],
            color=color_pronostico,
            linewidth=1.0,
        )

        ax.set_title(f"Fanchart para {variable}", fontsize=12, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
        ax.tick_params(axis="both", colors="black")

    fig.tight_layout(h_pad=2.2)
    return fig, axes


# 4. Funciones auxiliares para impulso-respuesta ----


def extraer_datos_irf(
    objeto_irf,
    impulso: str,
    respuesta: str,
    pasos_adelante,
    ortog: bool,
    variables: Iterable[str],
    inferior,
    superior,
) -> pd.DataFrame:
    """Extrae los datos de una funcion impulso-respuesta especifica.

    Parametros
    ----------
    objeto_irf : object
        Resultado de funciones impulso-respuesta producido por el modelo.
    impulso : str
        Variable que recibe el choque.
    respuesta : str
        Variable cuya respuesta se quiere analizar.
    pasos_adelante : array-like
        Horizontes de respuesta.
    ortog : bool
        Indica si se usan respuestas ortogonalizadas.
    variables : iterable of str
        Nombres de las variables del modelo.
    inferior : array-like
        Limites inferiores de los intervalos.
    superior : array-like
        Limites superiores de los intervalos.

    Retorna
    -------
    pandas.DataFrame
        Tabla con horizonte, IRF y limites del intervalo.
    """
    variables = list(variables)
    impulso_idx = variables.index(impulso)
    respuesta_idx = variables.index(respuesta)
    valores_irf = objeto_irf.orth_irfs if ortog else objeto_irf.irfs

    return pd.DataFrame(
        {
            "pasos_adelante": np.asarray(pasos_adelante),
            "irf": valores_irf[:, respuesta_idx, impulso_idx],
            "inferior": inferior[:, respuesta_idx, impulso_idx],
            "superior": superior[:, respuesta_idx, impulso_idx],
        }
    )


def _intercepto_var_desde_vecm(vecm_resultados):
    """Calcula la constante de la representacion VAR del VECM.

    Parametros
    ----------
    vecm_resultados : object
        Resultado del modelo VECM estimado.

    Retorna
    -------
    numpy.ndarray
        Vector de constantes asociado a la representacion en niveles.
    """
    deterministico = getattr(vecm_resultados.model, "deterministic", "")
    if "l" in deterministico:
        raise ValueError(
            "Las bandas bootstrap no estan implementadas para modelos "
            "con tendencia deterministica."
        )

    n_variables = vecm_resultados.neqs
    intercepto = np.zeros(n_variables)

    const = np.asarray(vecm_resultados.const, dtype=float)
    if const.size:
        intercepto = intercepto + const.reshape(n_variables, -1).sum(axis=1)

    const_coint = np.asarray(vecm_resultados.const_coint, dtype=float)
    if const_coint.size:
        const_coint = const_coint.reshape(vecm_resultados.coint_rank, -1)
        intercepto = intercepto + (vecm_resultados.alpha @ const_coint).sum(axis=1)

    return intercepto


def calcular_bandas_irf_bootstrap(
    vecm_resultados,
    total_pasos: int,
    ortog: bool,
    int_conf: float,
    variables: Iterable[str],
    semilla: int | None = None,
    runs: int = 100,
):
    """Calcula intervalos para funciones impulso-respuesta por bootstrap.

    Parametros
    ----------
    vecm_resultados : object
        Modelo VECM estimado.
    total_pasos : int
        Numero de pasos hacia adelante para las IRF.
    ortog : bool
        Indica si se calculan bandas para IRF ortogonalizadas.
    int_conf : float
        Nivel de confianza del intervalo.
    variables : iterable of str
        Nombres de las variables del modelo.
    semilla : int, optional
        Semilla para reproducir los resultados.
    runs : int, default 100
        Numero de replicas bootstrap.

    Retorna
    -------
    tuple
        Arreglos con los limites inferior y superior.
    """
    variables = list(variables)
    rng = np.random.default_rng(semilla)
    endog = np.asarray(vecm_resultados.model.endog, dtype=float)
    residuales = np.asarray(vecm_resultados.resid, dtype=float)
    var_rep = np.asarray(vecm_resultados.var_rep, dtype=float)
    intercepto = _intercepto_var_desde_vecm(vecm_resultados)

    p = vecm_resultados.k_ar
    n_obs, n_variables = endog.shape
    n_residuales = residuales.shape[0]
    deterministico = vecm_resultados.model.deterministic
    coint_rank = vecm_resultados.coint_rank

    irfs_bootstrap = []
    max_intentos = max(runs * 3, runs + 20)
    intentos = 0

    while len(irfs_bootstrap) < runs and intentos < max_intentos:
        intentos += 1
        indices = rng.integers(0, n_residuales, size=n_obs - p)
        resid_boot = residuales[indices]

        y_boot = np.zeros_like(endog)
        y_boot[:p, :] = endog[:p, :]

        for t in range(p, n_obs):
            prediccion = intercepto.copy()
            for lag in range(p):
                prediccion = prediccion + var_rep[lag] @ y_boot[t - lag - 1, :]
            y_boot[t, :] = prediccion + resid_boot[t - p, :]

        try:
            datos_boot = pd.DataFrame(y_boot, columns=variables)
            ajuste_boot = VECM(
                datos_boot,
                k_ar_diff=p - 1,
                coint_rank=coint_rank,
                deterministic=deterministico,
            ).fit()
            irf_boot = ajuste_boot.irf(total_pasos)
            valores_boot = irf_boot.orth_irfs if ortog else irf_boot.irfs
            irfs_bootstrap.append(valores_boot)
        except Exception:
            continue

    minimo_replicas = max(10, int(0.4 * runs))
    if len(irfs_bootstrap) < minimo_replicas:
        raise RuntimeError(
            "No fue posible obtener suficientes replicas bootstrap para las IRF."
        )

    if len(irfs_bootstrap) < runs:
        print(
            "Advertencia: se usaron "
            f"{len(irfs_bootstrap)} replicas bootstrap de {runs} solicitadas."
        )

    irfs_bootstrap = np.asarray(irfs_bootstrap)
    alpha = 1 - int_conf
    inferior = np.quantile(irfs_bootstrap, alpha / 2, axis=0)
    superior = np.quantile(irfs_bootstrap, 1 - alpha / 2, axis=0)

    return inferior, superior


def graficar_datos_irf(
    datos_irf: pd.DataFrame,
    titulo: str,
    ax=None,
    color_intervalo: str = "#bdbdbd",
    alpha_intervalo: float = 0.28,
):
    """Grafica una funcion impulso-respuesta con su intervalo.

    Parametros
    ----------
    datos_irf : pandas.DataFrame
        Tabla con horizonte, IRF y limites del intervalo.
    titulo : str
        Titulo de la grafica.
    ax : matplotlib.axes.Axes, optional
        Eje donde se dibuja la grafica. Si no se entrega, se crea uno nuevo.
    color_intervalo : str, default "#bdbdbd"
        Color del intervalo de confianza.
    alpha_intervalo : float, default 0.28
        Transparencia del intervalo de confianza.

    Retorna
    -------
    tuple
        Figura y eje que contienen la grafica.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    else:
        fig = ax.figure

    x = datos_irf["pasos_adelante"].to_numpy()
    irf = datos_irf["irf"].to_numpy()
    inferior = datos_irf["inferior"].to_numpy()
    superior = datos_irf["superior"].to_numpy()

    ax.set_axisbelow(True)
    ax.grid(True, color="#e0e0e0", linewidth=0.8)
    ax.fill_between(
        x,
        inferior,
        superior,
        color=color_intervalo,
        alpha=alpha_intervalo,
        linewidth=0,
        zorder=1,
    )
    ax.axhline(0, color="red", linewidth=0.8, zorder=3)
    ax.plot(x, irf, color="black", linewidth=0.8, zorder=4)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Pasos adelante")
    ax.set_ylabel("")
    return fig, ax


def graficar_irf_extraida(
    objeto_irf,
    impulso: str,
    respuesta: str,
    pasos_adelante,
    ortog: bool,
    variables: Iterable[str],
    inferior,
    superior,
    titulo: str,
    ax=None,
    color_intervalo: str = "#bdbdbd",
    alpha_intervalo: float = 0.28,
):
    """Extrae y grafica una funcion impulso-respuesta especifica.

    Parametros
    ----------
    objeto_irf : object
        Resultado de funciones impulso-respuesta producido por el modelo.
    impulso : str
        Variable que recibe el choque.
    respuesta : str
        Variable cuya respuesta se quiere analizar.
    pasos_adelante : array-like
        Horizontes de respuesta.
    ortog : bool
        Indica si se usan respuestas ortogonalizadas.
    variables : iterable of str
        Nombres de las variables del modelo.
    inferior : array-like
        Limites inferiores de los intervalos.
    superior : array-like
        Limites superiores de los intervalos.
    titulo : str
        Titulo de la grafica.
    ax : matplotlib.axes.Axes, optional
        Eje donde se dibuja la grafica.
    color_intervalo : str, default "#bdbdbd"
        Color de los intervalos de confianza.
    alpha_intervalo : float, default 0.28
        Transparencia de los intervalos de confianza.

    Retorna
    -------
    tuple
        Figura, eje y tabla de datos usados en la grafica.
    """
    datos_irf = extraer_datos_irf(
        objeto_irf,
        impulso,
        respuesta,
        pasos_adelante,
        ortog,
        variables,
        inferior,
        superior,
    )
    fig, ax = graficar_datos_irf(
        datos_irf,
        titulo,
        ax=ax,
        color_intervalo=color_intervalo,
        alpha_intervalo=alpha_intervalo,
    )
    return fig, ax, datos_irf


def graficar_grilla_irf(
    vecm_resultados,
    variables: Iterable[str],
    pasos_adelante,
    ortog: bool,
    int_conf: float,
    prefijo_titulo: str,
    semilla: int | None = None,
    runs: int = 100,
    figsize=None,
    color_intervalo: str = "#bdbdbd",
    alpha_intervalo: float = 0.28,
):
    """Grafica una grilla de funciones impulso-respuesta del VECM.

    Parametros
    ----------
    vecm_resultados : object
        Modelo VECM estimado.
    variables : iterable of str
        Nombres de las variables del modelo.
    pasos_adelante : array-like
        Horizontes de respuesta.
    ortog : bool
        Indica si se grafican IRF ortogonalizadas.
    int_conf : float
        Nivel de confianza de los intervalos.
    prefijo_titulo : str
        Texto inicial usado en los titulos de las graficas.
    semilla : int, optional
        Semilla para reproducir los intervalos.
    runs : int, default 100
        Numero de replicas usadas para calcular intervalos.
    figsize : tuple, optional
        Tamano de la figura completa.
    color_intervalo : str, default "#bdbdbd"
        Color de los intervalos de confianza.
    alpha_intervalo : float, default 0.28
        Transparencia de los intervalos de confianza.

    Retorna
    -------
    dict
        Objeto IRF, datos, figura, ejes y bandas de confianza.
    """
    variables = list(variables)
    pasos_adelante = np.asarray(pasos_adelante)
    total_pasos_futuros = len(pasos_adelante) - 1

    objeto_irf = vecm_resultados.irf(total_pasos_futuros)
    inferior, superior = calcular_bandas_irf_bootstrap(
        vecm_resultados,
        total_pasos=total_pasos_futuros,
        ortog=ortog,
        int_conf=int_conf,
        variables=variables,
        semilla=semilla,
        runs=runs,
    )

    n_variables = len(variables)
    if figsize is None:
        figsize = (5.5 * n_variables, 3.1 * n_variables)

    fig, axes = plt.subplots(n_variables, n_variables, figsize=figsize)
    axes = np.asarray(axes).reshape(n_variables, n_variables)

    combinaciones = []
    datos = []
    graficas = []

    for fila, respuesta in enumerate(variables):
        for columna, impulso in enumerate(variables):
            titulo = (
                f"{prefijo_titulo} de {_nombre_para_titulo(impulso)}"
                f" - respuesta de {_nombre_para_titulo(respuesta)}"
            )
            ax = axes[fila, columna]
            _, _, datos_irf = graficar_irf_extraida(
                objeto_irf,
                impulso,
                respuesta,
                pasos_adelante,
                ortog,
                variables,
                inferior,
                superior,
                titulo,
                ax=ax,
                color_intervalo=color_intervalo,
                alpha_intervalo=alpha_intervalo,
            )
            combinaciones.append(
                {
                    "impulso": impulso,
                    "respuesta": respuesta,
                    "titulo": titulo,
                }
            )
            datos.append(datos_irf.assign(impulso=impulso, respuesta=respuesta))
            graficas.append(ax)

    fig.tight_layout()

    return {
        "objeto_irf": objeto_irf,
        "combinaciones": pd.DataFrame(combinaciones),
        "datos": pd.concat(datos, ignore_index=True),
        "graficas": graficas,
        "figura": fig,
        "ejes": axes,
        "bandas_inferiores": inferior,
        "bandas_superiores": superior,
    }
