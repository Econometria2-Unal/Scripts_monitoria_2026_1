"""
Funciones auxiliares para graficacion - Script VAR en Python.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


sns.set_theme(style="whitegrid", context="notebook")


class _ResultadoPronosticoVAR(dict):
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
        # statsmodels 0.14 puede fallar al imprimir summary() cuando trend="n".
        # El resultado si existe; por eso imprimimos manualmente los criterios.
        n_filas = len(next(iter(resultado.ics.values())))
        indice = range(0, n_filas) if incluir_rezago_cero else range(1, n_filas + 1)
        tabla = pd.DataFrame(resultado.ics, index=indice)
        columnas = [col for col in ["aic", "bic", "fpe", "hqic"] if col in tabla]
        print(tabla[columnas])
        print("Rezagos seleccionados:")
        print(pd.Series(resultado.selected_orders))


def imprimir_matrices_acof(var_resultados, variables):
    """Imprime las matrices de coeficientes de un VAR estimado.

    Parametros
    ----------
    var_resultados : object
        Resultados del modelo VAR estimado.
    variables : iterable of str
        Nombres de las variables del modelo.

    Retorna
    -------
    None
        La funcion imprime una matriz de coeficientes por cada rezago.
    """
    for lag, matriz in enumerate(var_resultados.coefs, start=1):
        matriz_lag = pd.DataFrame(
            matriz,
            index=variables,
            columns=[f"L{lag}.{variable}" for variable in variables],
        )
        print(f"\nMatriz A_{lag}")
        print(matriz_lag)


def prueba_arch_por_ecuacion(residuales: pd.DataFrame, lags: int, variables=None):
    """Aplica pruebas ARCH univariadas a los residuales de un VAR.

    Parametros
    ----------
    residuales : pandas.DataFrame
        Residuales del modelo VAR, con una columna por ecuacion.
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
    """Aplica pruebas Jarque-Bera a los residuales de un VAR.

    Parametros
    ----------
    residuales : pandas.DataFrame
        Residuales del modelo VAR, con una columna por ecuacion.
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


def predecir_var(
    var_resultados,
    n_ahead: int = 12,
    ci: float = 0.95,
    indice=None,
    variables=None,
):
    """Calcula pronosticos de un VAR con intervalos de confianza.

    Parametros
    ----------
    var_resultados : object
        Resultados del modelo VAR estimado.
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
    ultimos_valores = var_resultados.endog[-var_resultados.k_ar :]
    pronostico_puntual, inferior, superior = var_resultados.forecast_interval(
        y=ultimos_valores,
        steps=n_ahead,
        alpha=alpha,
    )

    if variables is None:
        variables = getattr(var_resultados, "names", None)
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

    return _ResultadoPronosticoVAR(
        {
            "tablas": tablas,
            "pronostico": pronostico_df,
            "inferior": inferior_df,
            "superior": superior_df,
        }
    )


def pronostico_bootstrap_var(var_resultados, pasos, nboot=1000, semilla=None):
    """Calcula pronosticos por bootstrap para un modelo VAR.

    Parametros
    ----------
    var_resultados : object
        Resultados del modelo VAR estimado.
    pasos : int
        Numero de pasos hacia adelante que se quieren pronosticar.
    nboot : int, default 1000
        Numero de repeticiones usadas para construir los pronosticos.
    semilla : int, optional
        Semilla para reproducir los resultados.

    Retorna
    -------
    dict
        Pronosticos simulados, pronostico promedio e intervalos de confianza.
    """
    rng = np.random.default_rng(semilla)
    endog = np.asarray(var_resultados.endog)
    residuales = np.asarray(var_resultados.resid)
    coefs = np.asarray(var_resultados.coefs)
    intercepto = np.asarray(var_resultados.intercept)
    p = var_resultados.k_ar
    n_variables = endog.shape[1]

    ultimos_valores = endog[-p:, :]
    trayectorias = np.empty((nboot, pasos, n_variables))

    for b in range(nboot):
        y_boot = np.vstack([ultimos_valores.copy(), np.zeros((pasos, n_variables))])
        indices_residuales = rng.integers(0, residuales.shape[0], size=pasos)

        for h in range(p, p + pasos):
            prediccion = intercepto.copy()
            for lag in range(p):
                prediccion = prediccion + coefs[lag] @ y_boot[h - lag - 1, :]
            y_boot[h, :] = prediccion + residuales[indices_residuales[h - p], :]

        trayectorias[b, :, :] = y_boot[p:, :]

    alpha = 0.05
    return {
        "trayectorias": trayectorias,
        "pronostico": trayectorias.mean(axis=0),
        "inferior": np.quantile(trayectorias, alpha / 2, axis=0),
        "superior": np.quantile(trayectorias, 1 - alpha / 2, axis=0),
    }


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


def graficar_pronostico_var(
    pronostico: pd.DataFrame,
    inferior: pd.DataFrame,
    superior: pd.DataFrame,
):
    """Grafica pronosticos de un VAR con intervalos de confianza.

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

    fig.suptitle("Pronostico VAR", fontsize=11)
    fig.tight_layout()
    return fig, axes


def graficar_pronostico_bootstrap_var(
    boots: pd.DataFrame,
    variables=None,
    colores: dict[str, str] | None = None,
    figsize=(15, 4),
    titulo: str = "Pronostico puntual con bootstrapping",
):
    """Grafica pronosticos puntuales calculados por bootstrap.

    Parametros
    ----------
    boots : pandas.DataFrame
        Pronosticos puntuales por bootstrap, con una columna por variable.
    variables : iterable of str, optional
        Variables que se quieren graficar. Si no se entrega, se usan todas las
        columnas de `boots`.
    colores : dict, optional
        Colores asociados a cada variable. Si una variable no esta en el
        diccionario, matplotlib asigna el color automaticamente.
    figsize : tuple, default (15, 4)
        Tamano de la figura.
    titulo : str, default "Pronostico puntual con bootstrapping"
        Titulo general de la figura.

    Retorna
    -------
    tuple
        Figura y ejes con las graficas de pronostico por bootstrap.
    """
    boots = _asegurar_dataframe(boots)
    variables = list(boots.columns) if variables is None else list(variables)
    colores = colores or {}

    fig, axes = plt.subplots(1, len(variables), figsize=figsize)
    axes = np.atleast_1d(axes)
    eje_tiempo_bootstrap = _indice_para_grafica(boots.index)

    for ax, variable in zip(axes, variables):
        ax.plot(
            eje_tiempo_bootstrap,
            boots[variable].to_numpy(),
            color=colores.get(variable),
            linewidth=0.8,
        )
        ax.set_title(variable, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, color="#e0e0e0", linewidth=0.8)

    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    return fig, axes


def graficar_fanchart_var(
    datos,
    pronostico_var: dict,
    variables=None,
    colores=("blue", "lightblue"),
    figsize=None,
):
    """Grafica un fanchart de pronosticos de un modelo VAR.

    Parametros
    ----------
    datos : pandas.DataFrame
        Series observadas usadas como historia del fanchart.
    pronostico_var : dict
        Resultado generado por la funcion predecir_var.
    variables : iterable of str, optional
        Variables que se quieren graficar. Si no se entrega, se usan todas las
        columnas del pronostico.
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
    pronostico = pronostico_var["pronostico"]
    inferior = pronostico_var["inferior"]
    superior = pronostico_var["superior"]
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

        ax.plot(
            x_historia,
            historia,
            color="black",
            linewidth=0.7,
        )
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

        ax.set_title(f"Fanchart for variable {variable}", fontsize=12, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
        ax.tick_params(axis="both", colors="black")

    fig.tight_layout(h_pad=2.2)
    return fig, axes


# 2. Funciones auxiliares para graficar errores simulados ----


def graficar_diagnostico_errores(
    u_t,
    errores: Iterable[str],
    correlacion_muestral=None,
    barras: int = 25,
):
    """Grafica diagnosticos para errores simulados.

    Parametros
    ----------
    u_t : pandas.DataFrame or array-like
        Errores simulados del modelo.
    errores : iterable of str
        Nombres de los errores.
    correlacion_muestral : pandas.DataFrame, optional
        Matriz de correlaciones. Si no se entrega, se calcula con los datos.
    barras : int, default 25
        Numero de barras para los histogramas.

    Retorna
    -------
    dict
        Graficas de series, histogramas, QQ plots, correlaciones y datos usados.
    """
    errores = list(errores)
    errores_df = _asegurar_dataframe(u_t, columnas=errores)
    correlacion_muestral = (
        errores_df.corr() if correlacion_muestral is None else correlacion_muestral
    )

    fig_series, axes_series = plt.subplots(
        len(errores), 1, figsize=(10, 2.4 * len(errores)), sharex=True
    )
    axes_series = np.atleast_1d(axes_series)
    for ax, error in zip(axes_series, errores):
        sns.lineplot(
            x=np.arange(1, len(errores_df) + 1),
            y=errores_df[error].to_numpy(),
            ax=ax,
            linewidth=0.6,
        )
        ax.set_title(error, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig_series.suptitle("Errores simulados", fontsize=11)
    fig_series.tight_layout()

    fig_hist, axes_hist = plt.subplots(1, len(errores), figsize=(5 * len(errores), 4))
    axes_hist = np.atleast_1d(axes_hist)
    for ax, error in zip(axes_hist, errores):
        valores = errores_df[error].dropna().to_numpy()
        sns.histplot(
            valores,
            bins=barras,
            stat="density",
            ax=ax,
            color="lightblue",
            edgecolor="white",
        )
        grilla = np.linspace(valores.min(), valores.max(), 100)
        densidad = stats.norm.pdf(grilla, loc=valores.mean(), scale=valores.std(ddof=1))
        ax.plot(grilla, densidad, color="firebrick", linewidth=0.7)
        ax.set_title(error, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Densidad")
    fig_hist.suptitle("Distribucion empirica vs. normal", fontsize=11)
    fig_hist.tight_layout()

    fig_qq, axes_qq = plt.subplots(1, len(errores), figsize=(5 * len(errores), 4))
    axes_qq = np.atleast_1d(axes_qq)
    for ax, error in zip(axes_qq, errores):
        stats.probplot(errores_df[error].dropna().to_numpy(), dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor("royalblue")
        ax.get_lines()[0].set_markeredgecolor("royalblue")
        ax.get_lines()[1].set_color("firebrick")
        ax.set_title(error, fontsize=10)
        ax.set_xlabel("Cuantiles teoricos")
        ax.set_ylabel("Cuantiles muestrales")
    fig_qq.suptitle("QQ plots de los errores", fontsize=11)
    fig_qq.tight_layout()

    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        correlacion_muestral,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax_corr,
    )
    ax_corr.set_title("Correlacion entre errores", fontsize=11)
    ax_corr.set_xlabel("")
    ax_corr.set_ylabel("")
    fig_corr.tight_layout()

    return {
        "series": fig_series,
        "histograma": fig_hist,
        "qq": fig_qq,
        "correlacion": fig_corr,
        "datos": errores_df,
    }


def graficar_diagnostico_residuales_var(
    residuales: pd.DataFrame,
    lags: int = 20,
    incluir_cuadrados: bool = True,
):
    """Grafica diagnosticos de residuales para cada ecuacion del VAR.

    Parametros
    ----------
    residuales : pandas.DataFrame
        Residuales del modelo VAR, con una columna por variable.
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


# 3. Funciones auxiliares para impulso-respuesta ----


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
        Resultado de funciones impulso-respuesta producido por el modelo VAR.
    impulso : str
        Variable que recibe el choque.
    respuesta : str
        Variable cuya respuesta se quiere analizar.
    pasos_adelante : array-like
        Horizontes de respuesta.
    ortog : bool
        Indica si se usan respuestas ortogonalizadas.
    variables : iterable of str
        Nombres de las variables del VAR.
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


def calcular_bandas_irf_bootstrap(
    var_resultados,
    total_pasos: int,
    ortog: bool,
    int_conf: float,
    semilla: int | None = None,
    runs: int = 100,
):
    """Calcula intervalos para funciones impulso-respuesta por bootstrap.

    Parametros
    ----------
    var_resultados : object
        Modelo VAR estimado.
    total_pasos : int
        Numero de pasos hacia adelante para las IRF.
    ortog : bool
        Indica si se calculan bandas para IRF ortogonalizadas.
    int_conf : float
        Nivel de confianza del intervalo.
    semilla : int, optional
        Semilla para reproducir los resultados.
    runs : int, default 100
        Numero de replicas bootstrap.

    Retorna
    -------
    tuple
        Arreglos con los limites inferior y superior.
    """
    rng = np.random.default_rng(semilla)
    endog = np.asarray(var_resultados.endog)
    residuales = np.asarray(var_resultados.resid)
    coefs = np.asarray(var_resultados.coefs)
    intercepto = np.asarray(var_resultados.intercept)
    p = var_resultados.k_ar
    n_obs, n_variables = endog.shape
    n_residuales = residuales.shape[0]
    tendencia = getattr(var_resultados, "trend", "c")

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
            pred = intercepto.copy()
            for lag in range(p):
                pred = pred + coefs[lag] @ y_boot[t - lag - 1, :]
            y_boot[t, :] = pred + resid_boot[t - p, :]

        try:
            var_boot = var_resultados.model.__class__(y_boot)
            ajuste_boot = var_boot.fit(p, trend=tendencia)
            irf_boot = ajuste_boot.irf(total_pasos)
            valores_boot = irf_boot.orth_irfs if ortog else irf_boot.irfs
            irfs_bootstrap.append(valores_boot)
        except Exception:
            continue

    if len(irfs_bootstrap) < runs:
        raise RuntimeError(
            "No fue posible obtener suficientes replicas bootstrap para las IRF."
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
        Resultado de funciones impulso-respuesta producido por el modelo VAR.
    impulso : str
        Variable que recibe el choque.
    respuesta : str
        Variable cuya respuesta se quiere analizar.
    pasos_adelante : array-like
        Horizontes de respuesta.
    ortog : bool
        Indica si se usan respuestas ortogonalizadas.
    variables : iterable of str
        Nombres de las variables del VAR.
    inferior : array-like
        Limites inferiores de los intervalos.
    superior : array-like
        Limites superiores de los intervalos.
    titulo : str
        Titulo de la grafica.
    ax : matplotlib.axes.Axes, optional
        Eje donde se dibuja la grafica.
    color_intervalo : str, default "#bdbdbd"
        Color del intervalo de confianza.
    alpha_intervalo : float, default 0.28
        Transparencia del intervalo de confianza.

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
    var_resultados,
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
    metodo_bandas: str = "bootstrap",
):
    """Grafica una grilla de funciones impulso-respuesta del VAR.

    Parametros
    ----------
    var_resultados : object
        Modelo VAR estimado.
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
    metodo_bandas : {"bootstrap", "montecarlo"}, default "bootstrap"
        Metodo usado para calcular las bandas de confianza.

    Retorna
    -------
    dict
        Objeto IRF, datos, figuras, ejes y bandas de confianza.
    """
    variables = list(variables)
    pasos_adelante = np.asarray(pasos_adelante)
    total_pasos_futuros = len(pasos_adelante) - 1
    signif = 1 - int_conf

    objeto_irf = var_resultados.irf(total_pasos_futuros)
    if metodo_bandas == "bootstrap":
        inferior, superior = calcular_bandas_irf_bootstrap(
            var_resultados,
            total_pasos=total_pasos_futuros,
            ortog=ortog,
            int_conf=int_conf,
            semilla=semilla,
            runs=runs,
        )
    elif metodo_bandas == "montecarlo":
        inferior, superior = objeto_irf.errband_mc(
            orth=ortog,
            repl=runs,
            signif=signif,
            seed=semilla,
        )
    else:
        raise ValueError("metodo_bandas debe ser 'bootstrap' o 'montecarlo'.")

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


# 4. Funciones auxiliares para FEVD ----


def graficar_fevd_var(
    fevd,
    colores: dict[str, str] | None = None,
    figsize=(12, 12),
):
    """Grafica la descomposicion de varianza del error de pronostico.

    Parametros
    ----------
    fevd : object
        Resultado FEVD producido por un modelo VAR estimado.
    colores : dict, optional
        Colores asociados a cada variable.
    figsize : tuple, default (12, 12)
        Tamano de la figura.

    Retorna
    -------
    tuple
        Figura y ejes con las graficas FEVD.
    """
    variables = list(fevd.names)
    colores = colores or {
        "y_1": "#8B008B",  # magenta4
        "y_2": "#00CDCD",  # cyan3
        "y_3": "#6959CD",  # slateblue3
    }

    n_variables = len(variables)
    horizontes = np.arange(1, fevd.periods + 1)
    fig, axes = plt.subplots(n_variables, 1, figsize=figsize, sharex=False)
    axes = np.atleast_1d(axes)

    for i, variable_respuesta in enumerate(variables):
        ax = axes[i]
        acumulado = np.zeros(fevd.periods)
        handles = []

        for variable_impulso in variables:
            valores = fevd.decomp[i, :, variables.index(variable_impulso)]
            barras = ax.bar(
                horizontes,
                valores,
                bottom=acumulado,
                width=0.82,
                color=colores.get(variable_impulso, "grey"),
                edgecolor="black",
                linewidth=0.9,
                label=variable_impulso,
            )
            handles.append(barras[0])
            acumulado = acumulado + valores

        ax.set_title(f"FEVD for {variable_respuesta}", fontsize=14, pad=10)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.4, 0.8])
        ax.set_xticks(horizontes)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="both", colors="black")

        ax.legend(
            handles[::-1],
            variables[::-1],
            loc="center left",
            bbox_to_anchor=(0.89, 0.55),
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=1,
        )

    fig.tight_layout(h_pad=2.4)

    return fig, axes
