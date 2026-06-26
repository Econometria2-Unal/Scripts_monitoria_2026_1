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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


sns.set_theme(style="whitegrid", context="notebook")


# 1. Funciones auxiliares generales ----


def _asegurar_dataframe(datos, columnas: Iterable[str] | None = None) -> pd.DataFrame:
    if isinstance(datos, pd.DataFrame):
        return datos.copy()
    return pd.DataFrame(datos, columns=columnas)


def _nombre_para_titulo(variable: str) -> str:
    return variable.replace("_", "")


def graficar_ts(serie, titulo: str, color: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    serie = pd.Series(serie)
    sns.lineplot(x=serie.index, y=serie.to_numpy(), ax=ax, color=color, linewidth=1)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig, ax


def graficar_pronostico_var(
    pronostico: pd.DataFrame,
    inferior: pd.DataFrame,
    superior: pd.DataFrame,
):
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


# 2. Funciones auxiliares para graficar errores simulados ----


def graficar_diagnostico_errores(
    u_t,
    errores: Iterable[str],
    cor_u_muestral=None,
    bins: int = 25,
):
    errores = list(errores)
    errores_df = _asegurar_dataframe(u_t, columnas=errores)
    cor_u_muestral = errores_df.corr() if cor_u_muestral is None else cor_u_muestral

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
            bins=bins,
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
        cor_u_muestral,
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


def graficar_diagnostico_residuales_var(residuales: pd.DataFrame, lags: int = 20):
    figuras = {}

    for nombre, serie in residuales.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        fig.suptitle(f"Diagnostico de residuales - {nombre}", fontsize=13)

        sns.lineplot(x=serie.index, y=serie.to_numpy(), ax=axes[0, 0], color="steelblue")
        axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[0, 0].set_title("Residuales")
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("")

        sns.histplot(serie.to_numpy(), bins=30, ax=axes[0, 1], color="steelblue")
        axes[0, 1].set_title("Distribucion")
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("")

        plot_acf(serie, ax=axes[1, 0], lags=lags, title="ACF residuales")
        plot_pacf(serie, ax=axes[1, 1], lags=lags, method="ywm", title="PACF residuales")

        fig.tight_layout()
        figuras[nombre] = fig

    return figuras


# 3. Funciones auxiliares para impulso-respuesta ----


def extraer_datos_irf(
    IRF,
    impulso: str,
    respuesta: str,
    pasos_adelante,
    ortog: bool,
    variables: Iterable[str],
    inferior,
    superior,
) -> pd.DataFrame:
    variables = list(variables)
    impulso_idx = variables.index(impulso)
    respuesta_idx = variables.index(respuesta)
    valores_irf = IRF.orth_irfs if ortog else IRF.irfs

    return pd.DataFrame(
        {
            "pasos_adelante": np.asarray(pasos_adelante),
            "irf": valores_irf[:, respuesta_idx, impulso_idx],
            "inferior": inferior[:, respuesta_idx, impulso_idx],
            "superior": superior[:, respuesta_idx, impulso_idx],
        }
    )


def calcular_bandas_irf_bootstrap(
    var,
    total_pasos_futuros: int,
    ortog: bool,
    int_conf: float,
    semilla: int | None = None,
    runs: int = 100,
):
    rng = np.random.default_rng(semilla)
    endog = np.asarray(var.endog)
    residuales = np.asarray(var.resid)
    coefs = np.asarray(var.coefs)
    intercepto = np.asarray(var.intercept)
    p = var.k_ar
    n_obs, n_variables = endog.shape
    n_residuales = residuales.shape[0]
    tendencia = getattr(var, "trend", "c")

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
            var_boot = var.model.__class__(y_boot)
            ajuste_boot = var_boot.fit(p, trend=tendencia)
            irf_boot = ajuste_boot.irf(total_pasos_futuros)
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
    IRF_data_frame: pd.DataFrame,
    titulo: str,
    ax=None,
    color_intervalo: str = "#bdbdbd",
    alpha_intervalo: float = 0.28,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    else:
        fig = ax.figure

    x = IRF_data_frame["pasos_adelante"].to_numpy()
    irf = IRF_data_frame["irf"].to_numpy()
    inferior = IRF_data_frame["inferior"].to_numpy()
    superior = IRF_data_frame["superior"].to_numpy()

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
    IRF,
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
    datos_irf = extraer_datos_irf(
        IRF, impulso, respuesta, pasos_adelante, ortog, variables, inferior, superior
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
    var,
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
    variables = list(variables)
    pasos_adelante = np.asarray(pasos_adelante)
    total_pasos_futuros = len(pasos_adelante) - 1
    signif = 1 - int_conf

    IRF = var.irf(total_pasos_futuros)
    if metodo_bandas == "bootstrap":
        inferior, superior = calcular_bandas_irf_bootstrap(
            var,
            total_pasos_futuros=total_pasos_futuros,
            ortog=ortog,
            int_conf=int_conf,
            semilla=semilla,
            runs=runs,
        )
    elif metodo_bandas == "montecarlo":
        inferior, superior = IRF.errband_mc(
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
                IRF,
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
        "objeto_irf": IRF,
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
