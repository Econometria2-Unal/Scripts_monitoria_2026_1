# ============================================
# 3.  EJEMPLO ENDERS : VAR con 3 variables (IPI, CPI, Desempleo – USA)
# ============================================

#!pip install openpyxl: Instalamos la librería openpyxl para poder leer archivos de Excel
script_dir = Path(__file__).resolve().parent
base_path = script_dir.parent.parent / "datos" / "ENDERS.xlsx"
Base = pd.read_excel(base_path)
Base.head(10)

# Tenemos series de frecuencia trimestral desde 1960 Q1 - 2012 Q4 
# para el Índice de Producción Industrial, El índice de precios al consumidor y 
# la tasa de desempleo de Estados Unidos


#Las volvemos series trimestrales
fechas = pd.date_range(start="1960-01-01", periods=len(Base), freq="QS")

IPI = pd.Series(Base["IPI"].values, index=fechas)
CPI = pd.Series(Base["CPI"].values, index=fechas,)
UNEM = pd.Series(Base["Unem"].values, index=fechas)


## Ahora definimos la tasa de inflación y la tasa de crecimiento del IPI

dl_IPI = np.log(IPI).diff().dropna()
dl_CPI = np.log(CPI).diff().dropna()
dl_CPI.head()
#La tasa de desempleo ya es estacionaria, la dejamos tal cual


#~~~~Graficamos las series~~~~#

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
dl_IPI.plot(ax=axes[0], color="lightblue",    title="Tasa de crecimiento IPI")
dl_CPI.plot(   ax=axes[1], color="sienna",       title="Tasa de crecimiento del CPI")
UNEM.plot(  ax=axes[2], color="mediumpurple", title="Tasa de desempleo")
plt.tight_layout()
plt.show()



#~~~Hagamos la prueba ADF para cada serie~~~#


# ~~~ dl.IPI ~~~ #

adf_IPI = adfuller(dl_IPI,regression="n",maxlag=6)

print("ADF statistic:", adf_IPI[0])
print("Critical values:")
for key, value in adf_IPI[4].items():
    print(f"{key}: {value}"); print("p-value:", adf_IPI[1])


# ~~~ dl.CPI ~~~ #

adf_CPI = adfuller(dl_CPI,regression="c", maxlag=6)

print("ADF statistic:", adf_CPI[0])
print("Critical values:")
for key, value in adf_CPI[4].items():
    print(f"{key}: {value}");print("p-value:", adf_CPI[1])


# ~~~ UNEM ~~~ #

adf_UNEM = adfuller(UNEM,regression="c", maxlag=6 )

print("ADF statistic:", adf_UNEM[0])
print("Critical values:")
for key, value in adf_UNEM[4].items():
    print(f"{key}: {value}"); print("p-value:", adf_UNEM[1])


#Unimos las series en una misma matriz

UNEM_r = UNEM.iloc[:211]

UNEM_r.index = dl_IPI.index    # alinea el índice con dl_IPI, esto debido a que UNEM no se diferencio

Y = pd.concat([dl_IPI, UNEM_r, dl_CPI], axis=1)
Y.columns = ["dl_IPI", "Unem", "dl_CPI"]

Y.head()


# ============================================
# 3.1 Identificación
# ============================================

#Primero planteamos el modelo
modelo_enders = VAR(Y)


#Modelo VAR con tendencia y constante
lag_order_ct = modelo_enders.select_order(6, trend="ct")
print(lag_order_ct.summary())


#Modelo VAR con constante
lag_order_c = modelo_enders.select_order(6, trend="c")
print(lag_order_c.summary())


#Modelo VAR sin tendencia ni constante
lag_order_n = modelo_enders.select_order(6, trend="n")
print(lag_order_n.summary())

# Escogemos un VAR con 3 rezagos y constante.
# Por la presencia de constantes en la serie.


# ============================================
# 3.2 Estimación
# ============================================

var_enders_const = modelo_enders.fit(3, trend="c")
print(var_enders_const.summary())

#Veamos las raices del proceso
print("Raíces:", np.abs(var_enders_const.roots))

#Matriz de coeficientes: Resultados de las matrices A1, A2 y A3
print(var_enders_const.coefs)


# ============================================
# 3.3 Validacion de los supuestos del modelo
# ============================================


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# No autocorrelación de los residuales
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Guardamos los residuales del modelo VAR
residuales_enders = var_enders_const.resid

resIPI = residuales_enders.iloc[:,0]
resCPI = residuales_enders.iloc[:,1]
resUNEM = residuales_enders.iloc[:,2]

#~~~~~~~ Prueba Lyung-Box~~~~~~~~~~~

# dl. IPI
lb_IPI= acorr_ljungbox(resIPI, lags=[10,20,30,75], return_df=True)
print("Ljung-Box test - Residuals dl_IPI");print(lb_IPI)

# dl. CPI
lb_CPI = acorr_ljungbox(resCPI, lags=[10,20,30,75], return_df=True)
print("\nLjung-Box test - Residuals d_CPI");print(lb_CPI)

# UNEM
lb_UNEM = acorr_ljungbox(resUNEM, lags=[10,20,30,75], return_df=True)
print("\nLjung-Box test - Residuals UNEM");print(lb_UNEM)



#~~~ Prueba Portmanteu ~~~#

P_12=var_enders_const.test_whiteness(nlags=12);print(P_12.summary())
P_24=var_enders_const.test_whiteness(nlags=24);print(P_24.summary())
P_36=var_enders_const.test_whiteness(nlags=36);print(P_36.summary())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Grafica de los residuales
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


for nombre, serie in residuales_enders.items():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(f"Diagnóstico de residuales – {nombre}", fontsize=13)
    serie.plot(ax=axes[0, 0], title="Residuales", color="steelblue")
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 1].hist(serie, bins=30, color="steelblue", edgecolor="white")
    axes[0, 1].set_title("Distribución")
    plot_acf(serie, ax=axes[1, 0], lags=20, title="ACF residuales")
    plot_pacf(serie, ax=axes[1, 1], lags=20, title="PACF residuales")
    plt.tight_layout()
    plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Homocedasticidad de los residuales
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#Test tipo ARCH 
arch_res_IPI= het_arch(resIPI, nlags=12);print("ARCH test p-value:", arch_res_IPI[1])

arch_res_CPI = het_arch(resCPI, nlags=12);print("ARCH test p-value:", arch_res_CPI[1])

arch_res_UNEM = het_arch(resUNEM, nlags=12);print("ARCH test p-value:", arch_res_UNEM[1])



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Normalidad de los residuales
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

jb_IPI  = jarque_bera(resIPI);print(f"dl_IPI | p-valor: {jb_IPI.pvalue:.4f}")
jb_CPI  = jarque_bera(resCPI);print(f"dl_CPI | p-valor: {jb_CPI.pvalue:.4f}")
jb_UNEM = jarque_bera(resUNEM);print(f"UNEM   | p-valor: {jb_UNEM.pvalue:.4f}")


# ============================================
# 3.4 Pronóstico
# ============================================

steps = 12
forecast_enders = var_enders_const.forecast(y=var_enders_const.endog, steps=steps)
print(forecast_enders)

#Volvemos el pronostico a un DataFrame

forecast_enders = pd.DataFrame(
    forecast_enders,
    columns=Y.columns
);print(forecast_enders)


#~~~Grafica del pronostico~~~#

var_enders_const.plot_forecast(12)
fig = plt.gcf()
fig.set_size_inches(15, 10)
for ax in fig.axes:
    ax.legend(loc="upper left")   # ← mueve la leyenda a la izquierda
plt.tight_layout()
plt.show()



# ============================================
# 3.  Funciones impulso - respuesta
# ============================================

irf = var_enders_const.irf(18)
irf.plot()
plt.show()


#Representacion de las IRF sencillas 
print(irf.irfs)


#~~~IRF ortogonalizadas ~~~#
# Coeficientes de las IRF ortogonales
irf = var_enders_const.irf(18)
irf.plot(orth=True)


#Representacion de las IRF ortogonalizadas
print(irf.orth_irfs)


#============================================
#~~~~~~~~~FIN DEL CODIGO ;) ~~~~~~~~~~~~~~~~
#============================================
