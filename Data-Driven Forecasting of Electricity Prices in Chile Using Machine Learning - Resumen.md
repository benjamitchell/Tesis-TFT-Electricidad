# Resumen: "Data-Driven Forecasting of Electricity Prices in Chile Using Machine Learning"

**Referencia:** León, R.; Ramírez, G.; Cifuentes, C.; Vergara, S.; Aedo-García, R.; Ramis Lanyon, F.; Villalobos San Martin, R.J. *Appl. Sci.* **2026**, *16*, 1318. https://doi.org/10.3390/app16031318
Recibido: 17 dic 2025 | Aceptado: 23 ene 2026 | Publicado: 28 ene 2026 (MDPI, open access, CC BY)

---

## 1. Contexto y motivación

Forecasting del **SMP (System Marginal Price)** en el Sistema Eléctrico Nacional (NES) chileno. Motivación: alta penetración de generación renovable variable (VRG, >30% de la capacidad instalada, ~19 GW a abril 2025) y **congestión de transmisión persistente y estructural**, agravada por la topología radial y muy extendida (3.100+ km) del sistema chileno, sin interconexión internacional. La congestión provoca desacople espacial del SMP entre barras (Tabla 1 del paper cuantifica esto: hasta 43.3% de horas desacopladas en el corredor Charrúa–Puerto Montt en julio 2024).

## 2. Datos

- **Fuente**: plataforma pública del Coordinador Eléctrico Nacional (CEN).
- **Período**: 1 enero – 13 diciembre 2024 (un solo año calendario). Resolución horaria, 8.352 registros/variable.
- **Variables predictoras**: SMP en múltiples nodos (USD/MWh), demanda neta del sistema (GW = demanda total − VRG), generación despachada por tecnología (hidro, eólico, solar, carbón, gas natural).
- **Sin limpieza**: no aplicaron remoción de outliers ni normalización/escalado — datos "crudos" a propósito.

## 3. Segmentación espacial (clustering de barras)

Correlación de Pearson entre el SMP de 28 barras → 3 zonas:

| Zona | Barras | Rango de correlación | Barra representativa |
|---|---|---|---|
| I (norte) | 18 barras, Parinacota...N.P. Azúcar | 0.95–1.0 | **Crucero** |
| II (centro) | 7 barras, Quillota...Cautín | 0.85–1.0 | **Alto Jahuel** |
| III (sur) | Valdivia, Puerto Montt, Chiloé | 0.95–1.0 | **Puerto Montt** |

Solo modelan estas 3 barras (una por zona), no las 28.

## 4. Metodologías comparadas (M1 vs M2)

Ambas son **modelos por barra objetivo** (node-specific: un modelo independiente entrenado para Crucero, otro para Alto Jahuel, otro para Puerto Montt). La diferencia está en el **conjunto de inputs de SMP rezagado** que alimenta a cada modelo:

- **M1**: SMP rezagado (t−1 a t−5, es decir, 5 horas previas a la hora objetivo) de **todas** las barras del sistema, + generación por tecnología + demanda neta.
- **M2**: misma estructura temporal (t−1 a t−5) y mismas variables no-precio, pero el SMP rezagado se limita solo a las barras **del mismo cluster/zona correlacionada** que la barra objetivo.

El número de lags (5) se eligió empíricamente: probaron con menos lags y encontraron saturación de desempeño más allá de 5 horas.

## 5. Modelos (todos scikit-learn, sin deep learning)

LR (Linear Regression, baseline), BR (Bayesian Ridge), ARD (Automatic Relevance Determination), DTR (Decision Tree Regressor), RFR (Random Forest Regressor, `min_samples_split=20`), SVR (kernel polinomial grado 3, `C=10⁴`). Hiperparámetros default salvo SVR/RFR. Tuning solo con train+val, nunca con los meses de test.

## 6. Split de datos

- **Test**: enero y julio 2024 completos, reservados enteramente (para contrastar verano vs invierno).
- **Train**: resto del año (6.854 muestras), con split aleatorio 80/20 train/val dentro de eso.
- El split de train **no es cronológico** — es aleatorio dentro del año, y el test queda temporalmente "rodeado" por datos de entrenamiento de antes y después.

## 7. Métricas

MAE, RMSE, correlación de Pearson (ρ), R².

## 8. Resultados principales

**Por modelo**: RFR, SVR y BR consistentemente los mejores. DTR el peor en todos los casos.

**Por zona** (mejores modelos, MAE combinado):
- Crucero: RFR ≈ 7.71–7.79, R² ≈ 0.86–0.87
- Alto Jahuel: RFR ≈ 8.90–8.98, R² ≈ 0.82
- Puerto Montt: RFR/SVR/BR ≈ 17–21, R² ≈ 0.74–0.76 (casi el doble de error, atribuido a congestión estructural)

**M1 vs M2**: en 10/18 combinaciones modelo×zona, M2 fue mejor. Ventaja de M2 más clara en Puerto Montt (zona más congestionada, 4/6 modelos mejoran). En Crucero (bien conectada) casi no hay diferencia.

**Test de Wilcoxon signed-rank** (pareado, sobre errores absolutos horarios, α=0.05): confirma estadísticamente que M2 es significativamente mejor que M1 en Puerto Montt para DTR/LR/RFR/SVR; en Crucero y Alto Jahuel el patrón es mixto según el modelo.

**Análisis horario**: picos de error en 03:00, 08:00, 12:00 y 20:00 h, coincidiendo con transiciones de generación (entrada/salida de solar) y rampas de demanda.

**Importancia de variables** (permutation importance sobre RFR): domina el SMP rezagado (t−1) de la propia barra o de barras eléctricamente vecinas (ej. Alto Jahuel: importancia 0.72 para su propio SMP t−1). Generación solar rezagada aparece consistentemente en el top-5 pero con importancia bastante menor. Demanda neta rara vez relevante (solo 2/6 casos).

## 9. Conclusiones

- Ninguna metodología (M1/M2) es universalmente mejor — depende del modelo y de la barra.
- La congestión de transmisión es el factor más determinante del error, más que el modelo elegido.
- BR, RFR y SVR son consistentemente los mejores modelos.

## 10. Limitaciones declaradas por los autores

- Sin escalado/normalización de features.
- Sin remoción de outliers.
- Horizonte de solo 1 hora (no multi-step ni day-ahead).
- Un solo año de datos (no capturan variabilidad interanual).
- Excluyeron intencionalmente SHAP/interpretabilidad más profunda.
- No incluyeron viento ni radiación solar como variables meteorológicas explícitas (solo generación real por tecnología).

## 11. Trabajo futuro que proponen

Feature scaling, modelos híbridos, incorporar información de congestión de transmisión explícita, variables climáticas (viento, radiación solar), horizontes más largos (day-ahead, multi-step), análisis multi-año.

---

## Comparación con mi tesis (notas propias)

- **Barra en común: Crucero.** Mi TFT (sol_clima_LN) en Crucero: MAE=6.43, R²=0.895, vs. su mejor modelo (RFR) en Crucero: MAE=7.71, R²=0.87. Mi modelo le gana en la misma barra.
- Ellos usan solo variables **endógenas/internas al sistema** (SMP rezagado propio y de otras barras, generación real por tecnología, demanda neta) — ningún dato exógeno de clima/sol. Yo uso variables **exógenas** (elevación solar, temperatura, humedad, viento, precipitación, nubosidad) y nada de SMP rezagado explícito de otras barras (aunque el encoder de 168h del TFT sí ve el pasado de la propia serie).
- Su split de test (enero+julio "rodeados" por datos de entrenamiento del mismo año) es metodológicamente más débil que mi split cronológico puro (pasado→futuro, sin fuga temporal) para simular un escenario real de despliegue.
- Confirmación externa independiente de mi hallazgo de picos de error en horas de transición solar (ellos: 08:00 y 20:00; yo: mismo patrón).
- Puerto Montt como caso de alta congestión y peor desempeño en el paper — coincide con mi propia decisión de tratar P.MONTT aparte (features hídricas en vez de solares).
- Posible pendiente a explorar: variables internas del sistema (SMP de barras vecinas, generación real por tecnología) como features adicionales, dado que su análisis de importancia muestra que son las que más pesan en sus modelos.

---

## Anexo: Test de Wilcoxon signed-rank aplicado a mi tesis

Réplica de la Sección 3.4 del paper, pero comparando `prophet_clima` vs `sol_clima` (en vez de M1 vs M2) en las 7 barras × 2 arquitecturas (LN/DyT), pareando por hora exacta del test set. Implementado en `Resultados_resumen.ipynb`.

### Qué es y por qué se usa en vez de un t-test

Responde: dadas dos mediciones pareadas de lo mismo (error de prophet_clima y error de sol_clima en la misma hora exacta), ¿hay una diferencia sistemática, o es ruido? Un t-test pareado asume que las diferencias se distribuyen aprox. normal — supuesto que no se cumple con errores de precio eléctrico (colas pesadas, outliers). Wilcoxon es **no paramétrico**: no asume normalidad y es robusto a outliers porque no opera sobre los valores en sí, sino sobre sus **rangos**.

### Mecánica

1. Para cada hora *i* del test: `d_i = err_prophet_i - err_sol_i`.
2. Se descartan empates exactos (`d_i = 0`).
3. Se ordenan los `|d_i|` de menor a mayor → cada uno obtiene un rango.
4. Cada rango recupera el signo original de `d_i`.
5. Se suman los rangos positivos y los negativos por separado. Si son parecidos → no hay diferencia sistemática. Si uno domina → sí la hay.

Una hora con una diferencia de 100 USD/MWh pesa igual que cualquier otra en términos de rango (es "la más grande"), no arrastra el resultado proporcionalmente a su magnitud como sí lo haría una media — por eso es robusto a outliers puntuales.

### Hipótesis y lectura del p-value

- H0: la mediana de las diferencias pareadas es 0 (no hay diferencia sistemática).
- H1: sí la hay.
- p-value = probabilidad de observar una asimetría de rangos tan grande como la vista, *si H0 fuera cierta*. p < 0.05 → se rechaza H0 → hay diferencia real.
- Test de **dos colas** (`two-sided`): solo dice si hay diferencia, sin dirección.
- Test de **una cola** (`greater`): si es significativo, indica la dirección (quién es mejor).

### "Pareado" — por qué importa acá

Ambas observaciones vienen de la misma hora exacta, misma barra, misma arquitectura — el precio de esa hora específica afecta a ambos modelos por igual, así que se compara cómo reaccionan ambos modelos frente a las mismas condiciones exactas, no errores de horas distintas entre sí.

### Resultados obtenidos (test set, pareado por hora)

| Barra | Arq | MAE Prophet | MAE Sol | p (dos colas) | ¿Significativo? |
|---|---|---|---|---|---|
| ATACAMA | LN | 6.62 | 6.38 | 0.104 | No |
| CARDONES | LN | 6.78 | 6.40 | <0.001 | Sí, gana sol |
| CHARRUA | LN | 8.28 | 7.74 | <0.001 | Sí, gana sol |
| CRUCERO | LN | 6.76 | 6.43 | 0.056 | No (ver nota abajo) |
| P.AZUCAR | LN | 7.05 | 6.80 | <0.001 | Sí, gana sol |
| QUILLOTA | LN | 8.39 | 7.98 | <0.001 | Sí, gana sol |
| TARAPACA | LN | 6.80 | 6.78 | <0.001 | Sí, gana sol |
| ATACAMA | DyT | 6.92 | 6.40 | 0.552 | No |
| CARDONES | DyT | 6.88 | 6.57 | 0.121 | No |
| CHARRUA | DyT | 8.57 | 8.06 | <0.001 | Sí, gana sol |
| CRUCERO | DyT | 6.91 | 6.82 | <0.001 | Sí, gana sol |
| P.AZUCAR | DyT | 7.19 | 6.70 | <0.001 | Sí, gana sol |
| QUILLOTA | DyT | 8.81 | 8.59 | <0.001 | Sí, gana sol |
| TARAPACA | DyT | 7.16 | 6.62 | <0.001 | Sí, gana sol |

**Conclusión general**: en 10/14 combinaciones, la ventaja de sol_clima sobre prophet_clima es estadísticamente significativa (p<0.05), no solo una diferencia de MAE que podría ser azar muestral. En 4 combinaciones (ATACAMA LN/DyT, CARDONES DyT, CRUCERO LN) no hay diferencia significativa — ahí no se puede afirmar con rigor que un modelo es mejor que el otro, pese a que el MAE numérico se vea distinto.

**Caso destacado — CRUCERO LN**: el MAE dice que sol_clima gana (6.43 vs 6.76), pero mirando hora por hora, en el **58.9% de las horas prophet_clima tuvo el error MENOR** (mediana de la diferencia pareada = -0.05, prácticamente empate). La ventaja de sol_clima en el promedio viene de unos pocos outliers puntuales de prophet_clima (diferencia máxima de 49 USD/MWh a favor de sol en horas específicas), no de una ventaja sistemática hora a hora. Este matiz es invisible si solo se mira el MAE — es la razón principal por la que vale la pena usar este test en vez de solo comparar promedios.

### Limitación a declarar en la tesis

El test asume observaciones razonablemente independientes. En una serie horaria hay autocorrelación (errores de horas consecutivas correlacionados: si un modelo tiene un mal día, varias horas seguidas tienen error alto) — Wilcoxon no corrige esto, así que el p-value puede ser algo "optimista" (menos horas efectivamente independientes de las que asume). No invalida la conclusión, pero es honesto mencionarlo. El paper tampoco lo corrige, así que el nivel de rigor es equivalente al de la referencia.
