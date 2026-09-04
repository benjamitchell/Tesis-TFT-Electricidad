# Análisis de `Tesis.pdf` (versión actual) — qué cambiar, qué cortar, qué agregar

Fecha del análisis: 2026-07-31. Basado en lectura completa del PDF actual (95 págs) comparado contra el estado real del proyecto a esta fecha.

---

## 1. Problemas de "borrador sin terminar" — arreglar primero, son rápidos

Estos no son decisiones metodológicas, son cosas que quedaron a medio hacer y se notan mucho en una lectura de comité:

1. **Resumen**: literalmente dice "FALTA HACER UN RESUMEN". Obvio, pero mejor dejarlo para el final, cuando el resto del contenido esté estable (el resumen es lo último que se escribe bien).
2. **Apéndice A "Anexo"**: son puras 6 páginas de **Lorem Ipsum** (texto de relleno en latín falso). Esto es grave si se te escapa a la versión que entregas al comité — se ve como que ni siquiera revisaste el PDF final. O lo llenas con contenido real (tablas por barra que sacaste del cuerpo principal, código relevante, etc.) o lo eliminas del todo.
3. **Sección 4.6.5**, primer párrafo: quedaron variables sin rellenar, literal en el texto: *"La brecha entre el MAE promedio del modelo (X)USD/MWh... y la amplitud **madia** [sic] de los intervalos... (Y ) USD/MWh"*. Hay que poner los números reales y arreglar el typo "madia" → "media".
4. **Sección 1.5** (organización del documento): dice explícitamente *"esta sección la hago al final"* — y se nota, porque describe una estructura de capítulos (Cap 2 = marco teórico, Cap 3 = revisión de literatura, Cap 4 = mercado eléctrico y datos, Cap 5 = metodología, Cap 6 = resultados, Cap 7 = discusión, Cap 8 = conclusiones) que **no corresponde a la tabla de contenidos real** (que tiene: 1 Introducción, 2 Marco Teórico, 3 Metodología, 4 Resultados y Análisis). Hay que reescribirla desde cero al final, cuando sepas la estructura definitiva.
5. **Figuras con imagen faltante**: al menos tres referencias de imagen aparecen como texto crudo en vez de la figura (`imagenes/Posibles img/Alphas Aprendidos Precios.png DyT.png` en Fig 3.7, `imagenes/fig_error_dist_atacama.pdf` en Fig 4.7, `imagenes/fig_phase_shift_panel_2x2.pdf` en Fig 4.8). Verificar que esos archivos existan y compilen bien.

---

## 2. Lo grande: el documento describe una metodología que ya no es la que estás usando

Esto es lo más importante y confirma exactamente lo que sospechabas. Punto por punto:

### 2.1. No existe la comparación Sol+Clima (S+C) vs Prophet+Clima (P+C)
La Sección 3.4.1 ("Construcción de features y dataloaders") describe **un solo** feature set para el TFT: `known_reals` = las 4 componentes de Prophet (`trend`, `yearly`, `weekly`, `daily`) + `is_holiday`. **No hay ninguna mención a features solares** (`elevacion_solar`, `cos_elevacion`, `es_horario_verano`) en ninguna parte del documento. Confirma exactamente lo que dijiste. Este es el cambio más grande que falta: reescribir la Sección 3.4.1 para presentar ambos feature sets (S+C y P+C), y agregar al Capítulo 4 la comparación completa que ya tienes hecha (MAE en las 7 barras × LN/DyT, con el test de Wilcoxon).

### 2.2. Falta LW-ACP — que es tu aporte principal de CP ahora mismo
La Sección 2.6 (marco teórico de CP) y la 3.7/4.5 (metodología y resultados) cubren Split CP, EnbPI V2, ACI *offline*, ACI *online* y una comparación con regresión cuantílica. **No existe ninguna mención a la ponderación local por hora del día (σ_h) ni a LW-ACP.** Dado que en las últimas semanas este ha sido el hallazgo con más peso narrativo (σ_h calzando con las horas de transición solar, el ancho de banda subiendo justo ahí), es una omisión grande — probablemente la sección que más valor le agregaría al documento si la agregas. Yo la pondría como una subsección nueva dentro de 2.6 (teoría) y 3.7/4.5 (metodología/resultados), con la figura de error-por-hora vs σ_h que ya armamos.

Al mismo tiempo, **sí tienes AgACI en el marco teórico (Sección 2.6.4) pero nunca se implementa ni se reporta en Metodología ni Resultados** — es peso muerto, sácalo o impleméntalo.

### 2.3. Puerto Montt aparece tratado igual que las demás barras
Tabla 3.1, Sección 3.4.1, y todas las tablas de resultados (4.1 a 4.11) tratan a Puerto Montt como una barra más, con el mismo feature set (Prophet, sin nada hídrico). No hay ninguna mención a cotas de embalse, agua caída, ni a la decisión de tratar P.MONTT aparte. Dado el trabajo que hiciste (H+C vs S+C específicamente para esa barra, con la evidencia que armamos hoy mismo mostrando que S+C rinde ~2x peor ahí), esto necesita su propia subsección — y de hecho le da un argumento mucho más fuerte a la tesis: ya no es "decidimos tratar Puerto Montt distinto" sino "encontramos evidencia de que el enfoque estándar falla ahí, y por eso construimos una variante hídrica".

Dato curioso al leer las tablas: los números de Puerto Montt que ya tienes en el documento (MAE≈14.3, R²≈0.81-0.83 para TFT LN/DyT) son casi idénticos a los que **acabamos de calcular hoy mismo para "P.MONTT con S+C a nivel Horario"**. O sea, sin darte cuenta, los resultados que ya tenías escritos ERAN el experimento de control que te faltaba — coincidencia útil, pero confirma que hay que re-verificar todas las tablas contra los modelos actuales antes de darlas por buenas.

### 2.4. No existe el estudio de granularidad (Día/15min vs Horario)
No hay ninguna mención a comparar resolución diaria, 15 minutos u horaria. Esto es contenido enteramente nuevo por agregar, no una corrección — tiene sentido que quede pendiente hasta que termines de correr y evaluar esos experimentos.

### 2.5. El feature set descrito no coincide ni siquiera con la versión "vieja" (P+C) actual
Ojo con este detalle técnico: la Sección 3.4.1 describe `unknown_reals` = precio + **lags de 1/24/168 horas** + **medias/desviaciones móviles de 24/168 horas** + clima. Pero en tu pipeline actual (tanto S+C como P+C), `unknown_reals_price` es solo `['y_real'] + variables_clima`, **sin ningún lag ni rolling stat explícito** — confiando en que el encoder de 168h del TFT capture esa estructura temporal por sí solo. Este cambio metodológico (sacar los lags/rolling stats manuales) ya ocurrió en el código hace tiempo pero nunca se actualizó en el texto. Hay que decidir conscientemente: ¿vale la pena mencionar por qué se sacaron esas features (redundantes con lo que el TFT ya aprende del encoder), o simplemente actualizar la descripción a lo que hay ahora?

### 2.6. Fechas de los conjuntos de datos — verificar
Tabla 3.3 dice train hasta 2024-06, val hasta 2025-05, test hasta 2026-04. Dado que en el proyecto hubo un hito de "recuperar como un año extra de datos históricos" que gatilló buena parte de la reorganización reciente, **verifica que estas fechas sigan correspondiendo al split actual** antes de dar la tabla por válida — no lo pude confirmar desde este análisis.

---

## 3. Marco Teórico — coincido contigo, hay que acortar una parte y alargar otra

**Lo que sobra (tu instinto está bien encaminado):** la Sección 2.3.1-2.3.3 (mecanismo de atención genérico, atención multi-cabeza, arquitectura Transformer clásica completa con FFN y *positional encoding* senoidal) es una exposición de libro de texto de ~4 páginas que, en la práctica, **no se vuelve a usar en ningún resultado posterior** — el TFT ni siquiera usa *positional encoding* senoidal (usa LSTM encoder/decoder en su lugar, como tú mismo describes en 2.3.4). Yo la reduciría a un párrafo corto citando Vaswani et al. para quien quiera profundizar, y dejaría el detalle solo para lo que sí se usa después: la atención interpretable (2.3.4, porque la usas en el análisis de interpretabilidad del Cap. 4), la GRN/VSN (porque la importancia de variables sale de ahí), y la pérdida cuantílica (porque conecta con CP).

**Lo que no sobra:** la Sección 2.4 (LayerNorm vs DyT) sí es carga útil — es una comparación empírica propia de la tesis (Sección 4.4), así que el detalle matemático se justifica.

**Lo que falta (tu segundo punto):** el mercado eléctrico chileno y las barras están cubiertos en solo ~2 páginas en la Introducción (1.1), con datos generales del SEN pero sin profundizar en la topología de transmisión ni en por qué se eligieron justo estas 8 barras más allá de "representatividad geográfica". Acá hay una oportunidad directa: en la comparación que hicimos con el paper de León et al. (2026), ellos caracterizan formalmente la congestión de transmisión entre barras (su Tabla 1, con los mismos corredores que tú tienes: Crucero-Cardones, Cardones-P.Azúcar, etc.) y usan eso para justificar decisiones metodológicas. Vale la pena agregar una subsección con esa caracterización — te sirve además para justificar con más fuerza por qué Puerto Montt necesita tratamiento aparte (es la zona con más horas de desacople, 34-43% según ese paper).

---

## 4. Otros puntos menores que noté al leer

- El paper de León et al. (2026) que comparamos hoy — con el que le ganas en Crucero — es una referencia muy pertinente para el Estado del Arte / Discusión y todavía no está citada en ningún lado.
- La Sección 3.5.1 describe 6 estrategias de Stacking centradas todas en combinar Prophet con TFT — esto sigue siendo válido como comparación *interna* dentro de P+C, pero en la narrativa actual el Stacking pasa a un rol secundario (S+C, que no usa Prophet ni Stacking, es el que gana) — vale la pena reencuadrar esa sección para que quede claro que el Stacking es relevante dentro de la rama P+C, no el resultado final del modelo completo.
- El pie de página de la Sección 4.4.2 admite que una versión anterior del TFT-DyT dio resultados distintos y tuvo que rehacerse — dado cuánto ha cambiado el pipeline desde entonces (fix del bug de `incluir_features_horario`, fix del NaN de residuo, capacidad chica vs grande, etc.), yo directamente regeneraría todas las tablas de resultados desde los modelos actuales en vez de confiar en corridas antiguas, para no arrastrar ese tipo de inconsistencias.

---

## 5. Resumen ejecutivo — qué haría yo primero

1. Arreglar los "cabos sueltos" de la sección 1 (Lorem Ipsum, variables sin rellenar, figuras rotas) — es gratis y evita mala impresión.
2. Reescribir 3.4.1 para presentar S+C y P+C como dos ramas comparadas, no una sola.
3. Agregar la sección de LW-ACP a 2.6/3.7/4.5.
4. Agregar la sección de Puerto Montt (H+C vs S+C) con la evidencia que ya generamos.
5. Recortar 2.3.1-2.3.3, expandir la caracterización de barras/topología en 1.1.
6. Regenerar TODAS las tablas de resultados desde los modelos actuales antes de darlas por finales.
7. Dejar granularidad (D/15min) y ablation study para cuando esos experimentos terminen — no bloquean el resto.
