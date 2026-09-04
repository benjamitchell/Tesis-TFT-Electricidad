# Borrador — Paper para Applied Sciences (MDPI)

**Título de trabajo:** A Simplified Hybrid Deep Learning Approach with Locally-Weighted Adaptive Conformal Prediction for Electricity Price Forecasting in the Chilean Market

**Autores:** Edward Benjamín Mitchell García¹, Alejandro Jofré Cáceres¹
¹ Departamento de Ingeniería Matemática, Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile, Santiago, Chile

---

## Abstract (borrador, ~200 palabras máx.)

Se propone un pipeline híbrido de pronóstico horario del costo marginal para el Sistema Eléctrico Nacional (SEN) chileno, basado en un Temporal Fusion Transformer (TFT) alimentado con la posición solar como variable exógena (S+C), en vez de la descomposición estadística de Prophet usada en trabajos previos (P+C). S+C supera a P+C en 14 de 16 combinaciones de barra/arquitectura evaluadas, y el análisis de interpretabilidad de la Variable Selection Network confirma que el modelo efectivamente aprende a apoyarse en la señal solar. Para cuantificar la incertidumbre, se propone LW-ACP (Locally-Weighted Adaptive Conformal Prediction), que combina ponderación local por hora del día con adaptación online del nivel de cobertura de Adaptive Conformal Inference. LW-ACP logra cobertura empírica dentro de 0.06 puntos porcentuales del nominal en las ocho barras del SEN, con intervalos 16.1% más angostos que Split CP clásico, superando a cada mecanismo aplicado por separado. Un análisis cuantitativo de errores extremos revela un patrón sistemático de desfase de fase de ~1 hora en la transición solar diaria, presente en las ocho barras, que motiva y explica la necesidad de ambos mecanismos de LW-ACP. El modelo se compara favorablemente con literatura reciente de machine learning clásico para el mismo mercado.

*(Ajustar longitud exacta una vez armado el cuerpo — MDPI pide ~200 palabras, un solo párrafo, sin citas ni abreviaciones sin definir.)*

**Keywords:** electricity price forecasting; Temporal Fusion Transformer; conformal prediction; uncertainty quantification; solar position; Chilean electricity market

---

## Estructura (orden MDPI: Intro → Results → Discussion → Materials and Methods → Conclusions)

### 1. Introduction
- Contexto: mercado eléctrico chileno, alta penetración solar, volatilidad del costo marginal.
- Estado del arte: modelos estadísticos clásicos (ARIMA) → limitaciones → deep learning (TFT) → modelos híbridos con descomposición (Prophet).
- Vacío en la literatura: (a) predicciones puntuales sin cuantificación de incertidumbre con garantías, (b) falta de comparación sistemática de estrategias de *features* exógenas simples vs. complejas.
- Contribución del paper (2 frases, una por cada foco): S+C vs P+C + LW-ACP.
- *Candidato para la cita de Zhao et al. (2025) aquí, reforzando el punto (a).*

### 2. Results
*(Nota: en MDPI Results va ANTES de Materials and Methods — hay que redactar asumiendo que el método ya se explicó en un preprint/versión previa, o incluir un resumen breve de método al inicio de Results si el journal lo permite; revisar en la plantilla real cómo lo resuelven otros papers de MDPI con esta estructura)*
- 2.1 S+C vs. P+C: comparación central (14/16, tabla resumen — no las 8 barras completas, solo agregado + 1-2 barras de ejemplo)
- 2.2 LN vs. DyT (resumen del hallazgo, sin el detalle completo del test de Wilcoxon barra por barra)
- 2.3 Interpretabilidad de S+C (tabla agregada, 1 figura si el espacio lo permite)
- 2.4 LW-ACP: linaje SCP→ACP→LW-ACP (la tabla de 4 métodos, la más "vendible" del paper)
- 2.5 Análisis de errores extremos (versión condensada: la síntesis de 8 barras, sin el detalle de Anexo)
- 2.6 Comparación con literatura (León et al., condensado a 1 tabla)
- Puerto Montt y granularidad: candidatos a quedar fuera del paper principal o como sección muy breve / material suplementario — decidir con Jofré si aportan al mensaje central o distraen.

### 3. Discussion
- Por qué S+C simplifica sin sacrificar precisión (la posición solar como señal física exacta vs. descomposición estadística ajustada).
- Por qué LW-ACP funciona: los dos mecanismos son complementarios, no redundantes (conectar con el hallazgo de desfase de fase).
- Limitaciones (versión condensada de las de Conclusiones: EnbPI no implementado, Puerto Montt no resuelto, granularidad exploratoria).
- Comparación con el estado del arte (Zhao et al., León et al.) — qué aporta este trabajo que ellos no tienen.

### 4. Materials and Methods
- 4.1 Datos: fuente, período, barras del SEN.
- 4.2 Arquitectura TFT + LayerNorm/DyT.
- 4.3 Construcción de *features*: P+C vs. S+C.
- 4.4 Conformal Prediction: Split CP, ACI, LW Split CP, LW-ACP (con la formulación matemática completa, esto sí va detallado).
- 4.5 Métricas de evaluación.

### 5. Conclusions
- Versión condensada de las Conclusiones de la tesis (~1 párrafo, sin las 4 secciones separadas).

---

## Pendientes / decisiones abiertas
- [ ] Confirmar ruta de la plantilla MDPI una vez descargada.
- [ ] Decidir con Jofré si Puerto Montt (hídrico) y granularidad temporal entran al paper principal o quedan para un segundo paper / material suplementario.
- [ ] Revisar longitud real vs. límite de Applied Sciences (recomendado 15-25 páginas en su formato de dos columnas).
- [ ] Migrar figuras: la mayoría de las de la tesis sirven, pero hay que revisar cuáles resumir/combinar dado el espacio.
- [ ] Bibliografía: reutilizar `bibliografia.bib`, filtrando solo las referencias efectivamente citadas en el paper (a diferencia de la tesis, que usa `\nocite{*}`).
