# Tutor de Cálculo con RAG: Avance 2

Asistente conversacional experto en **derivadas**, basado en RAG (Retrieval-Augmented Generation). Responde preguntas a partir de un documento fuente, cita la página de la que sale cada respuesta y reconoce cuándo la información no está en el corpus.

**Autoras:** Laura Franco y Karen Bayona
**Aplicación desplegada:** https://tutor-calculo-rag-karenbayona-laurafranco.onrender.com
**Repositorio:** https://github.com/Karen-Bayona/proyectoIA_tutor_calculo1-LauraFranco-KarenBayona

> **Nota sobre el plan gratuito de Render:** el servicio se apaga tras 15 minutos sin tráfico. La primera visita después de una pausa puede tardar cerca de un minuto en responder. Abre la URL unos minutos antes de probarla.

---

## 1. Qué contiene este repositorio

| Archivo / carpeta | Contenido |
|---|---|
| `app.py` | Servidor Flask: rutas web, manejo del historial y comandos de línea (`--indexar`, `--evaluar`, `--comparar`) |
| `rag.py` | Pipeline RAG: extracción, chunking, embeddings, índice vectorial, recuperación y generación |
| `prompts.py` | System Prompt, ejemplos Few-Shot, delimitadores y formato de salida (heredados y refinados del Avance 1) |
| `derivadas.pdf` | Corpus del asistente |
| `chroma_db/` | Índice vectorial ya construido (persistente) |
| `preguntas_eval.json` | Conjunto de evaluación con respuestas de referencia |
| `resultados/` | Resultados de las evaluaciones con Ragas y gráfico comparativo |
| `templates/`, `static/` | Interfaz de chat (HTML y CSS) |
| `tutor_avance1.py` | Entrega del Avance 1 (conservada; también etiquetada como `avance-1`) |
| `requirements.txt`, `Procfile`, `.python-version` | Dependencias y configuración de despliegue |

---

## 2. Flujo RAG implementado

```
 derivadas.pdf
      │  (1) Ingesta: pypdf extrae el texto por página
      ▼
 Texto por página
      │  (2) Chunking: tamaño 1500, solape 150
      ▼
 Fragmentos + metadatos (documento, página, chunk)
      │  (3) Embeddings: gemini-embedding-001 (RETRIEVAL_DOCUMENT)
      ▼
 ChromaDB (índice persistente, similitud coseno)
      ▲
      │  (4) Recuperación: embedding de la pregunta (RETRIEVAL_QUERY),
      │      top_k = 4, filtro por umbral de similitud = 0.68
 Pregunta del usuario (+ historial)
      │
      ▼
 (5) Generación: LLM con System Prompt, Few-Shot, delimitadores y
     fragmentos recuperados → respuesta con fuentes
```

### Decisiones técnicas y justificación

| Etapa | Decisión | Justificación |
|---|---|---|
| Corpus | `derivadas.pdf` | Capítulo 6 «Derivadas» de *Cálculo diferencial e integral* (Prof. Javier Pérez, Universidad de Granada). Es un texto universitario con definiciones, teoremas, demostraciones y ejercicios resueltos sobre derivabilidad, reglas de derivación, Rolle, valor medio, L'Hôpital, Taylor y convexidad: justo el dominio del tutor. **Se partió de un libro completo de Cálculo 1, pero al indexarlo se agotó la cuota diaria gratuita de la API de embeddings de Gemini (el proceso se detenía al llegar a cerca de 1000).** Por eso se acotó el dominio del tutor a derivadas y se eligió un documento más corto. Al ser un único documento acotado, además, se puede comprobar si cada respuesta está respaldada por el texto. |
| Ingesta | `pypdf` en modo `layout`, texto página por página, con limpieza (ligaduras, guiones de fin de línea, espacios) | Conserva el número de página, necesario para citar la fuente. El modo `layout` separa mejor las palabras en este PDF que el modo normal. |
| Chunking | 1500 caracteres, solape 150 (10 %), dentro de cada página; el corte se ajusta al último punto (`. `) de la segunda mitad del fragmento si existe | Un fragmento de ese tamaño suele contener un enunciado completo (definición o teorema) con parte de su contexto o demostración. El solape evita que una definición o fórmula quede cortada justo en el límite entre dos fragmentos. Al cortar página por página, cada fragmento conserva su número de página para citarlo. |
| Embeddings | `gemini-embedding-001`, con tipo de tarea `RETRIEVAL_DOCUMENT` para los fragmentos y `RETRIEVAL_QUERY` para las preguntas | Modelo multilingüe que funciona con texto en español y está pensado para recuperación: distinguir documento y pregunta mejora la coincidencia entre una pregunta corta y un fragmento largo. Se usa por API, sin modelos locales, lo que importa porque el plan gratuito de Render solo tiene 512 MB de RAM. Los vectores son de 768 dimensiones. La indexación se hace en lotes de 5 fragmentos con pausa entre lotes y es **reanudable**: los fragmentos ya guardados se conservan, de modo que si se agota la cuota diaria se puede continuar al día siguiente. |
| Base vectorial | ChromaDB con `PersistentClient`, distancia coseno | Persistencia local simple; el índice viaja con el repositorio |
| Metadatos | `documento`, `pagina`, `chunk` | Permiten mostrar la fuente de cada respuesta |
| Recuperación | `top_k = 4`, umbral de similitud `0.68` | En la evaluación, la similitud mínima de las preguntas del documento fue 0.71 y la máxima de las preguntas fuera del documento fue 0.653. El umbral 0.68 queda en esa brecha y separa ambos grupos. `top_k = 4` dio mejor fidelidad que 3 (ver sección 3). |
| Generación | Groq (`openai/gpt-oss-120b`), temperatura 0.2 | Groq cumple dos funciones: **genera la respuesta final** a partir de la pregunta, el historial y los fragmentos recuperados (con el System Prompt y los delimitadores XML), y **reformula la pregunta** con el historial (temperatura 0) antes de buscar en el índice. Gemini se usa solo para los embeddings. La temperatura baja mantiene la respuesta fiel al texto. Cada proveedor se configura con su clave y su modelo por variables de entorno; las llamadas de generación se reintentan con espera si la API devuelve un error temporal, y si la reformulación falla se usa la pregunta original. Los saludos y despedidas se responden con frases fijas, sin consultar el índice ni el modelo. |

**Preguntas de seguimiento:** el navegador envía el historial de la conversación en cada turno. El servidor lo valida y limita su longitud, y la pregunta se reformula con ese contexto antes de buscar en el índice, para que preguntas como "¿y para qué sirve?" se resuelvan correctamente.

**Privacidad:** los documentos fuente y el índice permanecen en el repositorio. Al modelo solo se le envían los fragmentos recuperados para cada consulta, nunca el corpus completo.

**Fuera del corpus:** si ningún fragmento supera el umbral de similitud, la aplicación responde con una frase fija indicando que la información no está en el material, en lugar de inventar una respuesta.

---

## 3. Evaluación con Ragas

**Conjunto de evaluación:** `preguntas_eval.json`, con 20 preguntas y su respuesta de referencia: 16 sobre el documento y 4 fuera del corpus (Mundial 2018, ecuaciones diferenciales, series de Fourier y multiplicación de matrices). Las cuatro métricas de Ragas se calculan sobre las 16 preguntas del documento. Las 4 preguntas fuera del corpus miden el **control de alucinaciones** (si el asistente rechaza correctamente), y las 16 del documento sirven además para detectar **falsos rechazos**.

**Métricas:** `faithfulness`, `answer_relevancy`, `context_precision` y `context_recall`.

### Resultados

| Ejecución | Cambio respecto a la anterior | top_k | Umbral | faithfulness | answer_relevancy | context_precision | context_recall | Rechazo correcto (fuera del corpus) | Falsos rechazos (dentro del corpus) |
|---|---|---|---|---|---|---|---|---|---|
| `base` | Punto de partida | 4 | 0.45 | 0.951 | 0.804 | 0.849 | 1.000 | 3/4 | 0/16 |
| `umbral068` | Umbral 0.45 → 0.68 | 4 | 0.68 | **0.969** | 0.812 | 0.850 | 1.000 | **4/4** | 0/16 |
| `mejora2` | top_k 4 → 3 | 3 | 0.68 | 0.883 | 0.809 | **0.896** | 1.000 | 4/4 | 0/16 |

Parámetros constantes en las tres ejecuciones: chunk 1500, solape 150, `gemini-3.1-flash-lite` y `gemini-embedding-001`.

Gráfico comparativo: [`resultados/comparacion.png`](resultados/comparacion.png). Detalle de cada ejecución en `resultados/` (`*_detalle.csv`, `*_respuestas.json`, `*_resumen.json`).

> Cada métrica se promedia sobre 16 preguntas, por lo que diferencias de alrededor de 0.01 entre ejecuciones no son concluyentes. Las diferencias más grandes (control de alucinaciones, fidelidad con `top_k = 3` y precisión de contexto) son las que se interpretan a continuación. En la ejecución `base`, Ragas devolvió valor vacío en 2 celdas de `faithfulness` y 2 de `context_recall`; los promedios se calculan con las preguntas que sí tienen valor.

### Análisis (ejecución `base`, detalle en `resultados/base_detalle.csv`)

- **Métrica más baja: `answer_relevancy` (0.80).** Se atribuye a la **generación**. El valor es casi uniforme (entre 0.73 y 0.86 en las 16 preguntas) incluso cuando la respuesta es correcta y completa, lo que apunta a un efecto sistemático del estilo de respuesta y no a fallos en preguntas concretas. Las respuestas son largas, con enumeraciones, introducciones del tipo «Según el texto…» y aclaraciones no pedidas. La hipótesis (aún sin probar) es que ese formato aleja las respuestas de la pregunta original según la medida de Ragas. **Qué se haría:** ajustar el System Prompt y el formato de salida para dar respuestas más directas (la respuesta en la primera frase, sin aclaraciones adicionales) y volver a evaluar.
- **`faithfulness` (0.95):** los fallos puntuales son de **generación**, no de recuperación. En la pregunta sobre derivadas laterales (0.71), el modelo aplicó a un punto interior las equivalencias que el texto da para los extremos del intervalo. En monotonía (0.89) y en L'Hôpital (0.875) añadió precisiones que no aparecen en los fragmentos.
- **`context_precision` (0.85):** se atribuye a la **recuperación**. Con `top_k = 4` entran fragmentos poco útiles (ejercicios resueltos, contexto histórico): las preguntas sobre la recta tangente y sobre el teorema de Rolle puntúan 0.50.
- **`context_recall` (1.0):** el chunking y la recuperación traen toda la información necesaria; no es el cuello de botella.
- **Limitaciones del pipeline observadas en los fragmentos recuperados:** el texto extraído del PDF tiene símbolos matemáticos deformados y palabras pegadas, y el corte por caracteres hace que algunos fragmentos empiecen a mitad de palabra. Posibles mejoras: una extracción que respete la notación y un corte por párrafos o frases.

### Iteraciones de mejora

**Iteración 1 (adoptada): umbral de similitud 0.45 → 0.68**

- **Motivo:** con 0.45, una de las cuatro preguntas fuera del corpus pasaba el filtro y el modelo la respondía (rechazo correcto 3/4). Los datos mostraron una brecha limpia entre la similitud mínima de las preguntas del documento (0.71) y la máxima de las que no lo son (0.653), así que 0.68 las separa.
- **Resultado:** rechazo correcto de 3/4 a 4/4, `faithfulness` de 0.951 a 0.969, sin falsos rechazos (0/16) y sin pérdida de `context_recall`.

**Iteración 2 (evaluada y descartada): `top_k` 4 → 3**

- **Motivo:** mejorar `context_precision`, que era la métrica atribuida a la recuperación.
- **Resultado:** `context_precision` sube de 0.850 a 0.896, pero `faithfulness` baja de 0.969 a 0.883. `answer_relevancy` (0.812 → 0.809), el rechazo (4/4) y `context_recall` no cambian.
- **Decisión:** se mantiene `top_k = 4`, porque en un tutor es más importante que la respuesta se apoye fielmente en el texto que reducir el ruido del contexto. Con menos fragmentos, el modelo tiene menos respaldo para lo que afirma.

**Configuración final desplegada:** `top_k = 4`, umbral 0.68, chunk 1500 y solape 150.

---

## 4. Instalación y ejecución local

Requisitos: Python 3.14 (ver `.python-version`) y claves de API de Google Gemini y Groq.

```bash
git clone https://github.com/Karen-Bayona/proyectoIA_tutor_calculo1-LauraFranco-KarenBayona.git
cd proyectoIA_tutor_calculo1-LauraFranco-KarenBayona

python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz (nunca se sube al repositorio; el archivo `.gitignore` lo excluye):

```
GEMINI_API_KEY=tu_clave_de_gemini
GROQ_API_KEY=tu_clave_de_groq
```

El resto de parámetros (`TAM_CHUNK`, `SOLAPE`, `TOP_K`, `UMBRAL_SIMILITUD`, `TEMPERATURA`, modelos) tienen valores por defecto en `rag.py` y se pueden sobrescribir con variables de entorno.

### Comandos

```bash
python app.py                           # servidor web en http://127.0.0.1:5000
python app.py --indexar [--reiniciar]   # construye el índice vectorial (carpeta chroma_db/)
python app.py --evaluar [etiqueta]      # evalúa el RAG con Ragas (por defecto: base)
python app.py --comparar antes despues  # compara dos evaluaciones, p. ej. base mejora2
```

El repositorio ya incluye el índice construido, por lo que no es necesario ejecutar `--indexar` para usar la aplicación.

---

## 5. Despliegue en Render

1. Subir el proyecto a GitHub (incluyendo `chroma_db/`, y sin `.env`).
2. En [render.com](https://render.com), crear un **Web Service** conectado al repositorio.
3. Configuración:
   - **Branch:** `main`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --workers 1 --timeout 300`
   - **Instance type:** Free
4. En **Environment Variables** agregar `GEMINI_API_KEY` y `GROQ_API_KEY`. Las credenciales nunca están en el repositorio.
5. La versión de Python se toma de `.python-version`.

En producción la aplicación **no reconstruye el índice**: si faltara `chroma_db/`, falla con un mensaje claro en lugar de gastar la cuota de embeddings.

### Limitaciones del plan gratuito

- Se apaga tras 15 minutos sin tráfico; el primer acceso posterior tarda cerca de un minuto.
- Tiene un límite mensual de horas de instancia.
- Los archivos escritos en disco no persisten entre reinicios. Por eso el índice se sube ya construido y la aplicación solo lo lee.
- El historial de la conversación vive en el navegador, así que no se pierde si el servidor se reinicia.
- Los modelos gratuitos de Gemini y Groq tienen límites diarios de uso.

---

## 6. Uso de la aplicación

1. Abrir la URL pública y escribir una pregunta sobre derivadas.
2. La respuesta incluye las fuentes (documento y página) y el fragmento que la respalda.
3. Se pueden hacer preguntas de seguimiento que dependan de los turnos anteriores.
4. Si la pregunta no está relacionada con el corpus, el asistente lo indica.

---

## 7. Limitaciones y alcance

- **Alcance limitado a derivadas.** El proyecto partió de un libro completo de Cálculo 1, pero la indexación superó el límite gratuito de Gemini para embeddings, por lo que el corpus se redujo al capítulo de derivadas. Ampliarlo a todo Cálculo 1 exigiría un plan de pago de la API de embeddings o indexar por lotes en varios días.
- La calidad del texto extraído del PDF afecta a las fórmulas (símbolos deformados, palabras pegadas); ver análisis en la sección 3.

---

## 8. Avance 1

El Avance 1 (diseño del prompt) se conserva en `tutor_avance1.py` y en la etiqueta `avance-1` del repositorio. Sus decisiones de System Prompt, Few-Shot, delimitadores y formato de salida se reutilizaron y refinaron en `prompts.py`.
