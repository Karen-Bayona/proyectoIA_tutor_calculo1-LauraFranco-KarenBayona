# Tutor de Cálculo con RAG y Agentes: Avance 3

Asistente conversacional experto en **derivadas**, basado en RAG (Retrieval-Augmented Generation) y organizado como un **agente con herramientas**. Responde a partir de un documento fuente (citando la página), puede consultar una **lista cerrada de sitios web confiables** cuando el documento no alcanza, genera la referencia APA de su fuente y reconoce cuándo una pregunta está fuera de su alcance.

**Autoras:** Laura Franco y Karen Bayona
#
**Aplicación desplegada:** https://tutor-calculo-rag-karenbayona-laurafranco.onrender.com
#
**Repositorio:** https://github.com/Karen-Bayona/proyectoIA_tutor_calculo1-LauraFranco-KarenBayona

> **Nota sobre el plan gratuito de Render:** el servicio se apaga tras 15 minutos sin tráfico. La primera visita después de una pausa puede tardar cerca de un minuto en responder. Abre la URL unos minutos antes de probarla.

---

## 1. Qué contiene este repositorio

| Archivo / carpeta | Contenido |
|---|---|
| `app.py` | Servidor Flask: rutas web, manejo del historial y comandos de línea (`--indexar`, `--evaluar`, `--comparar`, `--probar-tool`, `--probar-casos`) |
| `rag.py` | Pipeline RAG y agente: extracción, chunking, embeddings, índice vectorial, recuperación, generación, **tools** (`buscar_en_base`, `buscar_web_segura`, `generar_referencia_apa`) y filtro de tema |
| `prompts.py` | System Prompt, ejemplos Few-Shot (incluido uno con fuente web), delimitadores, formato de salida, prompt de reformulación y **prompt del filtro de tema** |
| `derivadas.pdf` | Corpus del asistente |
| `chroma_db/` | Índice vectorial ya construido (persistente) |
| `preguntas_eval.json` | Conjunto de evaluación con respuestas de referencia |
| `resultados/` | Resultados de las evaluaciones con Ragas y gráfico comparativo |
| `templates/`, `static/` | Interfaz de chat (HTML y CSS): muestra las herramientas usadas y las fuentes separadas por origen |
| `tutor_avance1.py` | Entrega del Avance 1 (conservada; también etiquetada como `avance-1`) |
| `requirements.txt`, `Procfile`, `.python-version` | Dependencias (incluye `tavily-python`) y configuración de despliegue |

---

## 2. Novedades del Avance 3

| Novedad | Descripción |
|---|---|
| Agente con tools | El asistente decide entre tres herramientas: buscar en el documento, buscar en la web segura y generar la referencia APA |
| Búsqueda web segura | Tavily restringido a 4 dominios permitidos (sección 4) |
| Filtro de tema | Antes de salir a la web, un paso de verificación comprueba que la pregunta trate de cálculo diferencial o de su historia |
| Respaldo automático | Si el PDF no tiene información relevante, el agente intenta la web (siempre después del filtro de tema) |
| Fuentes separadas | La interfaz muestra por separado las fuentes de la base (PDF, con enlace a la página) y las fuentes web (con enlace a la URL) |
| Herramientas visibles | Cada respuesta indica qué herramientas se usaron |
| Rechazo coherente | Si el agente rechaza una pregunta, no muestra fuentes |
| Seguridad de claves | Ninguna clave está en el código: `GEMINI_API_KEY`, `GROQ_API_KEY` y `TAVILY_API_KEY` se leen de variables de entorno (con `.strip()`) |

---

## 3. Arquitectura del agente

La función `responder()` en `rag.py` orquesta las herramientas con reglas explícitas en código (no deja que el modelo decida solo a qué sitio salir). Así el comportamiento es predecible y auditable.

```
 Pregunta del usuario (+ historial)
      │
      ├─ ¿saludo o despedida?  ───────────────► respuesta fija (sin tools)
      │
      ├─ ¿pide "apa", "referencia" o "fuente"? ► generar_referencia_apa
      │
      ├─ Reformulación con historial (Groq, temperatura 0)
      │
      ├─ ¿pide explícitamente "busca en la web"?
      │       sí ► verificar_tema ──► NO: frase de rechazo
      │                          └──► SÍ: buscar_web_segura ► respuesta con fuentes web
      │
      ├─ buscar_en_base (PDF, umbral 0.68)
      │       con resultados ► generación con el contexto del PDF ► fuentes del PDF
      │
      └─ sin resultados en el PDF (respaldo)
              verificar_tema ──► NO: frase de rechazo
                             └──► SÍ: buscar_web_segura ► respuesta con fuentes web
```

### Herramientas

| Herramienta | Qué hace | Cuándo se usa |
|---|---|---|
| `buscar_en_base` | Busca en `derivadas.pdf` (embeddings + ChromaDB) los fragmentos con similitud ≥ 0.68 y devuelve el texto con documento y página | Preguntas conceptuales o teóricas de la materia (ruta por defecto) |
| `buscar_web_segura` | Consulta Tavily limitada a los dominios permitidos (máx. 3 resultados) y devuelve título, URL y contenido | El usuario pide buscar en la web, o el PDF no tiene información relevante y la pregunta pasa el filtro de tema |
| `generar_referencia_apa` | Devuelve la cita APA del documento base | El usuario pide la referencia, la fuente o el formato APA |
| `verificar_tema` | Filtro (Groq, temperatura 0) que responde SÍ o NO a si la pregunta trata de cálculo diferencial o de su historia | Siempre, justo antes de cualquier búsqueda web. Si el filtro falla por un error, rechaza por seguridad |

---

## 4. Búsqueda web segura

**Mecanismo:** `buscar_web_segura` usa la API de Tavily con el parámetro `include_domains`, de modo que solo se recuperan resultados de la lista de sitios permitidos. No se hace ninguna búsqueda abierta.

### Sitios permitidos

| Sitio permitido | Por qué es confiable para el dominio |
|---|---|
| `es.wikipedia.org` | Enciclopedia en español con artículos sobre conceptos de cálculo y biografías de matemáticos (por ejemplo, Guillaume de l'Hôpital), con referencias bibliográficas verificables. Se usa como apoyo y se cita siempre la URL, porque no es una fuente primaria |
| `mathshistory.st-andrews.ac.uk` | Archivo MacTutor de Historia de las Matemáticas de la Universidad de St Andrews. Es una fuente académica especializada en biografías de matemáticos y en el origen de teoremas y resultados |
| `analisismatematico.ugr.es` | Sitio del Departamento de Análisis Matemático de la Universidad de Granada, el mismo departamento y universidad del autor del texto base. Publica material docente de Cálculo y Análisis, por lo que es coherente con la notación y el nivel del corpus |
| `rae.es` | Real Academia Española: autoridad lingüística para el significado de los términos en español que aparecen en las explicaciones |

**Por qué una lista cerrada:** evita que el tutor use blogs, foros o páginas sin respaldo; todo lo que se muestra al estudiante viene de instituciones académicas o de referencia, y cada respuesta web incluye la URL de donde salió.

### Filtro de tema

Durante las pruebas se detectó que el respaldo web respondía preguntas ajenas al curso: ante «¿Cuál es la receta de la bandeja paisa?», Tavily encontraba el artículo de Wikipedia y el modelo lo usaba para responder. Pedirle al modelo que rechazara mediante el prompt no fue suficiente, así que se agregó una regla en código: antes de cualquier búsqueda web, `verificar_tema` comprueba que la pregunta sea de cálculo diferencial o de su historia (matemáticos, libros). Si no lo es, no se llama a Tavily y se responde la frase de rechazo.

### Ajustes del System Prompt

- La regla 1 permite responder con el contexto web (marcado con `[web: URL]`), resumirlo y mencionar la URL, sin agregar datos que no estén en el contexto.
- La regla 2 rechaza cuando el contexto no alcanza o la pregunta no tiene relación con el cálculo diferencial ni con su historia.
- Se agregó un ejemplo Few-Shot que muestra cómo responder con un fragmento web y citar su URL.

---

## 5. Flujo RAG implementado

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
| Recuperación | `top_k = 4`, umbral de similitud `0.68` | En la evaluación, la similitud mínima de las preguntas del documento fue 0.71 y la máxima de las preguntas fuera del documento fue 0.653. El umbral 0.68 queda en esa brecha y separa ambos grupos. `top_k = 4` dio mejor fidelidad que 3 (ver sección 8). |
| Generación | Groq (`openai/gpt-oss-120b`), temperatura 0.2 | Groq cumple tres funciones: **genera la respuesta final** a partir de la pregunta, el historial y el contexto recuperado (con el System Prompt y los delimitadores XML), **reformula la pregunta** con el historial (temperatura 0) antes de buscar, y ejecuta el **filtro de tema** (temperatura 0) antes de salir a la web. Gemini se usa solo para los embeddings. La temperatura baja mantiene la respuesta fiel al contexto. Cada proveedor se configura con su clave y su modelo por variables de entorno; las llamadas de generación se reintentan con espera si la API devuelve un error temporal, y si la reformulación falla se usa la pregunta original. Los saludos y despedidas se responden con frases fijas, sin consultar el índice ni el modelo. |

**Preguntas de seguimiento:** el navegador envía el historial de la conversación en cada turno. El servidor lo valida y limita su longitud, y la pregunta se reformula con ese contexto antes de buscar, para que preguntas como "¿y para qué sirve?" se resuelvan correctamente.

**Privacidad:** los documentos fuente y el índice permanecen en el repositorio. Al modelo solo se le envían los fragmentos recuperados (o los resultados web) para cada consulta, nunca el corpus completo.

**Fuera del alcance:** si ningún fragmento del PDF supera el umbral, el agente solo recurre a la web cuando la pregunta es de cálculo diferencial o de su historia. Si la pregunta no lo es, o la web segura no devuelve resultados, responde con una frase fija indicando que está fuera del material y de sus capacidades, sin mostrar fuentes, en lugar de inventar una respuesta.

---

## 6. Interfaz

La interfaz de chat (`templates/index.html` y `static/style.css`) muestra debajo de cada respuesta:

1. **Herramientas usadas:** por ejemplo, `buscar_en_base`, o `verificar_tema, buscar_web_segura`.
2. **Fuentes separadas por origen:**
   - **Fuentes de la base (PDF):** documento y página, con enlace que abre `derivadas.pdf` en esa página, más el fragmento que respalda la respuesta.
   - **Fuentes web:** título de la página con enlace a la URL original, más un extracto.
3. Si la pregunta fue rechazada, solo se muestra la frase de rechazo y las herramientas consultadas, sin fuentes.

Las fórmulas se dibujan con MathJax y el historial de la conversación vive en el navegador.

### Casos de prueba (comportamiento esperado)

| Pregunta | Ruta | Herramientas mostradas | Fuentes |
|---|---|---|---|
| ¿Qué es la derivada de una función en un punto? | PDF | `buscar_en_base` | Base (PDF) con página |
| Busca en la web la biografía de Guillaume de l'Hôpital | Web forzada | `verificar_tema`, `buscar_web_segura` | Web, con URL |
| ¿Cuál es la receta de la bandeja paisa? | PDF sin resultados, el filtro de tema rechaza | `buscar_en_base`, `verificar_tema` | Ninguna (frase de rechazo) |
| Dame la referencia APA | Referencia | `generar_referencia_apa` | Documento base |
| Calcula la derivada de f(x) = x^2 usando la definición | PDF, método socrático | `buscar_en_base` | Base (PDF) con página |

---

## 7. Problemas encontrados y soluciones

| Problema | Causa | Solución |
|---|---|---|
| La búsqueda web siempre terminaba en la frase de rechazo | La clave de Tavily estaba escrita (y truncada) dentro del código | La clave se lee de `TAVILY_API_KEY`; ninguna clave vive en el repositorio |
| `No module named 'tavily'` | La librería no estaba instalada en el entorno | Se agregó `tavily-python` a `requirements.txt` |
| Error de encabezado en Render (`Invalid leading whitespace... return character`) | La variable de entorno se guardó con un salto de línea al final | Las claves se leen con `.strip()`, y en Render se vuelve a pegar el valor sin Enter ni espacios |
| El modelo se negaba a usar el contexto web | La regla 1 del System Prompt prohibía fuentes distintas del PDF | Se reescribieron las reglas 1 y 2 y se agregó un ejemplo con fuente web |
| La consulta a Tavily llevaba la orden completa («Busca en la web...») y se borraban los apóstrofes de «l'Hôpital» | Limpieza de texto demasiado agresiva | `limpiar_consulta_web()` quita solo la orden y se conserva la pregunta original |
| El agente respondía preguntas ajenas al curso (receta de bandeja paisa) usando Wikipedia | El respaldo web aceptaba cualquier tema | Filtro `verificar_tema` antes de toda búsqueda web |
| Se veían fuentes web junto a una respuesta de rechazo | Las fuentes se armaban antes de revisar si el modelo rechazó | Si la respuesta es la frase de rechazo, la lista de fuentes queda vacía |
| «apa» activaba la referencia dentro de otras palabras (por ejemplo, «capa») | Búsqueda de subcadena | Se usa una expresión regular con límites de palabra |

---

## 8. Evaluación con Ragas

> **Alcance de estos resultados:** las métricas son las del Avance 2 y miden el pipeline RAG sobre el PDF (recuperación, umbral y generación). En el Avance 3 las preguntas que el PDF no responde pueden pasar por el filtro de tema y la web segura, de modo que el control de alucinaciones ahora depende también de ese filtro.

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

## 9. Instalación y ejecución local

Requisitos: Python 3.14 (ver `.python-version`) y claves de API de Google Gemini, Groq y Tavily.

```bash
git clone https://github.com/Karen-Bayona/proyectoIA_tutor_calculo1-LauraFranco-KarenBayona.git
cd proyectoIA_tutor_calculo1-LauraFranco-KarenBayona

python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz (nunca se sube al repositorio; el archivo `.gitignore` lo excluye). Los valores van sin comillas ni espacios:

```
GEMINI_API_KEY=tu_clave_de_gemini
GROQ_API_KEY=tu_clave_de_groq
TAVILY_API_KEY=tu_clave_de_tavily
```

El resto de parámetros (`TAM_CHUNK`, `SOLAPE`, `TOP_K`, `UMBRAL_SIMILITUD`, `TEMPERATURA`, modelos) tienen valores por defecto en `rag.py` y se pueden sobrescribir con variables de entorno.

### Comandos

```bash
python app.py                           # servidor web en http://127.0.0.1:5000
python app.py --indexar [--reiniciar]   # construye el índice vectorial (carpeta chroma_db/)
python app.py --evaluar [etiqueta]      # evalúa el RAG con Ragas (por defecto: base)
python app.py --comparar antes despues  # compara dos evaluaciones, p. ej. base mejora2
python app.py --probar-tool             # prueba la tool buscar_en_base sin usar el LLM
python app.py --probar-casos            # ejecuta casos de prueba del agente por consola
```

El repositorio ya incluye el índice construido, por lo que no es necesario ejecutar `--indexar` para usar la aplicación.

---

## 10. Despliegue en Render

1. Subir el proyecto a GitHub (incluyendo `chroma_db/`, y sin `.env`).
2. En [render.com](https://render.com), crear un **Web Service** conectado al repositorio.
3. Configuración:
   - **Branch:** `main`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --workers 1 --timeout 300`
   - **Instance type:** Free
4. En **Environment Variables** agregar `GEMINI_API_KEY`, `GROQ_API_KEY` y `TAVILY_API_KEY`. Cada valor se pega sin espacios ni saltos de línea. Las credenciales nunca están en el repositorio.
5. La versión de Python se toma de `.python-version`.
6. Con el despliegue automático activado, cada `git push` a `main` actualiza la aplicación.

En producción la aplicación **no reconstruye el índice**: si faltara `chroma_db/`, falla con un mensaje claro en lugar de gastar la cuota de embeddings.

### Limitaciones del plan gratuito

- Se apaga tras 15 minutos sin tráfico; el primer acceso posterior tarda cerca de un minuto.
- Tiene un límite mensual de horas de instancia.
- Los archivos escritos en disco no persisten entre reinicios. Por eso el índice se sube ya construido y la aplicación solo lee.
- El historial de la conversación vive en el navegador, así que no se pierde si el servidor se reinicia.
- Los modelos gratuitos de Gemini y Groq, y el plan gratuito de Tavily, tienen límites de uso.

---

## 11. Uso de la aplicación

1. Abrir la URL pública y escribir una pregunta sobre derivadas.
2. La respuesta indica las herramientas usadas y las fuentes (base o web) que la respaldan.
3. Para consultar la web, escribir por ejemplo «Busca en la web la biografía de Guillaume de l'Hôpital».
4. Para obtener la cita, escribir «Dame la referencia APA».
5. Se pueden hacer preguntas de seguimiento que dependan de los turnos anteriores.
6. Si la pregunta no está relacionada con el cálculo diferencial, el asistente lo indica y no muestra fuentes.

---

## 12. Limitaciones y alcance

- **Alcance limitado a derivadas.** El proyecto partió de un libro completo de Cálculo 1, pero la indexación superó el límite gratuito de Gemini para embeddings, por lo que el corpus se redujo al capítulo de derivadas. Ampliarlo a todo Cálculo 1 exigiría un plan de pago de la API de embeddings o indexar por lotes en varios días.
- La calidad del texto extraído del PDF afecta a las fórmulas (símbolos deformados, palabras pegadas); ver análisis en la sección 8.
- **Orquestación por reglas.** Las herramientas se eligen con reglas explícitas en código (palabras clave, umbral de similitud y filtro de tema), no mediante tool calling del modelo. Es más predecible, pero menos flexible ante peticiones ambiguas.
- **Filtro de tema.** Depende de un modelo de lenguaje: una pregunta de cálculo muy ambigua podría rechazarse por error, y una pregunta ajena muy bien disfrazada podría pasar.
- **Web restringida.** Solo se consultan 4 sitios, así que preguntas de actualidad o de otros temas quedan fuera. Wikipedia es una fuente de apoyo, no primaria.
- La referencia APA está fijada para el documento base (`derivadas.pdf`).

---

## 13. Avances anteriores

- **Avance 1:** diseño del prompt, conservado en `tutor_avance1.py` y en la etiqueta `avance-1` del repositorio. Sus decisiones de System Prompt, Few-Shot, delimitadores y formato de salida se reutilizaron y refinaron en `prompts.py`.
- **Avance 2:** pipeline RAG completo (ingesta, chunking, embeddings, ChromaDB), interfaz web, despliegue en Render y evaluación con Ragas (sección 8).
