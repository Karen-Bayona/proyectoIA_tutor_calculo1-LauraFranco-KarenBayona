"""
Flujo completo:
  1. extraer_paginas()      Ingesta: lee el texto del PDF, página por página
  2. dividir_en_chunks()    Chunking: corta cada página en fragmentos con solapamiento
  3. embeber()              Vectorización: convierte texto en vectores con Gemini
  4. construir_indice()     Base vectorial: guarda vectores + metadatos en Chroma (en disco)
  5. buscar()               Recuperación: busca los top_k fragmentos más parecidos
  6. generar_respuesta()    Generación: Groq responde usando el contexto recuperado
  7. responder()            Función central del Agente: coordina herramientas e historial
"""
import os
import re
import time
import unicodedata

from dotenv import load_dotenv

load_dotenv()

import chromadb
from google import genai
from google.genai import types
from groq import Groq
from pypdf import PdfReader

from prompts import (
    FRASE_FUERA_DE_CORPUS,
    PROMPT_REFORMULAR,
    PROMPT_TEMA,
    RESPUESTA_DESPEDIDA,
    RESPUESTA_SALUDO,
    SYSTEM_PROMPT,
)

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "derivadas.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
DOCUMENTO = "Cálculo diferencial e integral de funciones de una variable - Cap. 6: Derivadas"

MODELO_LLM = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MODELO_GROQ = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MODELO_EMBEDDINGS = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DIMENSIONES = 768
TEMPERATURA = float(os.getenv("TEMPERATURA", "0.2"))

TAM_CHUNK = int(os.getenv("TAM_CHUNK", "1500"))
SOLAPE = int(os.getenv("SOLAPE", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
UMBRAL_SIMILITUD = float(os.getenv("UMBRAL_SIMILITUD", "0.68"))

PAGINAS_OMITIDAS = set()
MIN_CHARS_PAGINA = 100
MIN_CHARS_CHUNK = 80
LOTE_EMBEDDINGS = 5
PAUSA_ENTRE_LOTES = 4.0
MAX_MENSAJES_HISTORIAL = 12
LARGO_FRAGMENTO_FUENTE = 300

NOMBRE_COLECCION = f"derivadas_c{TAM_CHUNK}_o{SOLAPE}"
ARCHIVO_MARCA = os.path.join(CHROMA_DIR, f"{NOMBRE_COLECCION}.ok")

_cliente_gemini = None
_cliente_groq = None
_coleccion = None


# ----------------------------------------------------------------------------
# Clientes de API y reintentos
# ----------------------------------------------------------------------------
def get_cliente():
    """Cliente Gemini para Embeddings."""
    global _cliente_gemini
    if _cliente_gemini is None:
        clave = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not clave:
            raise RuntimeError("Falta la variable GEMINI_API_KEY en el .env")
        _cliente_gemini = genai.Client(api_key=clave)
    return _cliente_gemini


def get_cliente_groq():
    """Cliente Groq para Generación y Reformulación de preguntas."""
    global _cliente_groq
    if _cliente_groq is None:
        clave = (os.getenv("GROQ_API_KEY") or "").strip()
        if not clave:
            raise RuntimeError("Falta la variable GROQ_API_KEY en el .env")
        _cliente_groq = Groq(api_key=clave)
    return _cliente_groq


def _con_reintentos(funcion, intentos=6, espera_base=10):
    """Ejecuta funcion() y reintenta si ocurre un error temporal de red o cuota."""
    for n in range(1, intentos + 1):
        try:
            return funcion()
        except Exception as error:
            print(f"  Detalle del error: {error}")

            if getattr(error, "code", None) in (400, 401, 403, 404):
                print("  Este error no es temporal: revisa el modelo o la clave en el .env.")
                raise
            if "PerDay" in str(error):
                print("  Se agotó la cuota DIARIA de Gemini. Los fragmentos ya guardados se conservan.")
                raise
            if n == intentos:
                raise
            espera = espera_base * n
            print(f"  La API devolvió un error ({type(error).__name__}); reintento en {espera}s...")
            time.sleep(espera)


# ----------------------------------------------------------------------------
# 1. INGESTA: extraer el texto del PDF
# ----------------------------------------------------------------------------
LIGADURAS = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def limpiar_texto(texto):
    for original, reemplazo in LIGADURAS.items():
        texto = texto.replace(original, reemplazo)
    texto = re.sub(r"-\n(?=[a-záéíóúñü])", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def _texto_de_pagina(pagina):
    try:
        return pagina.extract_text(extraction_mode="layout") or ""
    except Exception:
        return pagina.extract_text() or ""


def extraer_paginas(pdf_path=PDF_PATH):
    lector = PdfReader(pdf_path)
    for numero, pagina in enumerate(lector.pages, start=1):
        if numero in PAGINAS_OMITIDAS:
            continue
        try:
            texto = limpiar_texto(_texto_de_pagina(pagina))
        except Exception:
            continue
        if len(texto) >= MIN_CHARS_PAGINA:
            yield numero, texto


# ----------------------------------------------------------------------------
# 2. CHUNKING: partir cada página en fragmentos
# ----------------------------------------------------------------------------
def dividir_en_chunks(texto, tam=None, solape=None):
    tam = tam or TAM_CHUNK
    solape = solape if solape is not None else SOLAPE
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tam, len(texto))
        if fin < len(texto):
            corte = texto.rfind(". ", inicio + tam // 2, fin)
            if corte != -1:
                fin = corte + 1
        chunk = texto[inicio:fin].strip()
        if len(chunk) >= MIN_CHARS_CHUNK:
            chunks.append(chunk)
        if fin >= len(texto):
            break
        inicio = max(fin - solape, inicio + 1)
    return chunks


# ----------------------------------------------------------------------------
# 3. VECTORIZACIÓN: texto -> vector (embeddings con Gemini)
# ----------------------------------------------------------------------------
def embeber(textos, tipo, intentos=6, espera_base=10):
    vectores = []
    nombre_modelo = MODELO_EMBEDDINGS.replace("models/", "").strip()

    for i in range(0, len(textos), LOTE_EMBEDDINGS):
        lote = textos[i : i + LOTE_EMBEDDINGS]
        respuesta = _con_reintentos(
            lambda: get_cliente().models.embed_content(
                model=nombre_modelo,
                contents=lote,
                config=types.EmbedContentConfig(
                    task_type=tipo,
                    output_dimensionality=DIMENSIONES,
                ),
            ),
            intentos=intentos,
            espera_base=espera_base,
        )
        vectores.extend(e.values for e in respuesta.embeddings)
    return vectores


# ----------------------------------------------------------------------------
# 4. BASE DE DATOS VECTORIAL: ChromaDB
# ----------------------------------------------------------------------------
def obtener_coleccion():
    global _coleccion
    if _coleccion is None:
        cliente_db = chromadb.PersistentClient(path=CHROMA_DIR)
        _coleccion = cliente_db.get_or_create_collection(
            name=NOMBRE_COLECCION,
            metadata={"hnsw:space": "cosine"},
        )
    return _coleccion


def construir_indice(reiniciar=False):
    global _coleccion
    if reiniciar:
        cliente_db = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            cliente_db.delete_collection(NOMBRE_COLECCION)
        except Exception:
            pass
        if os.path.exists(ARCHIVO_MARCA):
            os.remove(ARCHIVO_MARCA)
        _coleccion = None
    coleccion = obtener_coleccion()

    print("Leyendo y fragmentando el PDF (tarda un minuto)...")
    ids, textos, metadatos = [], [], []
    for pagina, texto in extraer_paginas():
        for j, chunk in enumerate(dividir_en_chunks(texto)):
            ids.append(f"p{pagina}-c{j}")
            textos.append(chunk)
            metadatos.append({"documento": DOCUMENTO, "pagina": pagina, "chunk": j})
    print(f"  {len(ids)} fragmentos (tamaño={TAM_CHUNK}, solape={SOLAPE})")

    ya_guardados = set(coleccion.get(include=[])["ids"])
    pendientes = [i for i, id_ in enumerate(ids) if id_ not in ya_guardados]
    print(f"  {len(ya_guardados)} ya estaban guardados, faltan {len(pendientes)}")

    for inicio in range(0, len(pendientes), LOTE_EMBEDDINGS):
        indices = pendientes[inicio : inicio + LOTE_EMBEDDINGS]
        lote_textos = [textos[i] for i in indices]
        vectores = embeber(lote_textos, "RETRIEVAL_DOCUMENT", intentos=6, espera_base=10)
        coleccion.add(
            ids=[ids[i] for i in indices],
            embeddings=vectores,
            documents=lote_textos,
            metadatas=[metadatos[i] for i in indices],
        )
        print(f"  guardados {min(inicio + LOTE_EMBEDDINGS, len(pendientes))}/{len(pendientes)}")
        time.sleep(PAUSA_ENTRE_LOTES)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(ARCHIVO_MARCA, "w") as f:
        f.write(f"{coleccion.count()} fragmentos\n")
    print(f"Índice listo: {coleccion.count()} fragmentos en '{CHROMA_DIR}'")


def asegurar_indice():
    if not os.path.exists(ARCHIVO_MARCA):
        if os.getenv("RENDER"):
            raise RuntimeError(
                "Falta el índice vectorial (chroma_db/*.ok). "
                "Súbelo al repositorio; no se reconstruye en producción."
            )
        print("No hay índice completo todavía; construyéndolo...")
        construir_indice()


# ----------------------------------------------------------------------------
# 5. RECUPERACIÓN: búsqueda por similitud
# ----------------------------------------------------------------------------
def buscar(pregunta, top_k=None):
    top_k = top_k or TOP_K
    vector = embeber([pregunta], "RETRIEVAL_QUERY")[0]
    res = obtener_coleccion().query(
        query_embeddings=[vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    fragmentos = []
    for texto, meta, distancia in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        fragmentos.append(
            {
                "texto": texto,
                "pagina": meta["pagina"],
                "documento": meta["documento"],
                "similitud": round(1 - distancia, 3),
            }
        )
    return fragmentos


# ----------------------------------------------------------------------------
# TOOLS DEL AGENTE
# ----------------------------------------------------------------------------
def buscar_en_base(consulta: str) -> str:
    """Busca en el documento de cálculo (derivadas.pdf) fragmentos teóricos,
    definiciones y teoremas relevantes. Usar exclusivamente para preguntas
    conceptuales o teóricas de la materia."""
    TOP_K_TOOL = 4
    candidatos = buscar(consulta, top_k=TOP_K_TOOL)
    fragmentos = [f for f in candidatos if f["similitud"] >= UMBRAL_SIMILITUD]

    if not fragmentos:
        return "No se encontraron fragmentos relevantes en la base de conocimientos para esta consulta."

    resultado = []
    for f in fragmentos:
        resultado.append(
            f'[{f["documento"]} | Página {f["pagina"]} | Similitud: {f["similitud"]}]\n{f["texto"]}'
        )
    return "\n\n".join(resultado)


def generar_referencia_apa(documento: str = "derivadas.pdf") -> str:
    """Genera la cita y referencia bibliográfica formal en formato APA
    del documento base de la asignatura (derivadas.pdf). Usar cuando
    el usuario solicite la fuente bibliográfica o la citación oficial."""
    return (
        "Pérez, J. (s.f.). Cálculo diferencial e integral de funciones de una variable - "
        "Capítulo 6: Derivadas. Departamento de Análisis Matemático, Universidad de Granada."
    )


def buscar_web_segura(consulta: str) -> str:
    """Busca información pública o actual en la web (solo dominios permitidos)."""
    DOMINIOS_PERMITIDOS = [
        "es.wikipedia.org",
        "matematicas.ugr.es",
        "rae.es",
        "mathshistory.st-andrews.ac.uk",
    ]

    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not tavily_key:
        return "Error técnico en búsqueda web: falta TAVILY_API_KEY en el .env"

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_key)
        response = client.search(
            query=consulta,
            max_results=3,
            include_domains=DOMINIOS_PERMITIDOS,
        )
        resultados = []
        for item in response.get("results", []):
            resultados.append(
                f"[web: {item.get('url')}] Título: {item.get('title')}\n{item.get('content')}"
            )
        return "\n".join(resultados) if resultados else "No se encontraron resultados en los sitios web permitidos."
    except Exception as e:
        print(f"[TAVILY] Error: {e}")
        return f"Error técnico en búsqueda web: {str(e)}"


# ----------------------------------------------------------------------------
# 6. GENERACIÓN Y UTILIDADES
# ----------------------------------------------------------------------------
def formatear_historial(historial):
    lineas = []
    for mensaje in historial:
        quien = "Estudiante" if mensaje["rol"] == "user" else "Tutor"
        lineas.append(f"{quien}: {mensaje['texto']}")
    return "\n".join(lineas) if lineas else "(sin mensajes anteriores)"


def reformular_pregunta(pregunta, historial):
    if not historial:
        return pregunta
    try:
        cliente = get_cliente_groq()
        prompt = PROMPT_REFORMULAR.format(
            historial=formatear_historial(historial),
            pregunta=pregunta,
        )
        respuesta = cliente.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (respuesta.choices[0].message.content or pregunta).strip()
    except Exception as error:
        print(f"No se pudo reformular la pregunta ({error}); se usa la original.")
        return pregunta


def generar_respuesta(pregunta, contexto_texto, historial):
    mensaje_usuario = (
        f"<historial>\n{formatear_historial(historial)}\n</historial>\n\n"
        f"<contexto>\n{contexto_texto}\n</contexto>\n\n"
        f"<pregunta>\n{pregunta}\n</pregunta>"
    )
    cliente = get_cliente_groq()
    respuesta = _con_reintentos(
        lambda: cliente.chat.completions.create(
            model=MODELO_GROQ,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mensaje_usuario},
            ],
            temperature=TEMPERATURA,
        ),
        intentos=6,
        espera_base=15,
    )
    texto = (respuesta.choices[0].message.content or "").strip()
    return texto.replace("**", "")


SALUDOS = {
    "hola", "holi", "hey", "buenas", "buenos dias", "buenas tardes",
    "buenas noches", "que tal", "hola que tal", "hello", "hi",
}
DESPEDIDAS = {
    "gracias", "muchas gracias", "mil gracias", "ok gracias", "listo gracias",
    "chao", "adios", "hasta luego", "nos vemos", "bye",
}


def _normalizar(texto):
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9 ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _respuesta_fija(texto):
    return {
        "respuesta": texto,
        "fuentes": [],
        "contextos": [],
        "mejor_similitud": 0.0,
        "tools": [],
    }


def limpiar_consulta_web(texto):
    """Quita frases como 'busca en la web' para enviar a Tavily solo el tema."""
    texto = re.sub(
        r"(?i)\b(busca(r)?\s+en\s+(la\s+)?web|en\s+internet|en\s+la\s+web)\b",
        "",
        texto,
    )
    return texto.strip(" ,.:;")


def es_tema_calculo(pregunta):
    """Filtro de tema: solo se permite buscar en la web si la pregunta trata de cálculo
    diferencial o de su historia. Si el filtro falla, se rechaza por seguridad."""
    try:
        cliente = get_cliente_groq()
        respuesta = cliente.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": PROMPT_TEMA.format(pregunta=pregunta)}],
            temperature=0,
        )
        texto = _normalizar(respuesta.choices[0].message.content or "")
        print(f"[DEBUG] verificar_tema: {texto!r}")
        return texto.startswith("si")
    except Exception as error:
        print(f"[DEBUG] verificar_tema falló ({error}); se rechaza por seguridad.")
        return False


def _fuentes_web(contexto_texto):
    """Convierte el texto de buscar_web_segura en una lista de fuentes web (con URL)."""
    fuentes = []
    patron = r"\[web: (\S+?)\] Título: (.*?)\n(.*?)(?=\n\[web: |\Z)"
    for url, titulo, contenido in re.findall(patron, contexto_texto, flags=re.S):
        contenido = contenido.strip()
        recorte = contenido[:LARGO_FRAGMENTO_FUENTE]
        if len(contenido) > LARGO_FRAGMENTO_FUENTE:
            recorte += "..."
        fuentes.append(
            {
                "tipo": "web",
                "documento": titulo.strip() or url,
                "url": url,
                "pagina": None,
                "fragmento": recorte,
            }
        )
    return fuentes


def _resultado_web(pregunta, contexto_texto, historial, tools, similitud):
    """Genera la respuesta a partir del contexto web. Si el modelo rechaza, no muestra fuentes."""
    respuesta = generar_respuesta(pregunta, contexto_texto, historial)
    rechazada = FRASE_FUERA_DE_CORPUS in respuesta
    return {
        "respuesta": respuesta,
        "fuentes": [] if rechazada else _fuentes_web(contexto_texto),
        "contextos": [contexto_texto],
        "mejor_similitud": 0.0 if rechazada else similitud,
        "tools": tools,
    }


def _rechazo(tools):
    return {
        "respuesta": FRASE_FUERA_DE_CORPUS,
        "fuentes": [],
        "contextos": [],
        "mejor_similitud": 0.0,
        "tools": tools,
    }


# ----------------------------------------------------------------------------
# 7. FLUJO COMPLETO DEL AGENTE
# ----------------------------------------------------------------------------
def responder(pregunta, historial=None):
    historial = (historial or [])[-MAX_MENSAJES_HISTORIAL:]
    pregunta_limpia = pregunta.strip()
    mensaje = _normalizar(pregunta_limpia)

    if mensaje in SALUDOS:
        return _respuesta_fija(RESPUESTA_SALUDO)
    if mensaje in DESPEDIDAS:
        return _respuesta_fija(RESPUESTA_DESPEDIDA)

    if re.search(r"\b(apa|referencia|fuente)\b", mensaje):
        respuesta_apa = generar_referencia_apa()
        return {
            "respuesta": respuesta_apa,
            "fuentes": [
                {
                    "tipo": "base",
                    "documento": "derivadas.pdf",
                    "pagina": 1,
                    "fragmento": respuesta_apa,
                }
            ],
            "contextos": [respuesta_apa],
            "mejor_similitud": 1.0,
            "tools": ["generar_referencia_apa"],
        }

    # 2. Reformular pregunta si hay historial
    pregunta_optimizada = reformular_pregunta(pregunta, historial)

    # Si el usuario pide explícitamente buscar en la web, se usa Tavily de inmediato
    if "busca en la web" in mensaje or "en la web" in mensaje or "internet" in mensaje:
        consulta_web = limpiar_consulta_web(pregunta_optimizada)
        if not es_tema_calculo(consulta_web):
            return _rechazo(["verificar_tema"])
        tools = ["verificar_tema", "buscar_web_segura"]
        contexto_texto = buscar_web_segura(consulta_web)
        print(f"[DEBUG] consulta web: {consulta_web}")
        print(f"[DEBUG] contexto web: {contexto_texto[:300]}")

        if "No se encontraron resultados" in contexto_texto or "Error técnico" in contexto_texto:
            return _rechazo(tools)

        return _resultado_web(pregunta, contexto_texto, historial, tools, 0.85)

    # 3. Invocar la Tool de búsqueda vectorial en la base de conocimientos normal
    tools = ["buscar_en_base"]
    contexto_texto = buscar_en_base(pregunta_optimizada)

    # Evaluar si la base no encontró nada
    if "No se encontraron fragmentos relevantes" in contexto_texto:
        # Fallback automático a la web si el PDF tampoco tiene nada
        consulta_web = limpiar_consulta_web(pregunta_optimizada)
        if not es_tema_calculo(consulta_web):
            return _rechazo(["buscar_en_base", "verificar_tema"])
        tools = ["buscar_en_base", "verificar_tema", "buscar_web_segura"]
        contexto_texto = buscar_web_segura(consulta_web)
        print(f"[DEBUG] fallback web: {consulta_web}")
        print(f"[DEBUG] contexto web: {contexto_texto[:300]}")

        if "No se encontraron resultados" in contexto_texto or "Error técnico" in contexto_texto:
            return _rechazo(tools)

        return _resultado_web(pregunta, contexto_texto, historial, tools, 0.75)

    # 4. Generar respuesta con Groq usando el contexto del PDF
    respuesta = generar_respuesta(pregunta, contexto_texto, historial)

    candidatos = []
    fuentes = []

    if FRASE_FUERA_DE_CORPUS not in respuesta:
        candidatos = buscar(pregunta_optimizada, top_k=3)
        fuentes = [
            {
                "tipo": "base",
                "documento": frag["documento"],
                "pagina": frag["pagina"],
                "fragmento": (
                    frag["texto"][:LARGO_FRAGMENTO_FUENTE] + "..."
                    if len(frag["texto"]) > LARGO_FRAGMENTO_FUENTE
                    else frag["texto"]
                ),
            }
            for frag in candidatos
            if frag["similitud"] >= UMBRAL_SIMILITUD
        ]

    mejor_sim = candidatos[0]["similitud"] if candidatos else 0.0

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "contextos": [contexto_texto],
        "mejor_similitud": mejor_sim,
        "tools": tools,
    }