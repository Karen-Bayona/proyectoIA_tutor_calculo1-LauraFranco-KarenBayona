"""
Flujo completo:
  1. extraer_paginas()      Ingesta: lee el texto del PDF, página por página
  2. dividir_en_chunks()    Chunking: corta cada página en fragmentos con solapamiento
  3. embeber()              Vectorización: convierte texto en vectores con Gemini
  4. construir_indice()     Base vectorial: guarda vectores + metadatos en Chroma (en disco)
  5. buscar()               Recuperación: busca los top_k fragmentos más parecidos
  6. generar_respuesta()    Generación: Groq responde usando SOLO esos fragmentos
  7. responder()            Une todo, con historial de conversación
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
        clave = os.getenv("GEMINI_API_KEY")
        if not clave:
            raise RuntimeError("Falta la variable GEMINI_API_KEY en el .env")
        _cliente_gemini = genai.Client(api_key=clave)
    return _cliente_gemini


def get_cliente_groq():
    """Cliente Groq para Generación y Reformulación de preguntas."""
    global _cliente_groq
    if _cliente_groq is None:
        clave = os.getenv("GROQ_API_KEY")
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
          
            # Errores que no se arreglan reintentando
            if getattr(error, "code", None) in (400, 401, 403, 404):
                print("  Este error no es temporal: revisa el modelo o la clave en el .env.")
                raise
            # Cuota diaria agotada
            if "PerDay" in str(error):
                print("  Se agotó la cuota DIARIA de Gemini. Los fragmentos ya guardados se conservan:")
                print("  vuelve a ejecutar el mismo comando cuando se reinicie la cuota.")
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
    """Extrae el texto de una página. El modo 'layout' separa mejor las palabras en este PDF
    (el modo normal deja cosas como 'm ovim iento'); si falla, se usa el modo normal."""
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
    # Enviamos el nombre sin el prefijo "models/" (ej.: gemini-embedding-001)
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
        if os.getenv("RENDER"):  # Render define esta variable automáticamente
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
# 6. GENERACIÓN: Groq responde usando el contexto del documento
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


def generar_respuesta(pregunta, fragmentos, historial):
    contexto = "\n".join(
        f'<fragmento pagina="{f["pagina"]}">\n{f["texto"]}\n</fragmento>'
        for f in fragmentos
    )
    mensaje_usuario = (
        f"<historial>\n{formatear_historial(historial)}\n</historial>\n\n"
        f"<contexto>\n{contexto}\n</contexto>\n\n"
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
    """Minúsculas, sin tildes ni signos: '¡Hola!' -> 'hola'."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9 ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _respuesta_fija(texto):
    return {"respuesta": texto, "fuentes": [], "contextos": [], "mejor_similitud": 0.0}


# ----------------------------------------------------------------------------
# 7. FLUJO COMPLETO
# ----------------------------------------------------------------------------
def responder(pregunta, historial=None):
    historial = (historial or [])[-MAX_MENSAJES_HISTORIAL:]

    mensaje = _normalizar(pregunta)
    if mensaje in SALUDOS:
        return _respuesta_fija(RESPUESTA_SALUDO)
    if mensaje in DESPEDIDAS:
        return _respuesta_fija(RESPUESTA_DESPEDIDA)

    pregunta_busqueda = reformular_pregunta(pregunta, historial)
    candidatos = buscar(pregunta_busqueda)
    mejor = candidatos[0]["similitud"] if candidatos else 0.0
    fragmentos = [f for f in candidatos if f["similitud"] >= UMBRAL_SIMILITUD]

    if not fragmentos or mejor < UMBRAL_SIMILITUD:
        return {"respuesta": FRASE_FUERA_DE_CORPUS, "fuentes": [], "contextos": [], "mejor_similitud": mejor}

    respuesta = generar_respuesta(pregunta, fragmentos, historial)

    if FRASE_FUERA_DE_CORPUS in respuesta:
        fuentes = []
    else:
        fuentes = [
            {
                "documento": frag["documento"],
                "pagina": frag["pagina"],
                "fragmento": (
                    frag["texto"][:LARGO_FRAGMENTO_FUENTE] + "..."
                    if len(frag["texto"]) > LARGO_FRAGMENTO_FUENTE
                    else frag["texto"]
                ),
            }
            for frag in fragmentos[:3]
        ]

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "contextos": [frag["texto"] for frag in fragmentos],
        "mejor_similitud": mejor,
    }
