"""
app.py - Tutor de Cálculo: Derivadas (servidor web + evaluación + línea de comandos).

El pipeline RAG está en rag.py y los prompts en prompts.py. Este archivo contiene:
  1. Importaciones y configuración
  2. Evaluación con Ragas (preguntas en preguntas_eval.json)
  3. Comparación antes/después de dos evaluaciones
  4. Servidor web Flask (rutas / y /chat)
  5. Línea de comandos

Uso:
    python app.py                           arranca el servidor web
    python app.py --indexar [--reiniciar]   construye el índice vectorial (carpeta chroma_db/)
    python app.py --evaluar [etiqueta]      evalúa el RAG con Ragas (por defecto: base)
    python app.py --comparar antes despues  compara dos evaluaciones (ej.: --comparar base mejora1)

En la nube:
    gunicorn app:app --workers 1 --timeout 300

El servidor NO guarda conversaciones: el navegador envía el historial en cada mensaje.
"""
import json
import os
import sys
import time

from flask import Flask, jsonify, render_template, request, send_from_directory


from prompts import FRASE_FUERA_DE_CORPUS
from rag import (
    BASE_DIR,
    MAX_MENSAJES_HISTORIAL,
    MODELO_EMBEDDINGS,
    MODELO_LLM,
    SOLAPE,
    TAM_CHUNK,
    TOP_K,
    UMBRAL_SIMILITUD,
    asegurar_indice,
    construir_indice,
    responder,
)

# True cuando se ejecuta algo como "python app.py --indexar" (no es el servidor web).
MODO_CLI = __name__ == "__main__" and len(sys.argv) > 1


# ============================================================================
# 2. EVALUACIÓN CON RAGAS   (python app.py --evaluar <etiqueta>)
# ----------------------------------------------------------------------------
# Qué hace:
#   1. Lee preguntas_eval.json (16 preguntas del documento + 4 que NO están en el documento).
#   2. Corre cada pregunta por el pipeline RAG real (responder).
#   3. Ragas calcula faithfulness, answer_relevancy, context_precision y context_recall
#      sobre las preguntas del documento (Gemini actúa como "juez").
#   4. Las preguntas fuera del documento miden el control de alucinaciones:
#      el sistema debe responder la frase de rechazo, sin inventar.
# Los resultados se guardan en la carpeta resultados/ (se crea sola al evaluar).
# Flujo típico:
#   python app.py --evaluar base       corrida inicial
#   (cambia un parámetro, por ejemplo TOP_K en .env)
#   python app.py --evaluar mejora1    corrida tras la mejora
#   python app.py --comparar base mejora1
# ============================================================================
CARPETA_RESULTADOS = os.path.join(BASE_DIR, "resultados")
ARCHIVO_PREGUNTAS = os.path.join(BASE_DIR, "preguntas_eval.json")
PAUSA_ENTRE_PREGUNTAS = 15 # segundos (cuidar el límite de la capa gratuita)
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ResponseRelevancy genera varias preguntas por respuesta; Gemini no soporta pedir varios
# candidatos a la vez, así que usamos strictness=1 (una pregunta generada por respuesta).
STRICTNESS_RELEVANCY = 1


def cargar_preguntas():
    """Lee el conjunto de evaluación: pregunta, referencia (ground truth) y en_corpus (True/False)."""
    with open(ARCHIVO_PREGUNTAS, encoding="utf-8") as f:
        return json.load(f)


def correr_pipeline(items):
    """Ejecuta cada pregunta por el RAG y guarda respuesta, contextos y similitud."""
    filas = []
    for n, item in enumerate(items, start=1):
        print(f"[{n}/{len(items)}] {item['pregunta']}")
        r = responder(item["pregunta"])
        filas.append(
            {
                "pregunta": item["pregunta"],
                "referencia": item["referencia"],
                "en_corpus": item["en_corpus"],
                "respuesta": r["respuesta"],
                "contextos": r["contextos"],
                "mejor_similitud": r["mejor_similitud"],
                "rechazo": FRASE_FUERA_DE_CORPUS in r["respuesta"],
            }
        )
        time.sleep(PAUSA_ENTRE_PREGUNTAS)
    return filas


def evaluar_con_ragas(filas):
    """Calcula las 4 métricas de Ragas para las preguntas que sí están en el documento."""
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    clave = os.environ["GEMINI_API_KEY"]
    juez = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=MODELO_LLM, google_api_key=clave, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=f"models/{MODELO_EMBEDDINGS}", google_api_key=clave)
    )

    # Solo preguntas del documento que recuperaron contexto (si no hay contexto, Ragas no puede medir)
    validas = [f for f in filas if f["en_corpus"] and f["contextos"]]
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": f["pregunta"],
                "retrieved_contexts": f["contextos"],
                "response": f["respuesta"],
                "reference": f["referencia"],
            }
            for f in validas
        ]
    )
    resultado = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=STRICTNESS_RELEVANCY),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=juez,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, timeout=180),
    )
    df = resultado.to_pandas()

    # Nombres de columnas estables (cambian un poco entre versiones de Ragas)
    renombrar = {}
    for col in df.columns:
        c = col.lower()
        if "faithfulness" in c:
            renombrar[col] = "faithfulness"
        elif "relevancy" in c or "relevance" in c:
            renombrar[col] = "answer_relevancy"
        elif "precision" in c:
            renombrar[col] = "context_precision"
        elif "recall" in c:
            renombrar[col] = "context_recall"
    return df.rename(columns=renombrar)


def evaluar(etiqueta="base"):
    import pandas as pd

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    asegurar_indice()

    filas = correr_pipeline(cargar_preguntas())
    with open(os.path.join(CARPETA_RESULTADOS, f"{etiqueta}_respuestas.json"), "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=2)

    print("\nEjecutando Ragas (puede tardar varios minutos)...")
    df = evaluar_con_ragas(filas)
    df.to_csv(os.path.join(CARPETA_RESULTADOS, f"{etiqueta}_detalle.csv"), index=False, encoding="utf-8-sig")

    metricas = {}
    for nombre in METRICAS:
        metricas[nombre] = round(float(df[nombre].mean(skipna=True)), 3) if nombre in df.columns else None

    dentro = [f for f in filas if f["en_corpus"]]
    fuera = [f for f in filas if not f["en_corpus"]]
    rechazos_correctos = sum(f["rechazo"] for f in fuera)
    falsos_rechazos = sum(f["rechazo"] for f in dentro)

    resumen = {
        "etiqueta": etiqueta,
        "config": {
            "TAM_CHUNK": TAM_CHUNK,
            "SOLAPE": SOLAPE,
            "TOP_K": TOP_K,
            "UMBRAL_SIMILITUD": UMBRAL_SIMILITUD,
            "MODELO_LLM": MODELO_LLM,
            "MODELO_EMBEDDINGS": MODELO_EMBEDDINGS,
        },
        "metricas": metricas,
        "control_alucinaciones": {
            "rechazo_correcto_fuera_del_documento": f"{rechazos_correctos}/{len(fuera)}",
            "falso_rechazo_dentro_del_documento": f"{falsos_rechazos}/{len(dentro)}",
        },
        "similitud": {
            "minima_preguntas_del_documento": min((f["mejor_similitud"] for f in dentro), default=None),
            "maxima_preguntas_fuera_del_documento": max((f["mejor_similitud"] for f in fuera), default=None),
        },
    }
    with open(os.path.join(CARPETA_RESULTADOS, f"{etiqueta}_resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print(f"\n=== Resultados: {etiqueta} ===")
    print(pd.Series(metricas, name="promedio").to_string())
    print("\nControl de alucinaciones:", resumen["control_alucinaciones"])
    print("Similitud (sirve para calibrar UMBRAL_SIMILITUD):", resumen["similitud"])
    print(f"\nArchivos guardados en {CARPETA_RESULTADOS}")


# ============================================================================
# 3. COMPARACIÓN DE EVALUACIONES   (python app.py --comparar <antes> <despues>)
# ----------------------------------------------------------------------------
# Imprime una tabla y guarda resultados/comparacion.png (gráfico de barras).
# ============================================================================
def cargar_resumen(etiqueta):
    with open(os.path.join(CARPETA_RESULTADOS, f"{etiqueta}_resumen.json"), encoding="utf-8") as f:
        return json.load(f)


def comparar(antes_nombre, despues_nombre):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    antes, despues = cargar_resumen(antes_nombre), cargar_resumen(despues_nombre)

    print(f"\nConfiguración {antes_nombre}:  {antes['config']}")
    print(f"Configuración {despues_nombre}: {despues['config']}\n")
    print(f"| Métrica | {antes_nombre} | {despues_nombre} | Cambio |")
    print("|---|---|---|---|")
    valores_antes, valores_despues = [], []
    for m in METRICAS:
        a, d = antes["metricas"].get(m), despues["metricas"].get(m)
        valores_antes.append(a or 0)
        valores_despues.append(d or 0)
        cambio = f"{d - a:+.3f}" if a is not None and d is not None else "n/d"
        print(f"| {m} | {a} | {d} | {cambio} |")
    print(f"\nRechazo correcto fuera del documento: {antes['control_alucinaciones']['rechazo_correcto_fuera_del_documento']}"
          f" -> {despues['control_alucinaciones']['rechazo_correcto_fuera_del_documento']}")

    # Gráfico de barras agrupadas
    x = range(len(METRICAS))
    ancho = 0.38
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - ancho / 2 for i in x], valores_antes, ancho, label=antes_nombre)
    plt.bar([i + ancho / 2 for i in x], valores_despues, ancho, label=despues_nombre)
    plt.xticks(list(x), METRICAS, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Puntaje (0 a 1)")
    plt.title("Evaluación Ragas: antes vs. después")
    plt.legend()
    plt.tight_layout()
    ruta = os.path.join(CARPETA_RESULTADOS, "comparacion.png")
    plt.savefig(ruta, dpi=150)
    print(f"\nGráfico guardado en {ruta}")


# ============================================================================
# 4. SERVIDOR WEB (Flask)
# ============================================================================
app = Flask(__name__)

# Si el índice vectorial no existe, se construye al arrancar (la primera vez tarda unos minutos).
# Con comandos como "python app.py --indexar" no se hace aquí, para no construirlo dos veces.
if not MODO_CLI:
    asegurar_indice()


def limpiar_historial(historial):
    """Valida lo que envía el navegador: solo mensajes con rol válido y texto corto."""
    limpio = []
    if not isinstance(historial, list):
        return limpio
    for mensaje in historial[-MAX_MENSAJES_HISTORIAL :]:
        if not isinstance(mensaje, dict):
            continue
        rol = mensaje.get("role")
        texto = str(mensaje.get("text", "")).strip()[:1500]
        if rol in ("user", "bot") and texto:
            limpio.append({"rol": rol, "texto": texto})
    return limpio


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/documento")
def documento():
    return send_from_directory(BASE_DIR, "derivadas.pdf")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    mensaje = str(data.get("message", "")).strip()[:1000]

    if not mensaje:
        return jsonify({"response": "Por favor, escribe una pregunta sobre derivadas.", "sources": []})

    historial = limpiar_historial(data.get("history"))

    try:
        resultado = responder(mensaje, historial)
    except Exception:
        app.logger.exception("Error al responder")
        return (
            jsonify({"response": "Tuve un problema para consultar al tutor. Intenta de nuevo en unos segundos.", "sources": []}),
            500,
        )

    return jsonify({"response": resultado["respuesta"], "sources": resultado["fuentes"]})


# ============================================================================
# 5. LÍNEA DE COMANDOS
# ============================================================================
AYUDA = """Uso:
    python app.py                           arranca el servidor web
    python app.py --indexar [--reiniciar]   construye el índice vectorial (carpeta chroma_db/)
    python app.py --evaluar [etiqueta]      evalúa el RAG con Ragas (por defecto: base)
    python app.py --comparar antes despues  compara dos evaluaciones (ej.: --comparar base mejora1)"""


def ejecutar_comando(argumentos):
    comando = argumentos[0].lstrip("-")  # acepta "--indexar" e "indexar"
    resto = argumentos[1:]
    if comando == "indexar":
        # Si se corta (por ejemplo por la cuota diaria), vuelve a ejecutarlo SIN --reiniciar:
        # continúa donde quedó. Con --reiniciar se borra el índice y se empieza de cero.
        construir_indice(reiniciar="--reiniciar" in resto or "reiniciar" in resto)
    elif comando == "evaluar":
        evaluar(resto[0] if resto else "base")
    elif comando == "comparar":
        if len(resto) != 2:
            print("Uso: python app.py --comparar <antes> <despues>   (por ejemplo: python app.py --comparar base mejora1)")
            sys.exit(1)
        comparar(resto[0], resto[1])
    else:
        print(f"Comando desconocido: {argumentos[0]}\n\n{AYUDA}")
        sys.exit(1)


if __name__ == "__main__":
    if MODO_CLI:
        ejecutar_comando(sys.argv[1:])
    else:
        app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")