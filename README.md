# Tutor Académico Personalizado — Cálculo 1

## Descripción del proyecto

Este proyecto forma parte del desarrollo de un **Asistente Experto basado en RAG y Agentes**, un sistema de IA que funciona de manera local (preservando la privacidad de los datos) capaz de "leer" una base de conocimientos específica, responder preguntas y realizar tareas.

Para este proyecto se eligió el enfoque de **Tutor Académico Personalizado**, orientado a la asignatura de **Cálculo 1** (límites, derivadas y aplicaciones de la derivada). El asistente está diseñado para apoyar a estudiantes en el aprendizaje de estos temas, respondiendo únicamente con base en el material de la asignatura (guías, apuntes, libros) que se le proporcione como contexto.

## ¿Qué hace el asistente?

El tutor puede responder de dos maneras distintas, dependiendo de lo que necesite el estudiante:

- **Modo socrático**: cuando el estudiante está aprendiendo un concepto por primera vez, el asistente no da la respuesta de inmediato, sino que hace preguntas que lo guían a encontrarla por sí mismo.
- **Modo directo**: cuando el estudiante ya intentó resolver un ejercicio y quiere verificar su procedimiento, o hace una pregunta conceptual directa, el asistente da la explicación completa paso a paso.

En ambos casos, todas las respuestas se presentan en un formato consistente de **pasos numerados**, para facilitar el seguimiento del razonamiento matemático.

## Avance 1: Diseño de Prompts

Este avance se centra en el diseño y la estructuración del comportamiento del asistente, sin conectarlo todavía a un modelo real. Incluye:

### 1. System Prompt
Define la identidad del asistente (Tutor de Cálculo 1), su comportamiento pedagógico (socrático vs. directo), el formato de salida obligatorio (pasos numerados) y sus restricciones (no inventar fórmulas, no resolver tareas completas sin participación del estudiante).

### 2. Few-Shot Prompting
Se incluyeron 6 ejemplos de pares pregunta-respuesta que muestran al modelo el comportamiento y formato esperado, cubriendo distintos temas de Cálculo 1 (límites, derivadas básicas, regla de la cadena, optimización, razones de cambio) y los tres modos de respuesta (socrático, directo, conceptual).

### 3. Estrategias de Delimitadores
Se usan **etiquetas XML** (`<instrucciones_sistema>`, `<ejemplos>`, `<contexto_recuperado>`, `<pregunta_estudiante>`) para separar claramente cada parte del prompt final. Esto evita ambigüedad entre lo que es una instrucción, un ejemplo, el contexto recuperado por el sistema de RAG, y la pregunta real del estudiante.

### 4. Configuración del sistema
El modelo a utilizar es de la familia **Gemini** (vía la librería `google-genai`), con una **temperatura de 0.2** — un valor bajo elegido deliberadamente para priorizar precisión y consistencia matemática por encima de la creatividad, minimizando el riesgo de que el modelo invente pasos o fórmulas incorrectas.

## Estructura del código

El archivo `tutor_avance1.py` contiene:

- `SYSTEM_PROMPT`: el texto completo del system prompt.
- `FEW_SHOT_EXAMPLES`: la lista de los 6 ejemplos few-shot.
- `MODEL_CONFIG`: la configuración del modelo (nombre, temperatura, tokens máximos).
- `_formatear_ejemplos()`: función auxiliar que convierte los ejemplos a formato XML.
- `construir_prompt()`: función principal que ensambla el prompt final, uniendo instrucciones, ejemplos, contexto recuperado y la pregunta del estudiante.

## Cómo ejecutarlo

No se necesita ninguna instalación adicional, solo tener Python instalado.

```bash
python tutor_avance1.py
```

Esto imprime en la terminal el prompt completo ya ensamblado con delimitadores XML, listo para enviarse al modelo.
