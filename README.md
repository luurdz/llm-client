# llm-client

Proyecto integrador del curso de AI Engineering. Cada unidad suma una parte sobre la anterior:

- **Unidad 1 — Cliente asíncrono de LLMs** (`llm_client/`, `main.py`): interfaz común para OpenAI, Anthropic y Gemini, con respuestas normales y en streaming, validación de la configuración con Pydantic y errores controlados.
- **Unidad 2 — Pipeline de procesamiento validado** (`pipeline/`, `run_pipeline.py`): pipeline con LangChain (LCEL) que extrae información técnica de un texto (un log de error o una descripción de arquitectura) y devuelve un objeto validado con Pydantic.
- **Unidad 3 — Sistema RAG local** (`rag/`, `run_ingest.py`, `run_rag.py`, `data/`): responde preguntas sobre los documentos de una agencia de viajes ficticia usando solo esos documentos, guardados en una base vectorial local (ChromaDB). Si la respuesta no está en los documentos, responde "No lo sé".

La versión exacta de cada entrega: unidad 1 en el tag [`unidad-1`](https://github.com/luurdz/llm-client/tree/unidad-1), unidad 2 en la rama [`entrega-2`](https://github.com/luurdz/llm-client/tree/entrega-2) y unidad 3 en la rama [`entrega-3`](https://github.com/luurdz/llm-client/tree/entrega-3).

## Requisitos

- Python 3.12 o superior (probado con Python 3.14)
- Una API key del proveedor que se quiera usar. Gemini tiene una capa gratuita: https://aistudio.google.com/apikey

## Instalación

Crear y activar un entorno virtual:

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Configuración

Todas las unidades usan el mismo `.env`. Copiar `.env.example` a `.env` y completar la API key del proveedor que se va a usar.

Windows (PowerShell):

```bash
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | Proveedor a usar: `openai`, `anthropic` o `gemini`. Por defecto: `gemini` |
| `LLM_MODEL` | Modelo del proveedor elegido, por ejemplo `gemini-flash-lite-latest`, `gpt-4o-mini` o `claude-opus-5-5` |
| `LLM_FALLBACK_MODEL` | Opcional (unidad 2). Modelo del mismo proveedor que se usa cuando falla el principal, por ejemplo `gemini-3.5-flash-lite` |
| `EMBEDDING_MODEL` | Unidad 3. Modelo de embeddings del proveedor elegido: `gemini-embedding-001` (Gemini) o `text-embedding-3-small` (OpenAI). Anthropic no tiene modelo de embeddings |
| `GOOGLE_API_KEY` | API key de Gemini |
| `OPENAI_API_KEY` | API key de OpenAI |
| `ANTHROPIC_API_KEY` | API key de Anthropic |

Solo es obligatoria la key del proveedor elegido. El archivo `.env` está en el `.gitignore`.

Para cambiar de proveedor, alcanza con cambiar `LLM_PROVIDER` y `LLM_MODEL` en el `.env`. No hace falta tocar el código.

## Unidad 1 — Cliente asíncrono de LLMs

```bash
python main.py
```

El script le pregunta "¿Qué es la entropía?" al proveedor configurado y corre tres pruebas:

1. **Modo normal**: espera la respuesta completa y la muestra.
2. **Modo streaming**: muestra la respuesta de a pedazos, a medida que llega.
3. **Prueba de resiliencia**: hace la misma pregunta con una API key inválida y muestra el error que devuelve el cliente. El programa sigue funcionando después del error.

Si la configuración está mal (proveedor no soportado, falta el modelo o la API key está vacía), el script muestra un error de configuración y termina.

### Diseño

- **Interfaz común**: todos los clientes heredan de `BaseLLMClient` e implementan `generate()` y `generate_stream()`.
- **Factory**: `AsyncLLMManager` elige la clase del cliente con un diccionario (`CLIENTS`) según `config.provider`. Para agregar un proveedor hay que crear su cliente y sumar una entrada al diccionario.
- **Validación**: `LLMConfig` valida el proveedor, el modelo, la temperatura (de 0 a 2), `max_tokens` (mayor que 0), el timeout y la API key, que se guarda como `SecretStr` para que no aparezca en los logs. Todos los clientes reciben este mismo objeto de configuración.
- **Asincronía**: todas las llamadas usan la versión asíncrona de cada SDK (`AsyncOpenAI`, `AsyncAnthropic` y `client.aio` en Gemini).
- **Streaming**: `generate_stream()` es un generador asíncrono que devuelve objetos `StreamChunk`. Cada chunk tiene `content` o `error`, así un error en el medio del stream se distingue del texto.
- **Manejo de errores**: cada cliente traduce los errores de su propio SDK (rate limit, conexión, errores del servidor, pedidos inválidos) a un `ModelResponse` o `StreamChunk` con el campo `error` completo. Todas las llamadas tienen un timeout (`asyncio.timeout`, configurable en `LLMConfig`). Los reintentos los hacen los SDKs: OpenAI y Anthropic reintentan dos veces por defecto, y el cliente de Gemini está configurado con 3 intentos.
- **Diferencias entre proveedores**: Gemini y Anthropic reciben el mensaje de sistema como un parámetro aparte, y Gemini usa el rol `model` en lugar de `assistant`. Cada cliente convierte los mensajes al formato de su proveedor.

## Unidad 2 — Pipeline de procesamiento validado

```bash
python run_pipeline.py
```

El script tiene dos partes y muestra los logs de cada paso:

1. **Single text (`ainvoke`)**: procesa un texto con `process_text()`.
2. **Batch (`abatch`)**: procesa los tres textos de ejemplo a la vez con `process_texts()`:
   - **normal**: un log de error con varias tecnologías.
   - **ambiguous**: un texto que habla de "la base de datos" y "el caché" sin decir cuáles (prueba de estrés).
   - **non_technical**: un texto sin información técnica.

### Ejemplo de salida

Para el caso `normal`:

```json
{
  "tecnologias": ["FastAPI", "PostgreSQL", "Kafka", "Redis"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio de pedidos en FastAPI tiene timeouts al llamar a payment-service porque el pool de conexiones de PostgreSQL está agotado y la tasa de aciertos del caché en Redis bajó al 5%."
}
```

Para los casos `ambiguous` y `non_technical`, el modelo devuelve una lista de tecnologías vacía, el validador la rechaza y la cadena reintenta tres veces. En el lote, el error ocupa el lugar de ese texto y los demás resultados no se pierden:

```
WARNING pipeline.chain: The response was invalid. Retry
WARNING pipeline.chain: The response was invalid. Retry
WARNING pipeline.chain: The response was invalid. Retry
--- ambiguous ---
Extraction failed: Failed to parse TechnicalEntities from completion {"technologies": [], ...
```

### Uso desde código

```python
from pipeline.chain import process_text, process_texts

entities = await process_text("SQL Server está caído")
print(entities.model_dump_json(by_alias=True, indent=2))

results = await process_texts(["SQL Server está caído", "Redis responde lento"])
```

- `process_text()` devuelve un `TechnicalEntities`, o lanza el error si la extracción falla.
- `process_texts()` devuelve una lista en el mismo orden que los textos. Cada elemento es un `TechnicalEntities` o el error de ese texto.
- Con `by_alias=True`, las claves del JSON salen en castellano.

### Diseño

- **Esquema** (`pipeline/schemas.py`): `TechnicalEntities` con `technologies`, `criticality_level` (enum `baja` / `media` / `alta`) y `technical_summary`. Los nombres del código están en inglés y las claves del JSON en castellano, con `serialization_alias`. Las descripciones de los campos y el docstring de la clase forman parte del esquema que recibe el modelo.
- **Prompt** (`pipeline/prompts.py`): `ChatPromptTemplate` con un mensaje de sistema y la variable `{text}`.
- **Modelo** (`llm_client/langchain_models.py`): `build_chat_model()` crea `ChatOpenAI`, `ChatAnthropic` o `ChatGoogleGenerativeAI` según `LLMConfig`, reutilizando la configuración y el `.env` de la unidad 1. El pipeline usa temperatura 0 para que el mismo texto dé siempre el mismo resultado.
- **Cadena** (`pipeline/chain.py`): `prompt | model.with_structured_output(TechnicalEntities, include_raw=True) | check_output`, con `.with_retry()` de hasta 3 intentos.
- **Resiliencia**:
  - Los errores de validación (`OutputParserException`, `ValidationError`) se reintentan.
  - `check_output()` revisa el `finish_reason` de la respuesta. Si el modelo cortó la respuesta por llegar a `max_tokens`, lanza `IncompleteOutputError` sin reintentar, porque con el mismo límite se cortaría de nuevo.
  - Si está configurado `LLM_FALLBACK_MODEL`, la cadena usa `.with_fallbacks()`: cuando el modelo principal falla por un error del proveedor (servidor caído, rate limit, timeout o conexión), repite el pedido con el modelo alternativo. Los errores de validación no pasan al modelo alternativo.
- **Lote**: `process_texts()` usa `.abatch()` con `max_concurrency=5`, para no superar el rate limit del proveedor, y `return_exceptions=True`, para que un texto con error no corte el lote.
- **Logs**: `logging` muestra el inicio de cada procesamiento, el modelo alternativo configurado, cada respuesta inválida antes de un reintento, las respuestas cortadas y las validaciones correctas.

### Qué agregué además de la consigna

- `process_texts()` con `.abatch()` para procesar varios textos a la vez.
- Modelo alternativo con `.with_fallbacks()`.
- Detección de respuestas cortadas por `max_tokens` (`finish_reason`).

## Unidad 3 — Sistema RAG local

El sistema tiene dos pasos: primero se indexan los documentos de `data/` y después se hacen las preguntas.

### 1. Ingesta

```bash
python run_ingest.py
```

Lee los archivos `.md` de `data/`, los limpia, los corta en fragmentos y los guarda con sus embeddings en una base ChromaDB local, en `vectorstore/`. Si la base ya tiene datos, no vuelve a indexar:

```
INFO rag.ingest: Indexed 12 chunks                                   ← primera ejecución
INFO rag.ingest: Vector store already has data, skipping indexing    ← siguientes ejecuciones
```

Para volver a indexar desde cero (por ejemplo, después de cambiar los documentos o el modelo de embeddings), borrar la carpeta `vectorstore/` y correr el script de nuevo.

### 2. Preguntas

```bash
python run_rag.py
```

Hace dos preguntas cuya respuesta está en los documentos y dos "preguntas trampa" cuya respuesta no está. Resultado:

```
=== Kind: in_context ===
Question: ¿Cuánto me devuelven si cancelo un paquete de 2.000 dólares 20 días antes?
Answer: Si cancelas un paquete de 2.000 dólares 20 días antes de la salida, recibes un reembolso de 1.400 dólares, es decir, el 70%.
Sources: ['politica_de_cancelacion.md']
Question: ¿Puedo llevar un power bank en la valija despachada?
Answer: No, las baterías de litio y los cargadores portátiles (power banks) deben viajar siempre en el equipaje de mano, nunca en la valija despachada.
Sources: ['equipaje_y_checkin.md']

=== Kind: trap ===
Question: ¿Qué vacunas necesito para viajar a Kenia?
Answer: No lo sé
Sources: []
Question: ¿Tienen paquetes a Tokio?
Answer: No lo sé
Sources: []
```

Los logs muestran, para cada pregunta, de qué archivos son los fragmentos que encontró la búsqueda.

### Uso desde código

```python
from rag.chain import get_rag_response

response = await get_rag_response("¿Qué plan de seguro cubre la cancelación por enfermedad?")
print(response.answer)    # Solo el plan Premium incluye la cancelación por enfermedad.
print(response.sources)   # ['seguro_de_viaje.md']
```

### Dataset

`data/` tiene cuatro documentos de una agencia de viajes ficticia, Rumbo Sur Viajes: política de cancelación, equipaje y check-in, seguro de viaje, y paquetes y destinos. Los datos son inventados, así que el modelo no los conoce de antemano: si responde bien, es porque los sacó de los documentos.

### Diseño

- **Ingesta** (`rag/ingest.py`):
  - `clean_text()` saca el encabezado que se repite en los documentos y los saltos de línea de más.
  - `split_documents()` usa `RecursiveCharacterTextSplitter` medido en tokens con `tiktoken`: fragmentos de hasta 500 tokens con 50 de solapamiento, cortando primero por párrafos. Los cuatro documentos quedan en 12 fragmentos.
  - Cada fragmento guarda el nombre de su archivo en los metadatos (`source`), que después se usa para citar las fuentes.
  - `index_documents()` usa IDs fijos (`archivo-número`) y no indexa si la base ya tiene datos.
- **Base vectorial** (`rag/vectorstore.py`): ChromaDB persistente en `vectorstore/`, con similitud coseno (por defecto Chroma usa distancia euclídea). La ingesta y las preguntas abren la base con la misma función, `get_vectorstore()`, así que usan siempre el mismo modelo de embeddings.
- **Embeddings** (`llm_client/langchain_models.py`): `build_embeddings()` crea `GoogleGenerativeAIEmbeddings` u `OpenAIEmbeddings` según el proveedor, con el modelo de `EMBEDDING_MODEL`.
- **Recuperación**: el retriever trae los 4 fragmentos más parecidos a la pregunta (`TOP_K = 4`).
- **Cadena** (`rag/chain.py`):

  ```python
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
  | prompt | model | PydanticOutputParser(RagResponse)
  ```

  `format_docs()` arma el contexto con el nombre del archivo de cada fragmento, y la cadena tiene `.with_retry()` de hasta 3 intentos si el JSON de la respuesta no es válido.
- **Prompt** (`rag/prompts.py`): el modelo responde solo con la información del contexto. Si la respuesta no está, responde exactamente "No lo sé", con la lista de fuentes vacía, aunque conozca la respuesta. La búsqueda siempre trae 4 fragmentos, aunque ninguno tenga la respuesta, así que la decisión de responder o no la toma el modelo.
- **Respuesta** (`rag/schemas.py`): `RagResponse` con `answer` y `sources` (los archivos que el modelo usó para responder).
- **Async**: `get_rag_response()` ejecuta la cadena con `ainvoke`. La búsqueda y la llamada al modelo son asíncronas.

## Estructura del proyecto

```
llm-client/
├── main.py                       # Unidad 1: script de prueba del cliente
├── run_pipeline.py               # Unidad 2: script de prueba del pipeline
├── run_ingest.py                 # Unidad 3: ingesta de los documentos en ChromaDB
├── run_rag.py                    # Unidad 3: preguntas al sistema RAG
├── requirements.txt
├── .env.example                  # Plantilla de variables de entorno (compartida)
├── data/                         # Unidad 3: documentos de la agencia de viajes ficticia
├── vectorstore/                  # Unidad 3: base ChromaDB (la crea run_ingest.py, no se sube a git)
├── llm_client/                   # Conexión con los proveedores
│   ├── config.py                 # load_config(): arma LLMConfig desde el .env
│   ├── schemas.py                # Modelos de Pydantic: ChatMessage, LLMConfig, ModelResponse, StreamChunk
│   ├── langchain_models.py       # build_chat_model() y build_embeddings(): modelos de LangChain según el proveedor
│   ├── manager.py                # AsyncLLMManager: crea el cliente del proveedor configurado
│   └── clients/
│       ├── base.py               # BaseLLMClient: clase base abstracta
│       ├── openai_client.py
│       ├── anthropic_client.py
│       └── gemini_client.py
└── pipeline/                     # Procesamiento de textos
    ├── schemas.py                # TechnicalEntities y CriticalityLevel
    ├── prompts.py                # EXTRACTION_PROMPT
    └── chain.py                  # Cadena LCEL, check_output(), process_text() y process_texts()
└── rag/                          # Sistema RAG
    ├── ingest.py                 # Limpieza, chunking e indexación
    ├── vectorstore.py            # get_vectorstore() y get_retriever()
    ├── schemas.py                # RagResponse
    ├── prompts.py                # RAG_PROMPT
    └── chain.py                  # Cadena RAG y get_rag_response()
```

## Notas

- Gemini lo probé con la API real. OpenAI y Anthropic los probé con API keys inválidas y respuestas simuladas, porque no tengo keys pagas de esos proveedores.
- Anthropic no recibe `temperature`, porque los modelos actuales de Claude ya no lo aceptan. Lo mismo pasa con `gemini-3.5-flash-lite`, el modelo alternativo del ejemplo: ignora `temperature` y avisa con un `UserWarning`.
- El modelo alternativo lo probé forzando un timeout de 0.01 segundos en el modelo principal: la cadena respondió con `gemini-3.5-flash-lite`.
- El sistema RAG lo probé con Gemini (`gemini-embedding-001` para los embeddings). Con OpenAI debería funcionar cambiando `LLM_PROVIDER`, `LLM_MODEL` y `EMBEDDING_MODEL`, pero no lo probé.
- La regla "al menos una tecnología" está en un `field_validator` y no en `Field(min_length=1)`. Con `min_length`, la regla viaja en el esquema que recibe el modelo, y Gemini la cumple a la fuerza: ante un texto sin tecnologías inventaba valores como "Desconocido". Con el validador, el modelo puede devolver la lista vacía y el rechazo lo hace el código.
