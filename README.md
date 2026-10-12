# llm-client

Cliente asíncrono de LLMs con una interfaz común para OpenAI, Anthropic y Gemini. Soporta respuestas normales y en streaming, valida la configuración con Pydantic y devuelve errores controlados en lugar de cortar el programa.

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

Copiar `.env.example` a `.env` y completar la API key del proveedor que se va a usar.

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
| `GOOGLE_API_KEY` | API key de Gemini |
| `OPENAI_API_KEY` | API key de OpenAI |
| `ANTHROPIC_API_KEY` | API key de Anthropic |

Solo es obligatoria la key del proveedor elegido. El archivo `.env` está en el `.gitignore`.

## Uso

```bash
python main.py
```

El script le pregunta "¿Qué es la entropía?" al proveedor configurado y corre tres pruebas:

1. **Modo normal**: espera la respuesta completa y la muestra.
2. **Modo streaming**: muestra la respuesta de a pedazos, a medida que llega.
3. **Prueba de resiliencia**: hace la misma pregunta con una API key inválida y muestra el error que devuelve el cliente. El programa sigue funcionando después del error.

Si la configuración está mal (proveedor no soportado, falta el modelo o la API key está vacía), el script muestra un error de configuración y termina.

Para cambiar de proveedor, alcanza con cambiar `LLM_PROVIDER` y `LLM_MODEL` en el `.env`. No hace falta tocar el código.

## Estructura del proyecto

```
llm-client/
├── main.py                     # Script de prueba: carga el .env y corre las tres pruebas
├── requirements.txt
├── .env.example                # Plantilla de variables de entorno
└── llm_client/
    ├── schemas.py              # Modelos de Pydantic: ChatMessage, LLMConfig, ModelResponse, StreamChunk
    ├── manager.py              # AsyncLLMManager: crea el cliente del proveedor configurado
    └── clients/
        ├── base.py             # BaseLLMClient: clase base abstracta
        ├── openai_client.py
        ├── anthropic_client.py
        └── gemini_client.py
```

## Diseño

- **Interfaz común**: todos los clientes heredan de `BaseLLMClient` e implementan `generate()` y `generate_stream()`.
- **Factory**: `AsyncLLMManager` elige la clase del cliente con un diccionario (`CLIENTS`) según `config.provider`. Para agregar un proveedor hay que crear su cliente y sumar una entrada al diccionario.
- **Validación**: `LLMConfig` valida el proveedor, el modelo, la temperatura (de 0 a 2), `max_tokens` (mayor que 0), el timeout y la API key, que se guarda como `SecretStr` para que no aparezca en los logs. Todos los clientes reciben este mismo objeto de configuración.
- **Asincronía**: todas las llamadas usan la versión asíncrona de cada SDK (`AsyncOpenAI`, `AsyncAnthropic` y `client.aio` en Gemini).
- **Streaming**: `generate_stream()` es un generador asíncrono que devuelve objetos `StreamChunk`. Cada chunk tiene `content` o `error`, así un error en el medio del stream se distingue del texto.
- **Manejo de errores**: cada cliente traduce los errores de su propio SDK (rate limit, conexión, errores del servidor, pedidos inválidos) a un `ModelResponse` o `StreamChunk` con el campo `error` completo. Todas las llamadas tienen un timeout (`asyncio.timeout`, configurable en `LLMConfig`). Los reintentos los hacen los SDKs: OpenAI y Anthropic reintentan dos veces por defecto, y el cliente de Gemini está configurado con 3 intentos.
- **Diferencias entre proveedores**: Gemini y Anthropic reciben el mensaje de sistema como un parámetro aparte, y Gemini usa el rol `model` en lugar de `assistant`. Cada cliente convierte los mensajes al formato de su proveedor.

## Notas

- Gemini lo probé con la API real. OpenAI y Anthropic los probé con API keys inválidas y respuestas simuladas, porque no tengo keys pagas de esos proveedores.
- El cliente de Anthropic no envía `temperature`, porque los modelos actuales de Claude ya no lo aceptan.
