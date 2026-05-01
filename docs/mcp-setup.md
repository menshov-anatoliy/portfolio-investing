# Настройка MCP-серверов для Quant-разработки

Для того чтобы ИИ-ассистент работал как полноценный Quant-разработчик, необходимо подключить четыре MCP-сервера. Ниже приведены инструкции по установке и настройке каждого из них.

---

## Предварительные требования

| Инструмент | Минимальная версия | Установка |
|---|---|---|
| Node.js | 18+ | https://nodejs.org |
| Python | 3.10+ | https://python.org |
| `uv` (пакетный менеджер Python) | 0.4+ | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Jupyter Notebook | 7+ | `pip install notebook` |

---

## 1. `mcp-server-filesystem` — Файловая система

**Зачем:** ИИ может самостоятельно создавать структуру проекта на вашем диске — сохранять скрипты (`hrp_model.py`), скачивать исторические данные в `.csv` и сохранять сгенерированные графики бэктеста в папку `output/`.

**Установка:** пакет устанавливается автоматически при первом запуске через `npx`.

**Настройка в `.mcp.json`:**

```json
"filesystem": {
  "command": "npx",
  "args": [
    "-y",
    "@modelcontextprotocol/server-filesystem",
    "<path-to-repo>"
  ]
}
```

Замените `<path-to-repo>` на абсолютный путь к корню этого репозитория на вашем компьютере.

**Пример структуры, которую ИИ создаст автоматически:**

```
portfolio-investing/
├── data/               # Исторические данные (.csv)
├── models/             # Модели (hrp_model.py, mvo_model.py)
├── notebooks/          # Jupyter Notebook для анализа
└── output/             # Графики бэктеста и отчёты
```

---

## 2. `mcp-server-fetch` — HTTP/REST клиент

**Зачем:** Позволяет ИИ загружать актуальную документацию по библиотекам (`PyPortfolioOpt`, `Riskfolio-Lib`) и выполнять прямые API-запросы к финансовым источникам данных.

**Установка:**

```bash
uv tool install mcp-server-fetch
```

**Настройка в `.mcp.json`:**

```json
"fetch": {
  "command": "uvx",
  "args": ["mcp-server-fetch"]
}
```

**Что умеет:**
- Загрузка документации (`https://pyportfolioopt.readthedocs.io/`)
- Запросы к REST API (например, Alpha Vantage, Polygon.io)
- Чтение JSON-ответов от финансовых провайдеров

---

## 3. `mcp-server-brave-search` — Поиск в интернете

**Зачем:** Алгоритм HRP (Маркоса Лопеса де Прадо) имеет множество современных модификаций. Поиск позволит ИИ находить свежие научные статьи на SSRN и актуальные тикеры ETF для разных классов активов.

**Получение API-ключа:**

1. Зарегистрируйтесь на https://brave.com/search/api/
2. Создайте новый API-ключ (доступен бесплатный тариф с 2 000 запросов/месяц)
3. Скопируйте ключ

**Установка:** пакет устанавливается автоматически через `npx`.

**Настройка в `.mcp.json`:**

```json
"brave-search": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-brave-search"],
  "env": {
    "BRAVE_API_KEY": "ВАШ_КЛЮЧ_ЗДЕСЬ"
  }
}
```

**Пример использования ИИ:**
> «Найди последние статьи на SSRN о модификациях алгоритма Hierarchical Risk Parity за 2024 год»
> «Найди ETF на облигации развивающихся рынков с активами более $1 млрд»

---

## 4. `mcp-server-jupyter` — Исполнение Python-кода

**Зачем:** Критически важен. Позволяет ИИ исполнять Python-код прямо в Jupyter Notebook: строить матрицы корреляций, рисовать дендрограммы (деревья кластеров HRP) и сразу видеть ошибки, чтобы исправлять их на лету.

**Установка:**

```bash
uv tool install mcp-server-jupyter
pip install notebook
```

**Запуск Jupyter с токеном:**

```bash
jupyter notebook --NotebookApp.token=my_secret_token --NotebookApp.port=8888
```

**Настройка в `.mcp.json`:**

```json
"jupyter": {
  "command": "uvx",
  "args": ["mcp-server-jupyter"],
  "env": {
    "JUPYTER_BASE_URL": "http://localhost:8888",
    "JUPYTER_TOKEN": "my_secret_token"
  }
}
```

**Что умеет:**
- Запускать ячейки с кодом на Python
- Устанавливать пакеты (`pip install pyportfolioopt riskfolio-lib`)
- Строить графики (matplotlib, seaborn, plotly)
- Читать и записывать переменные между ячейками

---

## Подключение к AI-клиенту

Шаг 1 — скопируйте шаблон конфигурации и укажите свои пути и ключи:

```bash
cp .mcp.json.example .mcp.json
# Откройте .mcp.json и замените <path-to-repo>, <your-brave-api-key>, <your-jupyter-token>
```

> ⚠️ Файл `.mcp.json` добавлен в `.gitignore` — ваши ключи не попадут в репозиторий.

### Claude Desktop

Скопируйте содержимое вашего `.mcp.json` в конфигурационный файл Claude Desktop:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

### Cursor

Скопируйте `.mcp.json` в `.cursor/mcp.json` в корне вашего проекта.

### Windsurf / Codeium

Добавьте серверы через раздел **Settings → MCP Servers** в интерфейсе приложения.

---

## Проверка работоспособности

После настройки попросите ИИ выполнить тестовые команды:

```
1. Создай файл data/test.csv с тестовыми данными о ценах акций
2. Загрузи документацию PyPortfolioOpt с https://pyportfolioopt.readthedocs.io/en/latest/
3. Найди топ-5 ETF на золото по объёму активов
4. Запусти Python-код: import numpy as np; print(np.version.version)
```

---

## Полезные ссылки

| Ресурс | Ссылка |
|---|---|
| MCP официальная документация | https://modelcontextprotocol.io |
| `server-filesystem` на npm | https://www.npmjs.com/package/@modelcontextprotocol/server-filesystem |
| `server-fetch` на PyPI | https://pypi.org/project/mcp-server-fetch/ |
| `server-brave-search` на npm | https://www.npmjs.com/package/@modelcontextprotocol/server-brave-search |
| Brave Search API | https://brave.com/search/api/ |
| PyPortfolioOpt | https://pyportfolioopt.readthedocs.io/ |
| Riskfolio-Lib | https://riskfolio-lib.readthedocs.io/ |
| HRP — оригинальная статья Лопеса де Прадо | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678 |
