# portfolio-investing

Репозиторий для построения инвестиционных портфелей с использованием современных количественных методов: кластеризации активов по корреляционной дистанции, ребалансировки по принципу минимальной межгрупповой корреляции, walk-forward бэктеста с учётом издержек.

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
pip install -e .          # установить пакет в режиме разработки
```

### 2. Запуск эксперимента

```bash
# Запуск с конфигом по умолчанию
python scripts/run_experiment.py

# Или через console script (после pip install -e .)
run-experiment

# Опции CLI
run-experiment --help
run-experiment --config config/default_config.yaml \
               --output output \
               --start-date 2021-01-01 \
               --end-date 2024-12-31 \
               --n-clusters 4

# Только проверить конфиг без запуска
run-experiment --dry-run
```

Результат сохраняется в `output/strategy_report.csv`.

### 3. Запуск тестов

```bash
python -m pytest tests/ -v
```

---

## Архитектура

```
portfolio-investing/
├── config/
│   └── default_config.yaml      # Параметры: тикеры, даты, пороги, комиссии
├── scripts/
│   └── run_experiment.py        # CLI-скрипт запуска
├── src/
│   └── portfolio_investing/
│       ├── cli.py               # Click CLI, оркестрация эксперимента
│       ├── data/
│       │   └── loader.py        # Загрузка котировок (yfinance), очистка
│       ├── risk/
│       │   └── correlation.py   # Log-доходности, shrinkage, rolling-корреляции, d_ij
│       ├── clustering/
│       │   └── cluster.py       # Иерархическая кластеризация, silhouette, стабильность
│       ├── allocation/
│       │   └── weights.py       # Equal Weight / Inverse Vol / Risk Parity по кластерам
│       ├── rebalance/
│       │   └── rebalancer.py    # Календарный / пороговый / гибридный ребаланс
│       ├── backtest/
│       │   └── engine.py        # Walk-forward бэктест с комиссиями и проскальзыванием
│       └── reporting/
│           └── metrics.py       # CAGR, Sharpe, Sortino, Max Drawdown, turnover, CSV
└── tests/
    ├── test_correlation.py
    ├── test_clustering.py
    ├── test_allocation.py
    ├── test_rebalance.py
    └── test_smoke.py
```

---

## Конфигурация (`config/default_config.yaml`)

| Параметр | Описание |
|---|---|
| `tickers.*` | Тикеры по группам: российские акции, альтернативы, крипто, кэш |
| `backtest.start_date / end_date` | Период бэктеста |
| `correlation.rolling_window` | Rolling-окно корреляции (торговых дней, по умолчанию 126) |
| `correlation.shrinkage` | Использовать Ledoit-Wolf shrinkage (рекомендуется) |
| `clustering.n_clusters_min/max` | Диапазон числа кластеров (3–5) |
| `clustering.linkage` | Метод иерархической кластеризации (`ward`, `average`) |
| `allocation.methods` | Список методов аллокации для сравнения |
| `allocation.max_weight_per_asset` | Лимит веса на один актив (напр. 0.20) |
| `allocation.max_weight_per_cluster` | Лимит веса на кластер (напр. 0.50) |
| `rebalance.modes` | Режимы ребаланса: `monthly`, `quarterly`, `threshold`, `hybrid_monthly`, `hybrid_quarterly` |
| `rebalance.threshold_pct` | Порог отклонения для threshold-ребаланса (напр. 0.05 = 5%) |
| `costs.commission_pct` | Комиссия за сделку (напр. 0.001 = 0.1%) |
| `costs.slippage_pct` | Проскальзывание (напр. 0.001 = 0.1%) |
| `constraints.max_turnover_per_period` | Лимит оборачиваемости за период |
| `reporting.risk_free_rate` | Безрисковая ставка (напр. 0.10 = 10% для российского рынка) |

---

## Тикеры портфеля

| Тикер (yfinance) | Актив | Примечания |
|---|---|---|
| `SBERP.ME` | Сбербанк России (ап) | |
| `ZAYM.ME` | Займер | |
| `SIBN.ME` | Газпром нефть | |
| `BELU.ME` | НоваБев Групп | |
| `TATNP.ME` | Татнефть ап | |
| `PLZL.ME` | Полюс | |
| `DOMRF.ME` | ДОМ.РФ | |
| `PHOR.ME` | ФосАгро | |
| `MOEX.ME` | Московская Биржа | |
| `MDMG.ME` | Мать и дитя | |
| `LKOH.ME` | НК ЛУКОЙЛ | |
| `PARUS-LOG.ME` | ПАРУС-ЛОГИСТИКА | Может быть недоступен — пропускается с предупреждением |
| `RENTAL-PRO.ME` | ЗПИФ Рентал ПРО | Может быть недоступен — пропускается с предупреждением |
| `BTC-USD` | Bitcoin | |
| `TAO-USD` | Bittensor | |
| `ETH-USD` | Ethereum | |
| `USDT-USD` | Tether | Cash-like компонент |

**Недоступные тикеры** (PARUS-LOG.ME, RENTAL-PRO.ME) автоматически пропускаются: система логирует предупреждение и продолжает работу без них.

---

## Метрики стратегий

| Метрика | Описание |
|---|---|
| `cagr` | Compound Annual Growth Rate |
| `sharpe` | Коэффициент Шарпа (аннуализированный) |
| `sortino` | Коэффициент Сортино |
| `max_drawdown` | Максимальная просадка (отрицательное значение) |
| `avg_turnover` | Средняя оборачиваемость за период ребалансировки |
| `n_rebalances` | Число ребалансировок за период |

---

## Интерпретация результатов

Пример выходной таблицы `output/strategy_report.csv`:

```
allocation    rebalance        cagr   sharpe  sortino  max_drawdown  avg_turnover  n_rebalances
equal_weight  monthly         0.142    0.85     1.12        -0.28          0.18            48
equal_weight  quarterly       0.138    0.83     1.09        -0.29          0.14            16
equal_weight  threshold       0.145    0.87     1.15        -0.26          0.16            23
inverse_vol   monthly         0.151    0.91     1.20        -0.24          0.19            48
...
```

**Как выбрать стратегию-победителя:**
1. Максимальный Sharpe при допустимом Max Drawdown.
2. Проверить avg_turnover: высокий оборот при низких комиссиях — нормально.
3. Сравнить net-результат (CAGR уже учитывает издержки).

---

## Методология

### Корреляционная дистанция
```
d_ij = sqrt(0.5 * (1 - rho_ij))
```
где `rho_ij` — коэффициент корреляции Пирсона между доходностями активов i и j.

### Кластеризация
Иерархическая кластеризация (Ward linkage) по матрице дистанций.
Число кластеров выбирается по максимальному silhouette score в диапазоне [min_k, max_k].

### Walk-forward бэктест
1. Первые `min_periods` дней: инициализация кластеров и весов.
2. На каждый торговый день: дрейф весов по фактическим доходностям.
3. При триггере ребаланса: пересчёт кластеров/весов по всей доступной истории → применение ограничений → учёт комиссий и проскальзывания.

---

## MCP-серверы для AI-ассистента

Для работы ИИ как полноценного Quant-разработчика можно подключить MCP-серверы:

```bash
cp .mcp.json.example .mcp.json
```

Подробные инструкции: [docs/mcp-setup.md](docs/mcp-setup.md)
