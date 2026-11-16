# VAST Challenge 2017 MC1 · хранилище данных

PostgreSQL 17 хранит сырой журнал, очищенные события и витрины. [Исходный CSV](./data/sensors.csv) нужен для первоначального наполнения Bronze. [Описание набора](./data/description.md), [брошюра](./data/brochure.md) и [карта](./data/map.bmp) лежат рядом с ним.

| Слой | Таблица | Содержимое |
| --- | --- | --- |
| Bronze | `bronze.events` | Четыре текстовые колонки CSV и служебные поля для связи с исходным файлом. |
| Bronze | `bronze.csv_imports` | Точная копия начального CSV и его SHA-256. |
| Silver | `silver.events` | Проверенные события, порядок посещения датчиков и интервалы между ними. |
| Silver | `silver.rejected_events` | Отклонённые записи, причины и исходные поля. |
| Gold | `gold.trips` | Одна запись с признаками маршрута на `car_id` в каждом снимке. |
| Gold | `gold.hourly_traffic` | События и уникальные `car_id` по дате, часу, датчику и типу транспорта. |

## Первоначальная загрузка

```powershell
Copy-Item .env.example .env
docker compose up -d --wait
python -m venv .venv
.\.venv\bin\python.exe -m pip install -r requirements.txt
$env:DATABASE_URL = 'postgresql://vast:vast_local@127.0.0.1:5432/vast'
.\.venv\bin\python.exe load_datalake.py seed
.\.venv\bin\python.exe load_datalake.py process
```

При изменении пароля или порта в `.env` обновите `DATABASE_URL`. Команда `seed` записывает архив CSV и сырые записи в одной транзакции. Повторный `seed` с тем же CSV пропускает импорт.

## Добавление данных после загрузки

Приложение или оператор добавляет новые события напрямую в `bronze.events`. Поля сохраняются как текст и проверяются при построении Silver.

```sql
INSERT INTO bronze.events ("Timestamp", "car-id", "car-type", "gate-name")
VALUES ('2016-06-01 08:15:00', 'example-001', '1', 'entrance0');
```

В VS Code откройте таблицу `bronze.events` или выполните `SELECT * FROM bronze.events LIMIT 20`. Если база создана предыдущей версией, запустите `process` один раз для переноса уже сохранённых значений в отдельные колонки.

После записи запускайте обработку.

```powershell
.\.venv\bin\python.exe load_datalake.py process
```

`process` читает накопленные события из PostgreSQL. Для каждого набора Bronze он строит новый снимок Silver и Gold одной транзакцией. Маршруты и проверка дубликатов охватывают все накопленные события. Повторный запуск с тем же содержимым Bronze переиспользует снимок, а предыдущие снимки доступны для воспроизводимых выборок.

Silver сохраняет время без часового пояса. При равных отметках времени порядок событий определяется `bronze.events.event_id`. В снимках v3 колонка `source_line` ссылается на этот идентификатор. Для начального CSV номер строки в файле доступен в `bronze.events.source_file_line`.

Счётчики и контрольные суммы доступны в `public.vast_snapshots`. Для выборки признаков используйте идентификатор, который выводит команда `process`.

```sql
SELECT car_id, entry_hour, entry_dow, n_events, n_unique_gates, duration_h
FROM gold.trips
WHERE snapshot_id = '<snapshot_id>'
ORDER BY car_id;
```

Данные PostgreSQL сохраняются в именованном томе Docker. Для резервной копии всех слоёв и снимков используйте `pg_dump`.
