# MLOps HW1 — классификация Iris с DVC и MLflow

## Цель проекта

Цель проекта — показать минимальный, но полноценный MLOps-контур на примере задачи классификации ирисов (датасет Iris):

- воспроизводимое обучение модели;
- версионирование данных и артефактов с помощью DVC;
- автоматический запуск пайплайна одной командой;
- логирование параметров, метрик и модели в MLflow Tracking UI.

Модель: логистическая регрессия (`LogisticRegression`) из `scikit-learn`.

---

## Структура проекта

```bash
.
├── .dvc/                 # служебные файлы DVC
├── data/                 # данные (под управлением DVC)
│   ├── raw/              # сырые данные
│   │   └── iris.csv
│   └── processed/        # подготовленные данные
│       ├── train.csv
│       └── test.csv
├── src/                  # исходный код
│   ├── prepare.py        # подготовка данных (split train/test)
│   └── train.py          # обучение модели + логирование в MLflow
├── dvc.yaml              # описание DVC-пайплайна (stages: prepare, train)
├── dvc.lock              # зафиксированные версии стадий и файлов
├── params.yaml           # гиперпараметры сплита и модели
├── requirements.txt      # зависимости проекта
└── README.md             # документация (этот файл)

## Как запустить

Все команды выполняются из корня репозитория.

```bash
python -m venv .venv            # создать виртуальное окружение (один раз)
.venv\Scripts\activate          # активировать окружение (Windows)
pip install -r requirements.txt # установить зависимости
dvc pull                        # скачать данные из DVC-хранилища
dvc repro                       # запустить весь ML-пайплайн (prepare + train)

После выполнения `dvc repro`:

- будут созданы/обновлены файлы `data/processed/train.csv` и `data/processed/test.csv`;
- обучится модель и сохранится в файл `model.pkl`;
- метрика `accuracy` и параметры модели будут залогированы в MLflow.

---

## Краткое описание пайплайна

Пайплайн описан в файле `dvc.yaml` и состоит из двух стадий.

### Стадия `prepare`

- **Скрипт:** `src/prepare.py`
- **Входы:**
  - `data/raw/iris.csv`
  - `params.yaml` (секция `split`)
- **Выходы:**
  - `data/processed/train.csv`
  - `data/processed/test.csv`
- **Действия:**
  - чтение сырых данных;
  - разбиение на train/test c фиксированным `random_state`;
  - сохранение подготовленных выборок в `data/processed/`.

### Стадия `train`

- **Скрипт:** `src/train.py`
- **Входы:**
  - `data/processed/train.csv`
  - `data/processed/test.csv`
  - `params.yaml` (секция `model`)
- **Выходы:**
  - файл модели `model.pkl`
  - записи об эксперименте в MLflow (`mlflow.db`, папка `mlruns/`)
- **Действия:**
  - обучение модели логистической регрессии;
  - вычисление точности (`accuracy`) на тестовой выборке;
  - логирование параметров (`model`, `C`, `max_iter`), метрики (`accuracy`) и артефакта (`model.pkl`) в MLflow.

---

## Где смотреть UI MLflow

1. Запустить MLflow Tracking UI (в активированном `.venv`):

   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   Открыть браузер и перейти по адресу:

- http://localhost:5000

В интерфейсе MLflow:

- выбрать эксперимент **`iris_experiment`**;
- внутри эксперимента смотреть:
  - вкладку **Parameters** — параметры модели (`model`, `C`, `max_iter`);
  - вкладку **Metrics** — метрику `accuracy`;
  - вкладку **Artifacts** — файл модели `model.pkl` и другие артефакты (если будут добавлены).