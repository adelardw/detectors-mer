# MER — Multimodal Deepfake Detection

[English](#english) | [Русский](#русский)

---

## English

MER is a research repository for **multimodal deepfake detection**. The model combines:

- **FAU-based frame-level facial features**
- **rPPG-based video-level physiological features**
- **VideoMAE** as the main video backbone
- **LoRA adaptation** for efficient fine-tuning
- **Attention pooling + MLP head** for final binary classification (**REAL / FAKE**)

The project is currently in an **early research stage**. The core model code is available, while some infrastructure parts such as full preprocessing, dataset-specific pipelines, and weight distribution are still being cleaned up.

## Architecture

![MER architecture](docs/architecture.png)

### High-level idea

The model processes a video through two complementary branches:

1. **Frame-level branch** extracts facial action information from individual frames.
2. **Video-level branch** extracts rPPG features that capture physiological consistency over time.

Both branches are projected into a shared representation space, enriched with special tokens and positional encoding, and then fused through a **VideoMAE backbone**. The final representation is aggregated by an **attention pooler** and passed to an **MLP classifier**.

This design is motivated by the idea that deepfakes may look visually plausible frame by frame, but often struggle to preserve:

- realistic facial dynamics,
- temporal consistency,
- physiological patterns such as pulse-related variation.

## Repository structure

```text
src/
  backbones/      # external and custom backbones
  models/         # model definitions
  config/         # training configs (to be finalized)
  ...
```

### Main directories

- `src/models` — model implementations
- `src/config` — training configurations
- `src/backbones` — model backbones and external dependencies
- preprocessing — planned as a separate module and still under cleanup

## Setup

### 1. Environment

Run:

```bash
bash env.sh
```

This script prepares the environment and installs the required dependencies.

### 2. FAU weights

FAU weights must be downloaded separately.

At the moment this is not fully automated. A proper storage location and download script are planned.

### 3. rPPG weights

Weights for the rPPG branch come from the original **rPPG-Toolbox** repository and are expected under:

```text
src/backbones/rPPGToolbox/final_model_release
```

## Training

Training is implemented in **PyTorch Lightning**.

Current assumptions:

- the exact training entrypoint depends on the dataset format,
- preprocessing is partly project-specific,
- configs are being organized and documented.

In practice, the training logic is built around Lightning modules and can be adapted once the dataset interface is fixed.

## Current status

What is already available:

- core model code
- multimodal architecture
- training logic in Lightning style
- validation metrics and experimental runs

What is still being cleaned up:

- preprocessing module
- config documentation
- automatic download of external weights
- dataset-specific training instructions
- reproducible end-to-end setup

## Notes

- This repository is intended primarily as a **research codebase**, not yet as a polished production package.
- Some components depend on external model weights and internal dataset conventions.
- The codebase is being gradually refactored into a more reproducible public version.

## Citation

If you use this repository, please cite the project page or contact the author directly until a formal paper or technical report is released.

---

## Русский

MER — это исследовательский репозиторий для задачи **детекции дипфейков на основе мультимодальных признаков**. Модель объединяет:

- **FAU-признаки на уровне кадров**
- **rPPG-признаки на уровне видео**
- **VideoMAE** в качестве основного видеобэкбона
- **LoRA-адаптацию** для эффективного дообучения
- **attention pooling + MLP head** для итоговой бинарной классификации (**REAL / FAKE**)

Проект находится на **ранней исследовательской стадии**. Основной код модели уже есть, но часть инфраструктуры — например, полный препроцессинг, пайплайны под конкретные датасеты и распространение весов — ещё приводится в порядок.

## Архитектура

![Архитектура MER](docs/architecture.png)

### Общая идея

Модель обрабатывает видео через две взаимодополняющие ветки:

1. **Ветка уровня кадров** извлекает признаки лицевой динамики из отдельных кадров.
2. **Ветка уровня видео** извлекает rPPG-признаки, отражающие физиологическую согласованность сигнала во времени.

Далее обе ветки проецируются в общее пространство представлений, обогащаются специальными токенами и позиционным кодированием, после чего объединяются через **VideoMAE backbone**. Итоговое представление агрегируется с помощью **attention pooler** и передаётся в **MLP-классификатор**.

Идея архитектуры основана на том, что дипфейки могут выглядеть правдоподобно на уровне отдельных кадров, но часто хуже сохраняют:

- реалистичную лицевую динамику,
- временную согласованность,
- физиологические паттерны, например вариации, связанные с пульсом.

## Структура репозитория

```text
src/
  backbones/      # внешние и кастомные бэкбоны
  models/         # определения моделей
  config/         # конфиги обучения (будут дооформлены)
  ...
```

### Основные директории

- `src/models` — реализации моделей
- `src/config` — конфигурации обучения
- `src/backbones` — бэкбоны и внешние зависимости
- preprocessing — планируется как отдельный модуль и пока ещё дорабатывается

## Установка

### 1. Окружение

Запуск:

```bash
bash env.sh
```

Этот скрипт подготавливает окружение и устанавливает зависимости.

### 2. Веса FAU

Веса для FAU необходимо скачать отдельно.

Сейчас этот процесс ещё не полностью автоматизирован. В дальнейшем для них стоит добавить нормальное хранилище и отдельный скрипт скачивания.

### 3. Веса rPPG

Веса для rPPG-ветки берутся из исходного репозитория **rPPG-Toolbox** и ожидаются по пути:

```text
src/backbones/rPPGToolbox/final_model_release
```

## Обучение

Обучение реализовано на **PyTorch Lightning**.

Текущие допущения:

- точка входа в обучение зависит от формата датасета,
- препроцессинг частично завязан на конкретные данные,

На практике логика обучения уже построена вокруг Lightning-модулей и может быть адаптирована после фиксации интерфейса данных.

## Текущее состояние

Что уже есть:

- основной код модели
- мультимодальная архитектура
- логика обучения в стиле Lightning
- валидационные метрики и экспериментальные прогоны

Что ещё нужно дооформить:

- модуль препроцессинга
- документацию по конфигам
- автоматическую загрузку внешних весов
- инструкции по обучению под конкретные датасеты
- воспроизводимый end-to-end setup

## Примечания

- Репозиторий в первую очередь задуман как **исследовательский код**, а не как полностью отполированный production-пакет.
- Некоторые компоненты зависят от внешних весов моделей и внутренних соглашений по датасетам.
- Кодовая база постепенно приводится к более воспроизводимой публичной версии.

## Цитирование

Если вы используете этот репозиторий, пожалуйста, ссылайтесь на страницу проекта или свяжитесь с автором напрямую, пока не будет опубликована статья или технический отчёт.
