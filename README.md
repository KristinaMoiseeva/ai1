# Лабораторная работа 1 — вариант 4

Источник задания: `lab_1-2/assignment1.ipynb`, ветка `main`.

**Вариант 4:** задания 1 и 3 на MNIST: kNN + softmax.

## Состав

- `assignment1.ipynb` — выполненный ноутбук по пунктам 1.1–1.11 и 3.1–3.10.
- `scripts/classifiers/k_nearest_neighbor.py` — собственный kNN.
- `scripts/classifiers/softmax.py` — naive/vectorized softmax loss и градиент.
- `scripts/classifiers/linear_classifier.py` — SGD и `Softmax`.
- `data/mnist_subset.npz` — фиксированные train/validation/test подвыборки MNIST.
- `outputs/mnist/` — фактические результаты, графики и сохранённые модели.
- `REPORT.md` — краткая сводка результатов.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook assignment1.ipynb
```

В ноутбуке `RECOMPUTE=False` использует уже рассчитанные результаты. Для полного повторного обучения установите `RECOMPUTE=True`.

## Данные

Используется настоящий MNIST 28×28: 5000 train, 1000 validation и 1000 test. Test выбран из официальной тестовой части и не используется для подбора параметров.
