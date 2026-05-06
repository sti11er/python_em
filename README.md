# Online Topic Modeling with EM

Реализация online EM-алгоритмов для тематического моделирования с поддержкой регуляризации и локального E-step.

Проект содержит две модели:

* `OnlineEM` — стандартный online EM
* `OnlineEM_local_E` — версия с локальным распространением тематических вероятностей внутри документа

Модели обучаются на датасете **20 Newsgroups** и работают с BoW-представлением текста.  

---

## Возможности

* online обучение
* sparse / decorrelation регуляризация
* perplexity evaluation
* top words extraction
* визуализация `phi` и `theta`
* поддержка unigram / bigram признаков

---

## Установка

```bash id="9z6tr0"
pip install numpy scipy scikit-learn matplotlib seaborn
```

---

## Запуск

### Online EM

```bash id="f6fut9"
python online_em.py
```

### Local E-step версия

```bash id="a4lb34"
python online_em_local_e.py
```

---

## Основная идея

### E-step

p(t\mid d,w) \propto \phi_{wt}\theta_{td}

### Обновление `phi`

\phi_{wt} = \frac{n_{wt}}{\sum_w n_{wt}}

### Online update

\gamma_t=(\tau_0+t)^{-\kappa}

---

## Регуляризаторы

```python id="yd5w20"
model.add_regularizer(
    SparseRegularizer(tau=0.5, target="phi")
)

model.add_regularizer(
    DecorrelatorRegularizer(tau=0.05, target="phi")
)
```

---

## Пример

```python id="t2hm2d"
model = OnlineEM(
    num_topics=20,
    j_max=15,
    vocab_size=len(vocabulary),
    random_state=42
)

model.init_vocabulary(vocabulary)
model.fit_online(train_dict)

print(model.get_top_words())
```

---

## Метрика

Perplexity:

Perplexity = \exp\left(-\frac{\sum n_{dw}\log p(w\mid d)}{\sum n_{dw}}\right)

---

## Структура

```bash id="vb51tt"
.
├── online_em.py
├── online_em_local_e.py
└── README.md
```

---

## Реализации

* Online EM: 
* Local E-step EM: 
