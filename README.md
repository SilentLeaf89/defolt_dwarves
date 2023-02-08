Одно из заданий с Karpov Courses на финишной прямой.
Было предложено выполнять работу как тестовую, т.е. в ограниченный срок (не более 2-3 дней).
Текущая версия была подготовлена за 2 дня в рабочем режиме.

Пожалуйста, ознакомьтесь со структурой проекта ниже.

==============================

Начните с HW_6_baseline.ipynb в папке notebooks, если хотите взглянуть с самого начала и ознакомиться с легендой.
Начните с HW_6_lin_model.ipynb в папке notebooks, если интересует завершенное решение.
==============================

HW in interview block from Karpov Courses

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        └── models         <- Scripts to train models and then use trained models to make
            │                 predictions
            ├── predict_model.py
            └── train_model.py
        

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
