import pandas as pd
from catboost import Pool


def preprocessing_catboost_train(df: pd.DataFrame):
    """
    Принимает:
    - датафрейм с 24 колонками, без NaN. Подробнее в HW_6_processed.ipynb
    - долю разбития на тестовые данные в train_test_split от 0 до 1

    Возвращает:
    для train:
    - кортеж из двух Pool catboost-а:
    - X_test, y_test
    для test:
    - один Pool catboost-а.
    """

    assert df.shape[1] == 24, "Некорректное количество столбцов, должно быть 24"
    assert df.isna().sum().sum() == 0, "Датафрейм некорректен, содержит NaN значения"

    X = df.drop(columns=["Default"])
    y = df["Default"]

    cols = X.columns

    X = X[cols].astype(
        {
            "Secret_dwarf_info_1": "float64",
            "Secret_dwarf_info_2": "float64",
            "Secret_dwarf_info_3": "float64",
            "Successful_deals_count": "float64",
            "Tavern": "int",
            "Hashed_deal_detail_1": "float64",
            "Hashed_deal_detail_2": "int",
            "Hashed_deal_detail_3": "int",
            "Hashed_deal_detail_4": "float64",
            "Hashed_deal_detail_5": "int",
            "Age": "int",
            "Gender": "int",
            "Tavern_district_1": "int",
            "Tavern_district_2": "int",
            "Tavern_district_3": "int",
            "Tavern_district_4": "int",
            "Tavern_district_5": "int",
            "Tavern_district_6": "int",
            "Tavern_district_7": "int",
            "day_before_first_defolt": "int",
            "Deal_day": "int",
            "Deal_month": "int",
            "is_weekend": "int",
        }
    )

    cat_cols = [
        4,
        6,
        7,
        9,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        20,
        21,
        22,
    ]

    train_pool = Pool(X, label=y, cat_features=cat_cols)
    
    return (train_pool, y)


if __name__ == "__main__":
    preprocessing_catboost_train()
