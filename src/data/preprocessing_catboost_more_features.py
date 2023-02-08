import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import Pool


def preprocessing_catboost_more_features(df: pd.DataFrame, train: bool, test_size=0.25):
    """
    Принимает:
    - датафрейм с дополнительными колонками, без NaN. Подробнее в HW_6_processed_feature_gen.ipynb
    - долю разбития на тестовые данные в train_test_split от 0 до 1

    Возвращает:
    для train:
    - кортеж из двух Pool catboost-а:
    - X_test, y_test
    для test:
    - один Pool catboost-а.
    """

    if train:
        assert df.isna().sum().sum() == 0, "Датафрейм некорректен, содержит NaN значения"
    
    if train:
        X = df.drop(columns=["Default"])
        y = df["Default"]
    else:
        X = df

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
            "Secret_12": "float64",
            "Secret_23": "float64",
            "Secret_31": "float64",
            "Secret_11": "float64",
            "Secret_22": "float64",
            "Secret_33": "float64"

        }
    )

    if train:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=2023
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

    if train:
        train_pool = Pool(X_train, label=y_train, cat_features=cat_cols)
        test_pool = Pool(X_test, label=y_test, cat_features=cat_cols)
    else:
        test_pool = Pool(X, cat_features=cat_cols)
    
    return (train_pool, test_pool, X_test, y_test) if train else test_pool


if __name__ == "__main__":
    preprocessing_catboost_more_features()
