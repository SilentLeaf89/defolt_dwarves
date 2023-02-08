import pandas as pd


def clean_data(path: str, secret: bool):
    """
    Создание очищенного датасета из исходного csv.
    Подробное описание основных преобразований приведено в HW_6_baseline.ipynb
    Предназначен для:
        ../data/raw/train.csv
        ../data/raw/test.csv
    
    Принимает:
    - path: путь к csv файлу
    - secret: True - обрабатывать фичи Secret_dwarf_info_n, False - не обрабатывать
    Возвращает:
    - pandas DataFrame.
    """
    dwarves = pd.read_csv(path)
    dwarves["Successful_deals_count"] = dwarves["Successful_deals_count"].fillna(0)
    dwarves["Region"] = dwarves["Region"].fillna("Tavern_district_3")
    dwarves["First_default_date"] = dwarves["First_default_date"].fillna(
        dwarves["Deal_date"]
    )
    dwarves = dwarves.join(pd.get_dummies(dwarves["Region"], drop_first=True))
    dwarves["Gender"] = dwarves["Gender"].apply(lambda x: 0 if x == "Female" else 1)
    dwarves["First_default_date"] = pd.to_datetime(dwarves["First_default_date"])
    dwarves["First_deal_date"] = pd.to_datetime(dwarves["First_deal_date"])
    dwarves["Deal_date"] = pd.to_datetime(dwarves["Deal_date"])
    dwarves["day_before_first_defolt"] = (
        dwarves["First_default_date"] - dwarves["First_deal_date"]
    ).dt.days
    dwarves["Deal_day"] = dwarves["Deal_date"].dt.day
    dwarves["Deal_month"] = dwarves["Deal_date"].dt.month
    dwarves["is_weekend"] = dwarves["Deal_date"].dt.dayofweek.apply(
        lambda x: 1 if x > 4 else 0
    )

    if secret:
        dwarves["Secret_dwarf_info_1"] = dwarves["Secret_dwarf_info_1"].fillna(
            dwarves["Secret_dwarf_info_1"].median()
        )
        dwarves["Secret_dwarf_info_2"] = dwarves["Secret_dwarf_info_2"].fillna(
            dwarves["Secret_dwarf_info_2"].median()
        )
        dwarves["Secret_dwarf_info_3"] = dwarves["Secret_dwarf_info_3"].fillna(
            dwarves["Secret_dwarf_info_3"].median()
        )

    dwarves = dwarves.drop(
        columns=[
            "Deal_id",
            "Region",
            "Deal_date",
            "First_deal_date",
            "First_default_date",
            "Hashed_deal_detail_6",
        ]
    )

    return dwarves


if __name__ == "__main__":
    clean_data()
