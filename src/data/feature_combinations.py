import pandas as pd


def feature_combinations(df: pd.DataFrame):
    """
    Принимает:
    - датафрейм с 24 колонками, без NaN. Подробнее в HW_6_processed.ipynb

    Возвращает:
    - модифицированный датасет по ссылке
    """

    df['Secret_12'] = df['Secret_dwarf_info_1'] * df['Secret_dwarf_info_2']
    df['Secret_23'] = df['Secret_dwarf_info_2'] * df['Secret_dwarf_info_3']
    df['Secret_31'] = df['Secret_dwarf_info_3'] * df['Secret_dwarf_info_1']

    df['Secret_11'] = df['Secret_dwarf_info_1'] * df['Secret_dwarf_info_1']
    df['Secret_22'] = df['Secret_dwarf_info_2'] * df['Secret_dwarf_info_2']
    df['Secret_33'] = df['Secret_dwarf_info_3'] * df['Secret_dwarf_info_3']
    
    return


if __name__ == "__main__":
    feature_combinations()
