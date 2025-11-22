import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    split_params = params["split"]

    # читаем сырые данные
    df = pd.read_csv("data/raw/iris.csv")

    # делим на train/test
    train_df, test_df = train_test_split(
        df,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=df["species"],  # чтобы классы были равномерно
    )

    # создаём папку для обработанных данных
    os.makedirs("data/processed", exist_ok=True)

    # сохраняем
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Данные подготовлены:")
    print("train:", train_df.shape, "test:", test_df.shape)


if __name__ == "__main__":
    main()
