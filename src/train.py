import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yaml
import pickle


def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    model_params = params["model"]

    # читаем подготовленные данные
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # отделяем признаки и целевую переменную
    X_train = train_df.drop(columns=["species"])
    y_train = train_df["species"]

    X_test = test_df.drop(columns=["species"])
    y_test = test_df["species"]

    # создаём и обучаем модель
    model = LogisticRegression(
        max_iter=model_params["max_iter"],
        C=model_params["C"],
    )
    model.fit(X_train, y_train)

    # считаем метрику
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # сохраняем модель
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Модель сохранена в model.pkl")


if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yaml
import pickle


def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    model_params = params["model"]

    # читаем подготовленные данные
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # отделяем признаки и целевую переменную
    X_train = train_df.drop(columns=["species"])
    y_train = train_df["species"]

    X_test = test_df.drop(columns=["species"])
    y_test = test_df["species"]

    # создаём и обучаем модель
    model = LogisticRegression(
        max_iter=model_params["max_iter"],
        C=model_params["C"],
    )
    model.fit(X_train, y_train)

    # считаем метрику
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # сохраняем модель
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Модель сохранена в model.pkl")


if __name__ == "__main__":
    main()
