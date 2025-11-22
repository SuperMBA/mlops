import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yaml
import pickle
import mlflow


def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # загружаем параметры
    params = load_params()
    model_params = params["model"]

    # настраиваем MLflow
    # будет создан файл mlflow.db в папке проекта
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_experiment")

    # читаем подготовленные данные
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop(columns=["species"])
    y_train = train_df["species"]

    X_test = test_df.drop(columns=["species"])
    y_test = test_df["species"]

    # запускаем запись эксперимента
    with mlflow.start_run():
        # создаём и обучаем модель
        model = LogisticRegression(
            max_iter=model_params["max_iter"],
            C=model_params["C"],
        )
        model.fit(X_train, y_train)

        # метрика
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # логируем параметры и метрику в MLflow
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", model_params["C"])
        mlflow.log_param("max_iter", model_params["max_iter"])

        mlflow.log_metric("accuracy", acc)

        # сохраняем модель локально
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        print("Модель сохранена в model.pkl")

        # логируем модель как артефакт
        mlflow.log_artifact("model.pkl")


if __name__ == "__main__":
    main()
