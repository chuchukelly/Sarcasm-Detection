import argparse
import pandas as pd
import joblib
import os


def load_model(model_path="./models/model_weights.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model


def run_inference(input_path, output_path):
    # 1. 读入测试 csv
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    # 2. 取出文本（保证是字符串）
    texts = df["text"].astype(str)

    # 3. 加载训练好的 TF-IDF+SVM pipeline
    model = load_model("./models/model_weights.pkl")

    # 4. 直接 predict（因为 pipeline 里已经包含了 vectorizer）
    preds = model.predict(texts)

    # 5. 确保是 0/1
    # LinearSVC 本来就输出 0/1，如果你 label 是别的，记得在训练时就编码成 0/1
    df_out = pd.DataFrame({
        "text": texts,
        "prediction": preds.astype(int)
    })

    # 6. 保存结果
    df_out.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarcasm prediction script")

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV with a 'text' column")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save output CSV with 'text' and 'prediction' columns")

    args = parser.parse_args()

    run_inference(args.input, args.output)
