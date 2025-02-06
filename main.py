import os

import numpy as np

from moftransformer.utils import prepare_data
from src.predict import predict
from src.utils import create_raw_pcond, create_pcond_features


# Regression task proton conductivity
def main():
    # create raw_pcond.json
    create_raw_pcond("./root_cifs")

    # Get example path
    root_cifs = "./root_cifs"
    root_dataset = "./root_dataset"
    downstream = "pcond"

    train_fraction = 0.0
    test_fraction = 1.0  # only for inference

    # Run prepare data
    prepare_data(
        root_cifs,
        root_dataset,
        downstream=downstream,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
    )
    # create json file with global features
    create_pcond_features(root_dataset="./root_dataset")
    y_pred = []
    # inference for one model
    for seed in range(1):
        for i in range(5):
            root_dataset = "./root_dataset"
            load_path = f"./weights/regr_fold_{i}_seed_{seed}.ckpt"
            save_dir = os.path.join(root_dataset, "results/regression")
            preds = predict(
                root_dataset=root_dataset,
                load_path=load_path,
                downstream="pcond",
                split=["test"],
                save_dir=save_dir,
            )
            # os.rename(
            # os.path.join(save_dir, "test_prediction.csv"),
            # os.path.join(save_dir, f"test_prediction_fold_{i}_seed_{seed}.csv"),
            # )
            # print(preds)
            y_pred.append(preds[0]["regression_logits"][0])
    return np.mean(y_pred)


if __name__ == "__main__":
    y_pred = main()
    print(y_pred)
