import pandas as pd

base_file_string = "Data/speculative/Speculation-Dataset/"
nspec_test_file_list = [
    "tok/nspec_test.tok",
    "stm/nspec_test.stm",
    "bgm/nspec_test.bgm",
]
nspec_train_file_list = [
    "tok/nspec_train.tok",
    "stm/nspec_train.stm",
    "bgm/nspec_train.bgm",
]
spec_test_file_list = ["tok/spec_test.tok", "stm/spec_test.stm", "bgm/spec_test.bgm"]
spec_train_file_list = [
    "tok/spec_train.tok",
    "stm/spec_train.stm",
    "bgm/spec_train.bgm",
]

dfs = []
file_path_list = (
    nspec_test_file_list
    + nspec_train_file_list
    + spec_test_file_list
    + spec_train_file_list
)
for file_path in file_path_list:
    with open(base_file_string + file_path, "r") as file:
        data = file.readlines()
        print(file_path)
        print(len(data))
        print("")
        df = pd.DataFrame(data, columns=["Data"])

        if "tok" in file_path:
            df["type"] = "tok"
        elif "stm" in file_path:
            df["type"] = "stm"
        else:
            df["type"] = "bgm"

        if "train" in file_path:
            df["traintest"] = "train"
        else:
            df["traintest"] = "test"

        if "nspec" in file_path:
            df["speculative"] = "No"
        else:
            df["speculative"] = "Yes"
        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv("Data/combined.csv")
