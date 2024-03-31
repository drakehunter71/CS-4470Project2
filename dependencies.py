import pandas as pd

def description(df):
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []
    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())
    # creating an output df
    output = pd.DataFrame(
        {
            "variable": variables,
            "dtype": dtypes,
            "count": count,
            "unique": unique,
            "missing value": missing,
        }
    )
    return output