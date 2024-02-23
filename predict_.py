import click
import joblib
import pandas as pd


def predict():

    data = pd.read_csv("data/input.csv")

    artifacts = joblib.load("models/artifacts.joblib")

    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["encoder"]
    model = artifacts["model"]
    poly = artifacts["poly"]

    # Extract the used data
    data = data[num_features + fl_features + cat_features]

    # Apply imputer and encoder on data
    data[num_features] = imputer.transform(data[num_features])
    data_cat = enc.transform(data[cat_features]).toarray()

    # Generate polynomial features
    data_poly = poly.transform(data[num_features])

    # Convert to DataFrame with correct feature names
    data_poly = pd.DataFrame(data_poly, columns=poly.get_feature_names_out())

    # Combine the numerical and one-hot encoded categorical columns
    data = pd.concat(
        [
            data_poly.reset_index(drop=True),
            data[fl_features].reset_index(drop=True),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Make predictions
    predictions = model.predict(data)
   # predictions = predictions[:10]  # just picking 10 to display sample output :-)

    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input!)
    pd.DataFrame({"predictions": predictions}).to_csv(
        "output\predictions.csv", index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to output\predictions.cs")
    click.echo(
        f"Nbr. observations: {data.shape[0]} | Nbr. predictions: {predictions.shape[0]}"
    )
    ### -------------------------------------------------- ###


if __name__ == "__main__":
    # how to run on command line:
    # python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
    predict()