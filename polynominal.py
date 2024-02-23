import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


def train(data):
    """Trains a linear regression model on the full dataset and stores output."""

    # Load the data
    data = data

    # Define features to use
    num_features = [
        "nbr_frontages",
        "nbr_bedrooms",
        "latitude",
        "longitude",
        "total_area_sqm",
        "surface_land_sqm",
        "terrace_sqm",
        "garden_sqm",
        "construction_year",
        "primary_energy_consumption_sqm",
        "cadastral_income"
    ]

    fl_features = [
        "fl_furnished",
        "fl_open_fire",
        "fl_terrace",
        "fl_garden",
        "fl_swimming_pool",
        "fl_floodzone",
        "fl_double_glazing"
    ]

    cat_features = [
        "property_type",
        "subproperty_type",
        "region",
        "province",
        "locality",
        "equipped_kitchen",
        "state_building",
        "epc",
        "heating_type"
    ]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

    ''' POLYNOMIAL TEST'''
    poly_features = PolynomialFeatures(degree=3, include_bias=False)

    X_train_poly = poly_features.fit_transform(X_train[num_features])
    X_test_poly = poly_features.transform(X_test[num_features])

    X_train_poly = pd.concat(
        [
            pd.DataFrame(
                X_train_poly, columns=poly_features.get_feature_names_out(num_features)),
            X_train[fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out())
        ],
        axis=1,
    )

    X_test_poly = pd.concat(
        [
            pd.DataFrame(
                X_test_poly, columns=poly_features.get_feature_names_out(num_features)),
            X_test[fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out())
        ],
        axis=1,
    )

    something = enc.get_feature_names_out()
    print(something)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Evaluate the model
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        'poly': poly_features,  # O transformador de características polinomiais
        'model': model,         # O modelo de regressão linear
        'imputer': imputer,     # O imputador para tratar valores ausentes
        'enc': enc,         # O codificador one-hot
        'features': {           # Um dicionário das listas de características
            'num_features': num_features,
            'fl_features': fl_features,
            'cat_features': cat_features,
        }
    }
    joblib.dump(artifacts, "models/artifacts.joblib")
    print(artifacts.keys())

    return y_train, y_train_pred, y_test, y_test_pred
