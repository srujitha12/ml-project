!pip
install
numpy
pandas
scikit - learn
xgboost
deap
sentence - transformers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from deap import base, creator, tools, algorithms
import random
import warnings

warnings.filterwarnings("ignore")

# Load dataset
dataset_path = "/content/extended_5000_dataset.csv"  # Replace with your dataset path
data = pd.read_csv(dataset_path)

# Encode categorical features
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define target and features
y = data['TOTAL IPC CRIMES']
X = data.drop(columns=['TOTAL IPC CRIMES'])

# Feature correlation and initial cleaning
correlations = X.corr().abs()
upper_tri = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]
X = X.drop(columns=to_drop)

# Train-test split
y_binned = pd.qcut(y, q=4, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y_binned, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


# GA for Feature Selection
def evaluate_feature_selection(individual):
    selected_features = [col for i, col in enumerate(X.columns) if individual[i] == 1]
    if not selected_features:
        return -np.inf,
    model = Ridge(alpha=50.0)
    scores = cross_val_score(model, X_train_scaled[selected_features], y_train,
                             cv=10, scoring='r2')
    feature_penalty = -0.005 * sum(individual)
    return np.mean(scores) + feature_penalty,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_feature_selection)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=4)

population = toolbox.population(n=60)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

population, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.3,
                                      ngen=50, stats=stats, halloffame=hof, verbose=True)

best_features = [col for i, col in enumerate(X.columns) if hof[0][i] == 1]
X_train_selected = X_train_scaled[best_features]
X_test_selected = X_test_scaled[best_features]

print("Selected Features (GA):", best_features)

# Train models with adjusted hyperparameters
models = {
    "Ridge": Ridge(alpha=50.0),
    "Lasso": Lasso(alpha=25.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=5,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.5,
        random_state=42
    )
}

# Train all models
trained_models = {}
model_results = {}
for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred_train = model.predict(X_train_selected)
    y_pred_test = model.predict(X_test_selected)
    trained_models[name] = model
    model_results[name] = {
        "Train MAE": mean_absolute_error(y_train, y_pred_train),
        "Test MAE": mean_absolute_error(y_test, y_pred_test),
        "R^2": r2_score(y_test, y_pred_test)
    }

print("Model Accuracy Results:")
print(pd.DataFrame(model_results).T)


# Zero-Shot Learning Prediction
def predict_unseen_crime(unseen_crime):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    known_crimes = best_features  # Ensuring only GA-selected features are used
    known_embeddings = model.encode(known_crimes)
    unseen_embedding = model.encode([unseen_crime])[0]

    similarities = np.dot(known_embeddings, unseen_embedding) / (
                np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(unseen_embedding))
    top_matches = np.argsort(similarities)[-3:][::-1]
    valid_features = [known_crimes[i] for i in top_matches]

    if valid_features:
        # Create a DataFrame for the prediction with all selected features
        X_pred = pd.DataFrame(np.zeros((1, len(best_features))), columns=best_features)

        # Fill missing features with their mean value from the training set
        for feature in valid_features:
            if feature in X_train_selected.columns:
                X_pred[feature] = X_train_selected[feature].mean()

        # Ensure all columns from best_features are present in the DataFrame
        for feature in best_features:
            if feature not in valid_features:
                X_pred[feature] = X_train_selected[feature].mean()  # Fill missing features with the mean

        weighted_crime_rate = trained_models["RandomForest"].predict(X_pred)[0]
        return weighted_crime_rate
    return None


# Example ZSL prediction
unseen_crime = "Sedition"
predicted_rate = predict_unseen_crime(unseen_crime)
if predicted_rate is not None:
    print(f"Predicted crime rate for {unseen_crime}: {predicted_rate:.2f}")