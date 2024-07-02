# Earthquake Prediction System

This project aims to predict earthquake magnitudes and depths using machine learning techniques. The system retrieves earthquake data from the USGS Earthquake Hazards Program, processes the data, trains a predictive model, and visualizes the predictions on a world map.

## Flow of work
1. [Project Overview](#project-overview)
2. [Data Retrieval](#data-retrieval)
3. [Feature Selection](#feature-selection)
4. [Timestamp Conversion](#timestamp-conversion)
5. [Feature Scaling](#feature-scaling)
6. [Model Training](#model-training)
7. [Performance Metrics](#performance-metrics)
8. [Earthquake Prediction](#earthquake-prediction)
9. [Visualization](#visualization)
10. [How to Run](#how-to-run)
11. [Dependencies](#dependencies)
12. [License](#license)

## Project Overview

This project leverages machine learning to predict earthquake occurrences based on historical data. The main components include data retrieval, preprocessing, model training, and visualization. The predictions help in identifying regions prone to seismic activities, thereby enhancing disaster preparedness.

## Data Retrieval

The earthquake data, spanning from 2000 to 2024, is retrieved from the USGS Earthquake Hazards Program using the specified URL. The data is loaded into a Pandas DataFrame for structured and accessible analysis.

```python
import pandas as pd

url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv'
data = pd.read_csv(url)
```

## Feature Selection

Relevant features such as time, latitude, longitude, depth, and magnitude are selected from the dataset. These features are crucial for training the predictive model.

```python
features = ['time', 'latitude', 'longitude', 'depth', 'mag']
data = data[features]
```

## Timestamp Conversion

The timestamp feature is converted to Unix format for numerical analysis. Error handling ensures robustness in the conversion process.

```python
from datetime import datetime
import time

def convert_to_unix(timestamp):
    try:
        dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
        unix_timestamp = time.mktime(dt.timetuple())
        return unix_timestamp
    except Exception as e:
        print(f"Error converting timestamp: {timestamp}. Error: {e}")
        return float('nan')

data['unix_time'] = data['time'].apply(convert_to_unix)
```

## Feature Scaling

Feature scaling is performed using the Standard Scaler from scikit-learn to maintain consistency across features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['unix_time', 'latitude', 'longitude']])
```

## Model Training

A Gradient Boosting Regressor is used for predicting earthquake magnitudes and depths. Hyperparameter tuning is conducted using GridSearchCV.

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

X = scaled_data
y_mag = data['mag']
y_depth = data['depth']

X_train, X_test, y_train_mag, y_test_mag = train_test_split(X, y_mag, test_size=0.2, random_state=42)
X_train_depth, X_test_depth, y_train_depth, y_test_depth = train_test_split(X, y_depth, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5]
}
grid_search_mag = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5)
grid_search_mag.fit(X_train, y_train_mag)

best_model_mag = grid_search_mag.best_estimator_
best_model_mag.fit(X_train, y_train_mag)
predictions_mag = best_model_mag.predict(X_test)
```

## Performance Metrics

Various performance metrics such as Mean Squared Error (MSE), Accuracy, Precision, Recall, F1 Score, and Confusion Matrix are calculated to evaluate the model's accuracy.

```python
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

mse_mag = mean_squared_error(y_test_mag, predictions_mag)
# Other metrics can be similarly calculated
```

## Earthquake Prediction

The trained model generates predictions for new data, highlighting regions with a higher likelihood of earthquakes.

```python
new_data = scaler.transform(new_data)
predictions = best_model_mag.predict(new_data)
```

## Visualization

The predictions are visualized on a world map using GeoPandas and Matplotlib.

```python
import geopandas as gpd
import matplotlib.pyplot as plt

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(15, 10), color='lightgray', edgecolor='black')

# Plot predictions
gdf = gpd.GeoDataFrame(new_data, geometry=gpd.points_from_xy(new_data.longitude, new_data.latitude))
gdf.plot(ax=ax, color='blue', markersize=predictions ** 2, alpha=0.7, label='Predicted Magnitude')

plt.title('Predicted Earthquake Data on World Map')
plt.legend()
plt.show()
```

## How to Run

1. Clone the repository
    ```bash
    git clone https://github.com/yourusername/earthquake-prediction.git
    cd earthquake-prediction
    ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Run the main script
    ```bash
    python main.py
    ```

## Dependencies

- Pandas
- NumPy
- Scikit-learn
- GeoPandas
- Matplotlib
- mplcursors

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn geopandas matplotlib mplcursors
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive overview of your earthquake prediction project, ensuring that anyone who reads it can understand and use your code effectively.
