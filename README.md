# Health and Lifestyle Prediction Model

## Overview
This project develops an industry-level machine learning pipeline to predict health status based on various lifestyle factors. The model leverages structured data preprocessing, machine learning techniques for efficient training and evaluation.

## Features
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling.
- **Decision Boundary Visualization**: Plots a classification boundary between healthy and unhealthy individuals.

## Dataset
The dataset contains the following features:
- **Numerical Features**: `phy_fitness`, `sleep_hrs`, `mindfulness`, `daily_avg_steps`, `daily_avg_calories`
- **Categorical Features**: `diet_pref`, `act_level`, `career`, `gender`
- **Target Variable**: `is_healthy` (Binary classification: 1 = Healthy, 0 = Unhealthy)


## Installation
### Prerequisites
- Python 3.8+
- Anaconda (Optional, for virtual environments)

### Install Dependencies
```
pip install -r requirements.txt
```

## Results
- The trained model achieves **high accuracy** and **f1 scores** for health prediction.
- Decision boundary plots provide interpretability on the classification outcomes.

## License
This project is licensed under the MIT License.

