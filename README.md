# ASHRAE - Great Energy Predictor III

## Overview

This project aims to predict the hourly energy consumption of buildings using historical meter readings, weather data, and building characteristics. The competition dataset consists of over 20 million records, and the goal is to develop an accurate machine learning model to forecast meter readings for electricity, chilled water, steam, and hot water meters.

## Problem Statement

Buildings account for approximately 40% of global energy use and 30% of carbon emissions. Accurate energy consumption predictions can support energy efficiency, cost savings, and sustainability. This project leverages machine learning to enhance forecasting accuracy, benefiting energy planning, policy-making, and building operations.

## Dataset

The dataset, sourced from the ASHRAE Great Energy Predictor III competition, includes:

- **Building Metadata**: Building ID, primary use, square footage, year built, etc.
- **Weather Data**: Temperature, humidity, wind speed, cloud coverage, and more.
- **Meter Readings**: Hourly energy usage for each building.
- **Time Features**: Timestamp, hour, weekday, month, and season.

The dataset is partitioned into train and test sets, with a time-based split to prevent data leakage.

## Exploratory Data Analysis (EDA)

Key insights from the dataset:

- Buildings have varied energy consumption patterns depending on use type.
- Certain features, like square footage and temperature, influence energy use.
- Missing values in `year_built` and `floor_count` were imputed or dropped.
- Log transformation was applied to `meter_reading` to normalize its distribution.

## Feature Engineering

To improve model performance, several features were engineered:

- **Heating Degree Hours (HDH) & Cooling Degree Hours (CDH)**: Derived metrics to capture temperature impact.
- **Wind Chill Effect**: Accounts for wind's effect on perceived temperature.
- **Temporal Features**: Hour, weekday, season, and peak hour indicator.
- **Building Age**: Computed from `year_built` for a more interpretable feature.

## Machine Learning Models

The following models were trained and evaluated:

- **Baseline Models**:
  - Linear Regression
  - k-Nearest Neighbors (kNN)

- **Tree-Based Models**:
  - Decision Tree
  - Random Forest
  - Gradient Boosting Machine (GBM)
  - XGBoost
  - LightGBM

### Model Performance

| Model            | RMSLE Score |
|-----------------|------------|
| kNN             | 0.4902     |
| Random Forest   | **0.2750** |
| Gradient Boosting | 0.5165     |
| XGBoost        | 0.3993     |
| LightGBM       | 0.4469     |

Random Forest achieved the best performance with an RMSLE of **0.2750**.

© 2025 Meltem Sahin Ozkoc – Carnegie Mellon University

