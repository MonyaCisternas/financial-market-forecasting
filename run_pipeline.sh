#!/bin/bash

echo "Starting weekly retraining pipeline..."

python src/data_pipeline.py
python src/feature_engineering.py
python src/statistical_models.py
python src/regime_detection.py
python src/model_training.py

echo "Pipeline completed successfully."
