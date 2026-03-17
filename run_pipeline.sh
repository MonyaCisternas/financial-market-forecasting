#!/bin/bash

LOG_FILE = "logs/pipeline.log"

echo "====================================" >> $LOG_FILE
echo "Run started at $(date)" >> $LOG_FILE

python src/data_pipeline.py >> $LOG_FILE 2>&1
python src/feature_engineering.py >> $LOG_FILE 2>&1
python src/statistical_models.py >> $LOG_FILE 2>&1
python src/regime_detection.py >> $LOG_FILE 2>&1
python src/model_training.py >> $LOG_FILE 2>&1

echo "Run finished as $(date)" >> $LOG_FILE
