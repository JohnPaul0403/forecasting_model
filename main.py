import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lstm_engine import run_lstm_training

if __name__ == "__main__":
    run_lstm_training()
# This file serves as the entry point for running the LSTM training process.
# It imports the run_lstm_training function and executes it when the script is run directly.