import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="neuron.log"  # ファイル出力
)

pred_cells = [1, 2, 3]
firing_cells = [2, 3, 4]
logging.info(f"predicted_cells: {pred_cells}")
logging.info(f"firing_cells: {firing_cells}")

import csv

timestep = 'time=1'
with open("log.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([timestep, pred_cells, firing_cells])