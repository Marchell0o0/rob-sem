# Robot Control and Camera Calibration

This project provides tools for robot control and camera calibration.

## Usage

1. Camera Calibration:

```bash
# Capture calibration images
python -m calibration.capture_calibration_images --images-count 30

# Run calibration
python -m calibration.calibrate_with_chessboard --board-width 7 --board-height 5 --square-size 30 --marker-size 22
```

2. Robot Control:

```bash
# Run main application
python main.py --robot-type RV6S
```
