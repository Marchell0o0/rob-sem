# Robot Control and Camera Calibration

This project provides tools for robot control and camera calibration.

## Usage

1. Camera Calibration:

```bash
# Capture calibration images
python -m calibration.capture_calibration_images

# Run calibration
python -m calibration.calibrate_with_chessboard --board-width 9 --board-height 6 --square-size 5
```

2. Robot Control:

```bash
# Run main application
python main.py --robot-type RV6S
```
