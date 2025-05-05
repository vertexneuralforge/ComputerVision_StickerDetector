@echo off
python -m venv .venv
call .venv\Scripts\activate.bat
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python
echo GPU environment ready!
pause