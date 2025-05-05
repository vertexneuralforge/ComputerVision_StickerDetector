@echo off
python -m venv .venv
call .venv\Scripts\activate.bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python
echo CPU environment ready!
pause