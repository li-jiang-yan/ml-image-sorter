# ML Image Sorter
An application that sorts images in a given directory using machine learning

## Application icon

The application icon is in `assets/app_icon.ico` (Windows) and `assets/app_icon.png`.
The Qt window uses the ICO file at runtime. When building a Windows executable with
PyInstaller, pass the same file as the executable icon and bundle it for the window:

```powershell
pyinstaller --noconfirm --clean --onefile --windowed --name ImageSorter --icon=assets/app_icon.ico --add-data "assets/app_icon.ico;assets" --add-data "SqueezeNet;SqueezeNet" app.py
```
