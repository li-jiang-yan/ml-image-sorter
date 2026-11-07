import sys
from pathlib import Path

import pillow_heif
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

pillow_heif.register_heif_opener()


def load_heic_to_pixmap(file_path):
    # Open the HEIC file using Pillow
    image = Image.open(file_path)
    # Ensure it's converted to a mode PyQt handles cleanly (e.g., RGBA)
    image = image.convert('RGBA')

    # Convert PIL Image to QImage
    data = image.tobytes('raw', 'RGBA')
    q_image = QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format_RGBA8888)

    return QtGui.QPixmap.fromImage(q_image)


def load_pixmap(file_path):
    if file_path.lower().endswith('.heic'):
        pixmap = load_heic_to_pixmap(file_path)
        return pixmap

    pixmap = QtGui.QPixmap(file_path)
    return pixmap


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.create_folder_upload_box()
        self.create_image_gallery()

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self._folder_upload_box)
        main_layout.addWidget(self._image_gallery)
        self.setLayout(main_layout)

    def create_folder_upload_box(self):
        self._folder_upload_box = QtWidgets.QGroupBox('Folder upload')
        layout = QtWidgets.QHBoxLayout()

        self._folder_path = QtWidgets.QLineEdit(readOnly=True)
        self._browse_button = QtWidgets.QPushButton('Browse')
        self._extract_button = QtWidgets.QPushButton('Extract Images')

        self._browse_button.clicked.connect(self.open_folder_dialog)
        self._extract_button.clicked.connect(self.extract_images)
        self._extract_button.setEnabled(False)

        layout.addWidget(QtWidgets.QLabel('Folder'))
        layout.addWidget(self._folder_path)
        layout.addWidget(self._browse_button)
        layout.addWidget(self._extract_button)

        self._folder_upload_box.setLayout(layout)

    def create_image_gallery(self):
        self._image_gallery = QtWidgets.QGroupBox('Image gallery')
        layout = QtWidgets.QVBoxLayout()

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)

        gallery_content = QtWidgets.QWidget()
        self._gallery_grid = QtWidgets.QGridLayout()

        gallery_content.setLayout(self._gallery_grid)
        scroll_area.setWidget(gallery_content)
        layout.addWidget(scroll_area)
        self._image_gallery.setLayout(layout)

    def open_folder_dialog(self):
        self._dir = QtWidgets.QFileDialog.getExistingDirectory()

        if self._dir:
            self._folder_path.setText(self._dir)
            self._extract_button.setEnabled(True)

    def extract_images(self):
        dir_path = Path(self._dir)
        self._image_files = [str(entry) for entry in dir_path.rglob('*') if entry.is_file()]
        self.display_images()

    def display_images(self):
        while self._gallery_grid.count():
            item = self._gallery_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._image_files:
            self._gallery_grid.addWidget(
                QtWidgets.QLabel('No images found.'),
                0,
                0,
            )
            return

        columns = 4
        thumbnail_size = QtCore.QSize(160, 120)

        all_pixmaps = [
            {'path': image_path, 'pixmap': load_pixmap(image_path)}
            for image_path in self._image_files
        ]
        pixmaps = [pixmap for pixmap in all_pixmaps if not pixmap['pixmap'].isNull()]
        for index, pixmap in enumerate(pixmaps):
            image_path = pixmap['path']
            pixmap = pixmap['pixmap']

            thumbnail = QtWidgets.QLabel()
            thumbnail.setPixmap(
                pixmap.scaled(
                    thumbnail_size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
            thumbnail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            thumbnail.setToolTip(image_path)

            row = index // columns
            column = index % columns
            self._gallery_grid.addWidget(thumbnail, row, column)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.showMaximized()
    widget.show()

    sys.exit(app.exec())
