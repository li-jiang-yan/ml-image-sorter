import sys
from pathlib import Path

import pillow_heif
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

pillow_heif.register_heif_opener()


def load_image_to_qimage(file_path):
    """Decode an image in a worker thread and return an owned QImage."""
    if file_path.lower().endswith('.heic'):
        image = Image.open(file_path).convert('RGBA')
        data = image.tobytes('raw', 'RGBA')
        return QtGui.QImage(
            data,
            image.width,
            image.height,
            QtGui.QImage.Format_RGBA8888,
        ).copy()

    image = QtGui.QImage(file_path)
    return image if not image.isNull() else None


class ImageLoaderSignals(QtCore.QObject):
    image_loaded = QtCore.Signal(str, QtGui.QImage)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal()


class ImageLoader(QtCore.QRunnable):
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
        self.signals = ImageLoaderSignals()

    @QtCore.Slot()
    def run(self):
        total = len(self.file_paths)

        try:
            for completed, file_path in enumerate(self.file_paths):
                try:
                    image = load_image_to_qimage(file_path)
                    if image is not None:
                        self.signals.image_loaded.emit(file_path, image)
                except Exception as error:
                    print(f'Could not load {file_path}: {error}')
                finally:
                    self.signals.progress.emit(completed, total)
        finally:
            self.signals.finished.emit()


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
        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._image_loader = None
        self._loaded_image_count = 0

        layout.addWidget(QtWidgets.QLabel('Folder'))
        layout.addWidget(self._folder_path)
        layout.addWidget(self._browse_button)
        layout.addWidget(self._extract_button)

        self._folder_upload_box.setLayout(layout)

    def create_image_gallery(self):
        self.create_clustering_menu()
        self._clustering_menu.setVisible(False)

        self._image_gallery = QtWidgets.QGroupBox('Image gallery')
        layout = QtWidgets.QVBoxLayout()

        self._progress = QtWidgets.QProgressBar(self)
        self._progress.setValue(0)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)

        gallery_content = QtWidgets.QWidget()
        self._gallery_grid = QtWidgets.QGridLayout()

        gallery_content.setLayout(self._gallery_grid)
        scroll_area.setWidget(gallery_content)
        layout.addWidget(self._progress)
        layout.addWidget(self._clustering_menu)
        layout.addWidget(scroll_area)
        self._image_gallery.setLayout(layout)

    def create_clustering_menu(self):
        self._clustering_menu = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        _, top, _, bottom = layout.getContentsMargins()
        layout.setContentsMargins(0, top, 0, bottom)

        self._cluster_button = QtWidgets.QPushButton('Cluster Images')

        self._cluster_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )

        layout.addWidget(self._cluster_button, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        self._clustering_menu.setLayout(layout)

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

        self._loaded_image_count = 0
        self._browse_button.setEnabled(False)
        self._extract_button.setEnabled(False)
        self._clustering_menu.setVisible(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._image_loader = ImageLoader(self._image_files)
        self._image_loader.signals.image_loaded.connect(self.add_image_to_gallery)
        self._image_loader.signals.progress.connect(self.update_progress)
        self._image_loader.signals.finished.connect(self.finish_loading_images)
        self._thread_pool.start(self._image_loader)

    @QtCore.Slot(str, QtGui.QImage)
    def add_image_to_gallery(self, image_path, image):
        # QPixmap creation and widget updates stay on the GUI thread.
        pixmap = QtGui.QPixmap.fromImage(image)
        if pixmap.isNull():
            return

        thumbnail = QtWidgets.QLabel()
        thumbnail.setPixmap(
            pixmap.scaled(
                QtCore.QSize(160, 120),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )
        thumbnail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        thumbnail.setToolTip(image_path)

        columns = 4
        row = self._loaded_image_count // columns
        column = self._loaded_image_count % columns
        self._gallery_grid.addWidget(thumbnail, row, column)
        self._loaded_image_count += 1

    @QtCore.Slot(int, int)
    def update_progress(self, completed, total):
        self._progress.setValue(completed / total * 100)

    @QtCore.Slot()
    def finish_loading_images(self):
        if self._loaded_image_count == 0:
            self._gallery_grid.addWidget(QtWidgets.QLabel('No images found.'), 0, 0)
        self._image_loader = None
        self._browse_button.setEnabled(True)
        self._extract_button.setEnabled(True)
        self._progress.setVisible(False)
        self._clustering_menu.setVisible(True)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.showMaximized()
    widget.show()

    sys.exit(app.exec())
