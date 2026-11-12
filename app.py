import sys
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from sklearn.cluster import AgglomerativeClustering

pillow_heif.register_heif_opener()

MODEL_PROTO = r'SqueezeNet\SqueezeNet_v1.1\deploy.prototxt'
MODEL_WEIGHTS = r'SqueezeNet\SqueezeNet_v1.1\squeezenet_v1.1.caffemodel'

net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)


@cache
def load_image_to_qimage(file_path):
    """Decode an image in a worker thread and return an owned QImage."""
    rgb_array = load_image_to_rgb(file_path)
    if rgb_array is None:
        return
    height, width, channels = rgb_array.shape
    bytes_per_line = channels * width
    return QtGui.QImage(
        rgb_array.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888
    ).copy()


@cache
def load_image_to_rgb(file_path):
    if file_path.lower().endswith('.heic'):
        pil_img = Image.open(file_path).convert('RGB')
        return np.array(pil_img)

    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        bgr = cv2.imread(file_path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@cache
def embed_image(file_path):
    rgb_array = load_image_to_rgb(file_path)
    if rgb_array is None:
        return
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(
        bgr_array,
        scalefactor=1.0 / 255.0,
        size=(224, 224),
        mean=(104.0, 117.0, 123.0),  # mean of ImageNet dataset (according to Google)
        swapRB=True,  # squeezenet is BGR not RGB
        crop=False,
    )
    net.setInput(blob)
    pool10 = net.forward('pool10')
    embedding_vector = np.mean(pool10, axis=(2, 3)).flatten()
    return embedding_vector / np.linalg.norm(embedding_vector)


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


class ImageEmbedderSignals(QtCore.QObject):
    image_embedded = QtCore.Signal(str, np.ndarray)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal()


class ImageEmbedder(QtCore.QRunnable):
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
        self.signals = ImageEmbedderSignals()

    @QtCore.Slot()
    def run(self):
        total = len(self.file_paths)

        try:
            for completed, file_path in enumerate(self.file_paths):
                try:
                    if load_image_to_qimage(file_path) is not None:
                        embedding = embed_image(file_path)
                        self.signals.image_embedded.emit(file_path, embedding)
                except Exception as error:
                    print(f'Could not load {file_path}: {error}')
                finally:
                    self.signals.progress.emit(completed, total)
        finally:
            self.signals.finished.emit()


class ImageClustererSignals(QtCore.QObject):
    finished = QtCore.Signal(AgglomerativeClustering)


class ImageClusterer(QtCore.QRunnable):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.embedding_matrix = embedding_matrix
        self.signals = ImageClustererSignals()

    @QtCore.Slot()
    def run(self):
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(
            self.embedding_matrix
        )
        self.signals.finished.emit(model)


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
        self._cluster_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)

        self._cluster_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._cluster_button.clicked.connect(self.embed_images)
        self._cluster_slider.setRange(0, 0)
        self._cluster_slider.setEnabled(False)
        self._cluster_slider.sliderReleased.connect(self.group_images)

        layout.addWidget(self._cluster_button, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._cluster_slider)

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
        else:
            self._clustering_menu.setVisible(True)
            self._cluster_button.setEnabled(True)
            self._cluster_slider.setRange(0, 0)
            self._cluster_slider.setEnabled(False)
        self._image_loader = None
        self._browse_button.setEnabled(True)
        self._extract_button.setEnabled(True)
        self._progress.setValue(0)

    def embed_images(self):
        self._embedding_file_paths = []
        self._embeddings = []
        self._cluster_button.setEnabled(False)
        self._image_embedder = ImageEmbedder(self._image_files)
        self._image_embedder.signals.image_embedded.connect(self.add_embedding)
        self._image_embedder.signals.progress.connect(self.update_progress)
        self._image_embedder.signals.finished.connect(self.finish_embedding_images)
        self._thread_pool.start(self._image_embedder)

    @QtCore.Slot(str, np.ndarray)
    def add_embedding(self, image_path, embedding):
        self._embedding_file_paths.append(image_path)
        self._embeddings.append(embedding)

    @QtCore.Slot()
    def finish_embedding_images(self):
        embedding_matrix = np.array(self._embeddings)
        self._image_embedder = None
        self._image_clusterer = ImageClusterer(embedding_matrix)
        self._image_clusterer.signals.finished.connect(self.finish_clustering_images)
        self._thread_pool.start(self._image_clusterer)

    @QtCore.Slot(AgglomerativeClustering)
    def finish_clustering_images(self, model):
        self._image_clusterer = None
        self._model = model
        self._progress.setValue(100)
        self._cluster_slider.setRange(0, 1000)
        self._cluster_slider.setValue(0)
        self._cluster_slider.setEnabled(True)

    def group_images(self):
        distance_threshold = self._cluster_slider.value() / 1000
        n_samples = self._model.n_leaves_
        remainders = list(range(n_samples))
        children = [
            self._model.children_[index]
            for index, distance in enumerate(self._model.distances_)
            if distance < distance_threshold
        ]
        explored_indexes = []
        groups = []

        def get_group_leaves(i):
            if i >= n_samples:
                index = i - n_samples
                nodes = children[index]
                explored_indexes.append(index)
                return get_group_leaves(nodes[0]) + get_group_leaves(nodes[1])
            else:
                return [int(i)]

        for index in range(len(children) - 1, -1, -1):
            if index in explored_indexes:
                continue
            nodes = children[index]
            explored_indexes.append(index)
            group = get_group_leaves(nodes[0]) + get_group_leaves(nodes[1])
            remainders = [remainder for remainder in remainders if remainder not in group]
            groups.append(group)

        groups += [[remainder] for remainder in remainders]


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.showMaximized()
    widget.show()

    sys.exit(app.exec())
