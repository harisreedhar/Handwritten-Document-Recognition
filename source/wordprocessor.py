# original text editor by Martin Fitzpatrick
# tutorial: https://www.mfitzp.com/pyqt-examples/python-rich-text-editor/
# source: https://www.mfitzp.com/d/wordprocessor.zip

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *

import os
import sys
import uuid

from spellchecker import SpellChecker
spell = SpellChecker()

THICKNESS = [0, 1, 2, 3, -1, -2, -3]
RESOLUTIONS = [1200, 1000, 800, 600, 2200, 2000, 1800, 1600]
DEBUG_IMAGES = ['Source', 'Blurred', 'Edge', 'Blob', 'Bounding Boxes']
FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 24, 36, 48, 64, 72, 96, 144, 288]
IMAGE_EXTENSIONS = ['.jpg','.png','.bmp']
HTML_EXTENSIONS = ['.htm', '.html']

def hexuuid():
    return uuid.uuid4().hex

def splitext(p):
    return os.path.splitext(p)[1].lower()

# https://stackoverflow.com/questions/24106903/resizing-qpixmap-while-maintaining-aspect-ratio
class ImageLabel(QLabel):
    def __init__(self, img):
        super(ImageLabel, self).__init__()
        if img is None:
            img = './images/no_preview.png'
        self.setFrameStyle(QFrame.StyledPanel)
        self.pixmap = QPixmap(img)

    def paintEvent(self, event):
        size = self.size()
        painter = QPainter(self)
        point = QPoint(0,0)
        scaledPix = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation)
        point.setX((size.width() - scaledPix.width())/2)
        point.setY((size.height() - scaledPix.height())/2)
        painter.drawPixmap(point, scaledPix)

    def changePixmap(self, img):
        self.pixmap = QPixmap(img)
        self.repaint()

    def changePixmapWithArray(self, img):
        img = np.asarray(img)
        img = img.astype('uint8')
        w,h,ch = img.shape
        if img.ndim == 1:
            img =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        qimg = QImage(img.data, h, w, 3*h, QImage.Format_RGB888)
        qpixmap = QPixmap(qimg)
        self.pixmap = QPixmap(qimg)
        self.repaint()

class TextEdit(QTextEdit):
    def canInsertFromMimeData(self, source):
        if source.hasImage():
            return True
        else:
            return super(TextEdit, self).canInsertFromMimeData(source)
    def insertFromMimeData(self, source):
        cursor = self.textCursor()
        document = self.document()
        if source.hasUrls():
            for u in source.urls():
                file_ext = splitext(str(u.toLocalFile()))
                if u.isLocalFile() and file_ext in IMAGE_EXTENSIONS:
                    image = QImage(u.toLocalFile())
                    document.addResource(QTextDocument.ImageResource, u, image)
                    cursor.insertImage(u.toLocalFile())
                else:
                    # If we hit a non-image or non-local URL break the loop and fall out
                    # to the super call & let Qt handle it
                    break
            else:
                # If all were valid images, finish here.
                return
        elif source.hasImage():
            image = source.imageData()
            uuid = hexuuid()
            document.addResource(QTextDocument.ImageResource, uuid, image)
            cursor.insertImage(uuid)
            return
        super(TextEdit, self).insertFromMimeData(source)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        layout = QVBoxLayout()
        self.editor = TextEdit()
        # Setup the QTextEdit editor configuration
        self.editor.setAutoFormatting(QTextEdit.AutoAll)
        self.editor.selectionChanged.connect(self.update_format)
        # Initialize default font size.
        font = QFont('Times', 16)
        self.editor.setFont(font)
        # We need to repeat the size to init the current format.
        self.editor.setFontPointSize(16)

        # self.path holds the path of the currently open file.
        # If none, we haven't got a file open yet (or creating new).
        self.path = None
        self.debugImages = None

        ######################## Recognition tool bar begin ########################

        rec_toolbar = QToolBar("Recogntion")
        rec_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(rec_toolbar)

        self.progressBar = QProgressBar(self)
        self.progressBar.hide()

        # Execute recognition
        run_recognition_action = QAction(QIcon(os.path.join('images', 'play.png')), "Start Recognition", self)
        run_recognition_action.setStatusTip("Start Recognition")
        run_recognition_action.triggered.connect(self.triggerRecognition)
        rec_toolbar.addAction(run_recognition_action)

        # Autocorrection
        self.autocorrect_action = QAction(QIcon(os.path.join('images', 'autocorrect.png')), "Autocorrection", self)
        self.autocorrect_action.setStatusTip("Automatic spelling correction")
        self.autocorrect_action.setCheckable(True)
        self.autocorrect_action.setChecked(True)
        self.autocorrect_action.toggled.connect(self.triggerRecognition)
        rec_toolbar.addAction(self.autocorrect_action)

        # Thickness
        self.thickness = QComboBox()
        self.thickness.adjustSize()
        self.thickness.addItems([str(i) for i in THICKNESS])
        self.thickness.currentIndexChanged[str].connect(self.triggerRecognition)
        rec_toolbar.addWidget(self.thickness)

        # Resolution
        self.resolution = QComboBox()
        self.resolution.adjustSize()
        self.resolution.addItems([str(i) for i in RESOLUTIONS])
        self.resolution.currentIndexChanged[str].connect(self.triggerRecognition)
        rec_toolbar.addWidget(self.resolution)

        # DebugImages
        self.debug_preview = QComboBox()
        self.debug_preview.adjustSize()
        self.debug_preview.addItems(DEBUG_IMAGES)
        self.debug_preview.currentIndexChanged[str].connect(self.updatePreviewImage)
        rec_toolbar.addWidget(self.debug_preview)

        self.rec_toolbar = rec_toolbar

        # add toolbar, image, progressbar and editor in horizontal layout
        hlayout = QHBoxLayout()
        gridLayout = QGridLayout()

        self.imageWidget = ImageLabel(self.path)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setOffset(0.0)
        self.imageWidget.setGraphicsEffect(shadow)

        gridLayout.addWidget(self.rec_toolbar)
        gridLayout.addWidget(self.imageWidget)
        gridLayout.addWidget(self.progressBar)
        gridLayout.setRowStretch(0,1)
        gridLayout.setColumnStretch(0,1)

        hlayout.addLayout(gridLayout)
        hlayout.addWidget(self.editor)
        layout.addLayout(hlayout)

        ######################### Recognition tool bar end  #########################

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Uncomment to disable native menubar on Mac
        # self.menuBar().setNativeMenuBar(False)

        file_toolbar = QToolBar("File")
        file_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(file_toolbar)
        file_menu = self.menuBar().addMenu("&File")

        open_file_action = QAction(QIcon(os.path.join('images', 'blue-folder-open-document.png')), "Open file...", self)
        open_file_action.setStatusTip("Open file")
        open_file_action.triggered.connect(self.file_open)
        file_menu.addAction(open_file_action)
        file_toolbar.addAction(open_file_action)

        saveas_file_action = QAction(QIcon(os.path.join('images', 'disk--pencil.png')), "Save As...", self)
        saveas_file_action.setStatusTip("Save current page to specified file")
        saveas_file_action.triggered.connect(self.file_saveas)
        file_menu.addAction(saveas_file_action)
        file_toolbar.addAction(saveas_file_action)

        print_action = QAction(QIcon(os.path.join('images', 'printer.png')), "Print...", self)
        print_action.setStatusTip("Print current page")
        print_action.triggered.connect(self.file_print)
        file_menu.addAction(print_action)
        file_toolbar.addAction(print_action)

        # Format tool bar
        format_toolbar = QToolBar("Format")
        format_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(format_toolbar)
        format_menu = self.menuBar().addMenu("&Format")

        # We need references to these actions/settings to update as selection changes, so attach to self.
        self.fonts = QFontComboBox()
        self.fonts.currentFontChanged.connect(self.editor.setCurrentFont)
        format_toolbar.addWidget(self.fonts)

        self.fontsize = QComboBox()
        self.fontsize.addItems([str(s) for s in FONT_SIZES])

        # Connect to the signal producing the text of the current selection. Convert the string to float
        # and set as the pointsize. We could also use the index + retrieve from FONT_SIZES.
        self.fontsize.currentIndexChanged[str].connect(lambda s: self.editor.setFontPointSize(float(s)) )
        format_toolbar.addWidget(self.fontsize)

        self.bold_action = QAction(QIcon(os.path.join('images', 'edit-bold.png')), "Bold", self)
        self.bold_action.setStatusTip("Bold")
        self.bold_action.setShortcut(QKeySequence.Bold)
        self.bold_action.setCheckable(True)
        self.bold_action.toggled.connect(lambda x: self.editor.setFontWeight(QFont.Bold if x else QFont.Normal))
        format_toolbar.addAction(self.bold_action)
        format_menu.addAction(self.bold_action)

        self.italic_action = QAction(QIcon(os.path.join('images', 'edit-italic.png')), "Italic", self)
        self.italic_action.setStatusTip("Italic")
        self.italic_action.setShortcut(QKeySequence.Italic)
        self.italic_action.setCheckable(True)
        self.italic_action.toggled.connect(self.editor.setFontItalic)
        format_toolbar.addAction(self.italic_action)
        format_menu.addAction(self.italic_action)

        self.underline_action = QAction(QIcon(os.path.join('images', 'edit-underline.png')), "Underline", self)
        self.underline_action.setStatusTip("Underline")
        self.underline_action.setShortcut(QKeySequence.Underline)
        self.underline_action.setCheckable(True)
        self.underline_action.toggled.connect(self.editor.setFontUnderline)
        format_toolbar.addAction(self.underline_action)
        format_menu.addAction(self.underline_action)

        format_menu.addSeparator()

        self.alignl_action = QAction(QIcon(os.path.join('images', 'edit-alignment.png')), "Align left", self)
        self.alignl_action.setStatusTip("Align text left")
        self.alignl_action.setCheckable(True)
        self.alignl_action.triggered.connect(lambda: self.editor.setAlignment(Qt.AlignLeft))
        format_toolbar.addAction(self.alignl_action)
        format_menu.addAction(self.alignl_action)

        self.alignc_action = QAction(QIcon(os.path.join('images', 'edit-alignment-center.png')), "Align center", self)
        self.alignc_action.setStatusTip("Align text center")
        self.alignc_action.setCheckable(True)
        self.alignc_action.triggered.connect(lambda: self.editor.setAlignment(Qt.AlignCenter))
        format_toolbar.addAction(self.alignc_action)
        format_menu.addAction(self.alignc_action)

        self.alignr_action = QAction(QIcon(os.path.join('images', 'edit-alignment-right.png')), "Align right", self)
        self.alignr_action.setStatusTip("Align text right")
        self.alignr_action.setCheckable(True)
        self.alignr_action.triggered.connect(lambda: self.editor.setAlignment(Qt.AlignRight))
        format_toolbar.addAction(self.alignr_action)
        format_menu.addAction(self.alignr_action)

        self.alignj_action = QAction(QIcon(os.path.join('images', 'edit-alignment-justify.png')), "Justify", self)
        self.alignj_action.setStatusTip("Justify text")
        self.alignj_action.setCheckable(True)
        self.alignj_action.triggered.connect(lambda: self.editor.setAlignment(Qt.AlignJustify))
        format_toolbar.addAction(self.alignj_action)
        format_menu.addAction(self.alignj_action)

        format_group = QActionGroup(self)
        format_group.setExclusive(True)
        format_group.addAction(self.alignl_action)
        format_group.addAction(self.alignc_action)
        format_group.addAction(self.alignr_action)
        format_group.addAction(self.alignj_action)

        format_menu.addSeparator()

        # A list of all format-related widgets/actions, so we can disable/enable signals when updating.
        self._format_actions = [
            self.fonts,
            self.fontsize,
            self.bold_action,
            self.italic_action,
            self.underline_action,
            # We don't need to disable signals for alignment, as they are paragraph-wide.
        ]

        # Initialize.
        self.update_format()
        self.update_title()

        # Load Model
        from segmentation import Segmenter
        from recognition import getModel, preprocess
        (self.model, self.b) = getModel()
        self.segmenter = Segmenter
        self.getModel = getModel
        self.preprocess = preprocess

        self.show()

    def block_signals(self, objects, b):
        for o in objects:
            o.blockSignals(b)

    def update_format(self):
        """
        Update the font format toolbar/actions when a new text selection is made. This is neccessary to keep
        toolbars/etc. in sync with the current edit state.
        :return:
        """
        # Disable signals for all format widgets, so changing values here does not trigger further formatting.
        self.block_signals(self._format_actions, True)

        self.fonts.setCurrentFont(self.editor.currentFont())
        # Nasty, but we get the font-size as a float but want it was an int
        self.fontsize.setCurrentText(str(int(self.editor.fontPointSize())))

        self.italic_action.setChecked(self.editor.fontItalic())
        self.underline_action.setChecked(self.editor.fontUnderline())
        self.bold_action.setChecked(self.editor.fontWeight() == QFont.Bold)

        self.alignl_action.setChecked(self.editor.alignment() == Qt.AlignLeft)
        self.alignc_action.setChecked(self.editor.alignment() == Qt.AlignCenter)
        self.alignr_action.setChecked(self.editor.alignment() == Qt.AlignRight)
        self.alignj_action.setChecked(self.editor.alignment() == Qt.AlignJustify)

        self.block_signals(self._format_actions, False)

    def dialog_critical(self, s):
        dlg = QMessageBox(self)
        dlg.setText(s)
        dlg.setIcon(QMessageBox.Critical)
        dlg.show()

    def dialog_message(self, s):
        dlg = QMessageBox(self)
        dlg.setText(s)
        dlg.show()

    def file_open(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open file", "", "HTML documents (*.html);Text documents (*.txt);All files (*.*)")
        text = ""

        try:
            if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.imageWidget.changePixmap(path)
            else:
                with open(path, 'rU') as f:
                    text = f.read()

        except Exception as e:
            self.dialog_critical(str(e))

        else:
            self.path = path
            # Qt will automatically try and guess the format as txt/html
            self.editor.setText(text)
            self.update_title()

    def file_save(self):
        if self.path is None:
            # If we do not have a path, we need to use Save As.
            return self.file_saveas()

        text = self.editor.toHtml() if splitext(self.path) in HTML_EXTENSIONS else self.editor.toPlainText()

        try:
            with open(self.path, 'w') as f:
                f.write(text)

        except Exception as e:
            self.dialog_critical(str(e))

    def file_saveas(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "HTML documents (*.html);Text documents (*.txt);All files (*.*)")

        if not path:
            # If dialog is cancelled, will return ''
            return

        text = self.editor.toHtml() if splitext(path) in HTML_EXTENSIONS else self.editor.toPlainText()

        try:
            with open(path, 'w') as f:
                f.write(text)

        except Exception as e:
            self.dialog_critical(str(e))

        else:
            self.path = path
            self.update_title()

    def file_print(self):
        dlg = QPrintDialog()
        if dlg.exec_():
            self.editor.print_(dlg.printer())

    def update_title(self):
        self.setWindowTitle("%s - Handwritten Document Recognizer" % (os.path.basename(self.path) if self.path else "Untitled"))

    def edit_toggle_wrap(self):
        self.editor.setLineWrapMode( 1 if self.editor.lineWrapMode() == 0 else 0 )

    def textFromImage(self, path):
        self.progressBar.show()
        self.progressBar.setValue(0)
        _thickness = int(self.thickness.currentText())
        _resolution = int(self.resolution.currentText())
        _autocorrect = self.autocorrect_action.isChecked()

        s = self.segmenter(path, thickness=_thickness, size=_resolution)
        self.debugImages = s.debugImages
        lines = s.getLineTexts()

        finalText = ""
        lineNums = len(lines)
        wordNums = len([word for line in lines for word in line])
        wordCount = 0
        totalProb = 0

        for line in lines:
            for img in line:
                try:
                    recognized, prob = self.recognize_with_autocorrection(img, _autocorrect)
                    finalText += recognized
                    totalProb += prob
                except Exception as e:
                    print("ERROR: ", e)
                    finalText += "______"
                finalText += " "
                wordCount += 1
                self.progressBar.setValue((wordCount/wordNums)*100)
            finalText += "\n"

        accuracy = np.around((totalProb/wordCount)*100, decimals=2)
        self.dialog_message(str(f"\t\t\t\nLines : {len(lines)}\nWords : {wordCount}"))
        self.progressBar.hide()
        self.debug_preview.setCurrentText(DEBUG_IMAGES[-1])
        self.updatePreviewImage()
        #self.model.sess.close()
        return finalText

    def updatePreviewImage(self):
        text = str(self.debug_preview.currentText())
        if text == 'Source':
            if self.path:
                self.imageWidget.changePixmap(self.path)
            return
        if self.debugImages:
            arr = []
            images = [Image.fromarray(i[0]).convert('RGB') for i in self.debugImages]
            for d in self.debugImages:
                if text == 'Blurred':
                    arr = images[0]
                if text == 'Edge':
                    arr = images[1]
                if text == 'Blob':
                    arr = images[2]
                if text == 'Bounding Boxes':
                    arr = images[4]
            self.imageWidget.changePixmapWithArray(arr)

    def recognize(self, img):
        _img = self.preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 32))
        _b = self.b(None, [_img])
        (recognized, probability) = self.model.inferBatch(_b, True)
        return recognized[0], probability[0]

    def recognize_with_autocorrection(self, img, autocorrect):
        text, prob = self.recognize(img)
        if autocorrect:
            if prob < 0.5:
                text = spell.correction(text)
        return text, prob

    def triggerRecognition(self, *args):
        if self.path:
            try:
                text = self.textFromImage(self.path)
            except Exception as e:
                self.dialog_critical(str(e))
            else:
                self.editor.setText(text)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("Handwritten Document Recognizer")

    # Dark Theme
    # app.setStyle("Fusion")
    # palette = QPalette()
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # palette.setColor(QPalette.WindowText, Qt.white)
    # palette.setColor(QPalette.Base, QColor(25, 25, 25))
    # palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    # palette.setColor(QPalette.ToolTipBase, Qt.black)
    # palette.setColor(QPalette.ToolTipText, Qt.white)
    # palette.setColor(QPalette.Text, Qt.white)
    # palette.setColor(QPalette.Button, QColor(53, 53, 53))
    # palette.setColor(QPalette.ButtonText, Qt.white)
    # palette.setColor(QPalette.BrightText, Qt.red)
    # palette.setColor(QPalette.Link, QColor(42, 130, 218))
    # palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    # palette.setColor(QPalette.HighlightedText, Qt.black)
    # app.setPalette(palette)

    window = MainWindow()
    app.exec_()
