# 3rd party
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QStyledItemDelegate


class TableCellDelegate(QStyledItemDelegate):
    """Делегат, переопределяющий для ячеек таблицы Qt свойства расположения текста и возможности редактирования содержимого"""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter

    def createEditor(self, parent, option, index):
        return None
