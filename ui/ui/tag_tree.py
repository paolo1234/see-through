from typing import List, Union, Tuple, Dict
import re

import numpy as np
from qtpy.QtWidgets import QSizePolicy, QMenu, QAction, QStyledItemDelegate, QLabel, QTreeView, QCheckBox, QStyleOptionViewItem, QVBoxLayout, QStyle, QMessageBox, QStyle,  QApplication, QWidget
from qtpy.QtCore import Qt, QItemSelectionModel, QItemSelection, QSize, Signal, QUrl, QModelIndex, QRectF, QRect, QEvent
from qtpy.QtGui import QFont, QTextCursor, QStandardItemModel, QStandardItem, QAbstractTextDocumentLayout, QColor, QPalette, QTextDocument, QTextCharFormat, QContextMenuEvent, QPixmap, QIcon, QPainter, QMouseEvent, QKeySequence, QShortcut

from .ui_config import pcfg
from .logger import logger as LOGGER
from .proj import ProjSeg
from . import shared
from .misc import ndarray2pixmap
from live2d.scrap_model import Drawable


TREECHR_FONTSIZE = 12

TREE_CHECKBOX_PADR = 32


class HTMLDelegate( QStyledItemDelegate ):
    def __init__(self, x_shift=0, y_shfit=0):
        super().__init__()
        self.doc = QTextDocument()
        self.doc.setUndoRedoEnabled(False)
        self.doc.setDocumentMargin(0)
        self.x_shift = x_shift
        self.y_shift = y_shfit

    def paint(self, painter: QPainter, option, index):                                                                                                                                                                               
        # https://wiki.qt.io/Center_a_QCheckBox_or_Decoration_in_an_Itemview
        widget = option.widget
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        painter.save()
        self.doc.setDefaultFont(opt.font)
        self.doc.setHtml(opt.text)
        
        if opt.state & QStyle.State_Selected:
            # colour = opt.palette.highlight().color()
            painter.fillRect(opt.rect, SEARCHRST_HIGHLIGHT_COLOR)

        opt.text = ''
        
        painter.translate(opt.rect.left(), opt.rect.top())

        clip = QRectF(0, 0, opt.rect.width(), opt.rect.height())
        painter.setClipRect(clip)
        ctx = QAbstractTextDocumentLayout.PaintContext()
        ctx.clip = clip
        ctx.palette.setColor(QPalette.ColorRole.Text, QColor(*shared.FOREGROUND_FONTCOLOR))
        painter.translate(self.x_shift, self.y_shift)
        self.doc.documentLayout().draw(painter, ctx)
        painter.restore()
        style = QApplication.style() if opt.widget is None else opt.widget.style()

        if not opt.icon.isNull():
            iconRect = style.subElementRect(QStyle.SE_ItemViewItemDecoration, opt, widget)
            iconRect = QStyle.alignedRect(opt.direction, Qt.AlignmentFlag.AlignLeft, iconRect.size(), opt.rect)
            mode = QIcon.Mode.Normal
            if not opt.state & QStyle.State_Enabled:
                mode = QIcon.Mode.Disabled
            elif opt.state & QStyle.State_Selected:
                mode = QIcon.Selected

            state = QIcon.State.On if opt.state & QStyle.State_Open else QIcon.State.Off
            opt.icon.paint(painter, iconRect, Qt.AlignmentFlag.AlignLeft, mode, state)

        if index.flags() & Qt.ItemIsUserCheckable:
            if opt.checkState == Qt.Unchecked:
                opt.state |= QStyle.State_Off
            elif opt.checkState == Qt.PartiallyChecked:
                opt.state |= QStyle.State_NoChange
            elif opt.checkState == Qt.Checked:
                opt.state |= QStyle.State_On
            rect = style.subElementRect(QStyle.SE_ItemViewItemCheckIndicator, opt, widget)
            opt.rect = QStyle.alignedRect(opt.direction, Qt.AlignmentFlag.AlignRight, rect.size(), opt.rect)
            opt.rect.setLeft(opt.rect.left() - TREE_CHECKBOX_PADR)
            opt.state = opt.state & ~QStyle.State_HasFocus
            style.drawPrimitive(QStyle.PE_IndicatorItemViewItemCheck, opt, painter, widget)

    def editorEvent(self, event: QMouseEvent, model, option, index):
        # https://wiki.qt.io/Center_a_QCheckBox_or_Decoration_in_an_Itemview
        flags = model.flags(index)
        if not (flags & Qt.ItemIsUserCheckable) or not (option.state & QStyle.State_Enabled) or not (flags & Qt.ItemIsEnabled):
            return False

        value = index.data(Qt.CheckStateRole)
        # print(value)
        # if not value.isValid():
        #     return False
        widget = option.widget
        style = widget.style() if option.widget else QApplication.style()

        if ((event.type() == QEvent.MouseButtonRelease) or (event.type() == QEvent.MouseButtonDblClick) or (event.type() == QEvent.MouseButtonPress)):
            viewOpt = QStyleOptionViewItem(option)
            self.initStyleOption(viewOpt, index)
            checkRect = style.subElementRect(QStyle.SE_ItemViewItemCheckIndicator, viewOpt, widget)
            checkRect = QStyle.alignedRect(viewOpt.direction, Qt.AlignmentFlag.AlignRight, checkRect.size(), viewOpt.rect)
            checkRect.setLeft(checkRect.left() - TREE_CHECKBOX_PADR)
            if (event.button() != Qt.LeftButton or not checkRect.contains(event.pos())):
                return False
            if ((event.type() == QEvent.MouseButtonPress) or (event.type() == QEvent.MouseButtonDblClick)):
                return True
        elif (event.type() == QEvent.KeyPress):
            if (event.key() != Qt.Key_Space and event.key() != Qt.Key_Select):
                return False
        else:
            return False
        
        state = Qt.CheckState(value)
        if (flags & Qt.ItemIsUserTristate):
            state = ((Qt.CheckState)((state + 1) % 3))
        else:
            state = Qt.Unchecked if state == Qt.Checked else Qt.Checked
        return model.setData(index, state, Qt.CheckStateRole)


SEARCHRST_HIGHLIGHT_COLOR = QColor(30, 147, 229, 60)

def get_rstitem_renderhtml(text: str, span: Tuple[int, int], font: QFont = None) -> str:
    if text == '':
        return text
    doc = QTextDocument()
    if font is None:
        font = doc.defaultFont()
    font.setPointSizeF(TREECHR_FONTSIZE)
    doc.setDefaultFont(font)
    doc.setPlainText(text.replace('\n', ' '))
    cursor = QTextCursor(doc)
    cursor.setPosition(span[0])
    cursor.setPosition(span[1], QTextCursor.MoveMode.KeepAnchor)
    cfmt = QTextCharFormat()
    cfmt.setBackground(SEARCHRST_HIGHLIGHT_COLOR)
    cursor.setCharFormat(cfmt)
    html = doc.toHtml()
    cleaned_html = re.findall(r'<body(.*?)>(.*?)</body>', html, re.DOTALL)
    if len(cleaned_html) > 0:
        cleaned_html = cleaned_html[0]
        return f'<body{cleaned_html[0]}>{cleaned_html[1]}</body>'
    else:
        return ''

class DrawableElements(QStandardItem):
    def __init__(self, did: str, tag: str, visible: bool = True, name: str = None):
        super().__init__()
        self.did = did
        self.tag = tag
        self.visible = visible
        self.name = name
        self.setText(name if name else did)
        font = self.font()
        font.setPointSizeF(TREECHR_FONTSIZE)
        self.setFont(font)
        self.setEditable(False)
        self.setCheckable(True)
        self.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
        self.setSelectable(True)

    def set_visible_state(self, visible: bool):
        self.visible = visible
        self.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)

    def set_display_name(self, name: str):
        self.name = name
        self.setText(name if name else self.did)


class TagItem(QStandardItem):
    def __init__(self, tag: str):
        super().__init__()
        # self.setData(result_counter, Qt.ItemDataRole.UserRole)
        self.tag = tag
        self.setText('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + tag)
        font = self.font()
        font.setPointSizeF(TREECHR_FONTSIZE)
        self.setFont(font)
        self.setEditable(False)
        icon = QPixmap(32, 32)
        icon.fill(QColor(*shared.get_cls_color(tag)))
        icon = QIcon(icon)
        self.setIcon(icon)
        self.setCheckable(True)
        self.setCheckState(Qt.CheckState.Checked)


class SearchResultModel(QStandardItemModel):
    # https://stackoverflow.com/questions/32229314/pyqt-how-can-i-set-row-heights-of-qtreeview
    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.SizeHintRole:
            size = QSize()
            item = self.itemFromIndex(index)
            size.setHeight(item.font().pointSize()+12)
            return size
        else:
            return super().data(index, role)


class TagTree(QTreeView):

    drawable_selection_changed = Signal(str, bool, bool)
    tag_selection_changed = Signal(str)
    _block_selection_signal = False

    reveal_drawable = Signal()
    set_selection_tag = Signal(str)

    # Fase 8: operazioni sui livelli
    delete_drawables = Signal()
    duplicate_drawables = Signal()
    rename_drawable = Signal()
    move_drawable_up = Signal()
    move_drawable_down = Signal()
    drawable_visibility_changed = Signal(str, bool)

    hide_tags = Signal(str)
    show_tags = Signal(str)
    hide_non_selected = Signal()
    show_all_tags = Signal()
    _block_hide_tag_signal = False

    propagate_page = Signal()

    search_page = Signal()

    def __init__(self, parent: QWidget = None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        
        self.hide_tag_shortcut = QShortcut(Qt.Key.Key_1, self)
        self.hide_tag_shortcut.activated.connect(lambda : self.hide_tags.emit(''))
        self.hide_tag_action = QAction(self.tr('Hide Selected Tags'), self, shortcut=Qt.Key.Key_1)

        self.show_tag_shortcut = QShortcut(Qt.Key.Key_2, self)
        self.show_tag_shortcut.activated.connect(lambda : self.show_tags.emit(''))
        self.show_tag_action = QAction(self.tr('Show Selected Tags'), self, shortcut=Qt.Key.Key_2)
        
        self.hide_non_sel_shortcut = QShortcut(Qt.Key.Key_3, self)
        self.hide_non_sel_shortcut.activated.connect(self.hide_non_selected)
        self.hide_non_sel_tag_action = QAction(self.tr('Hide All but Selected Tags'), self, shortcut=Qt.Key.Key_3)
        
        self.show_all_tags_shortcut = QShortcut(Qt.Key.Key_4, self)
        self.show_all_tags_shortcut.activated.connect(self.show_all_tags)
        self.show_all_tag_action = QAction(self.tr('Show all Tags'), shortcut=Qt.Key.Key_4)

        # Fase 8: operazioni layer da tastiera
        self.del_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self)
        self.del_shortcut.activated.connect(self.delete_drawables.emit)
        self.dup_shortcut = QShortcut(QKeySequence('Ctrl+D'), self)
        self.dup_shortcut.activated.connect(self.duplicate_drawables.emit)
        self.ren_shortcut = QShortcut(QKeySequence('F2'), self)
        self.ren_shortcut.activated.connect(self.rename_drawable.emit)
        self.show_shortcut = QShortcut(QKeySequence('H'), self)
        self.show_shortcut.activated.connect(lambda: self.set_drawables_visible(True))

        sm = SearchResultModel()
        self.sm = sm
        self.setItemDelegate(HTMLDelegate())
        self.root_item = sm.invisibleRootItem()
        self.setModel(sm)
        font = self.font()
        font.setPointSizeF(TREECHR_FONTSIZE)
        self.setFont(font)
        self.setUniformRowHeights(True)

        self.setHeaderHidden(True)
        self.expandAll()
        self.setMinimumWidth(450)
        self.setViewportMargins(0,0,0,0)
        self.setSelectionMode(self.selectionMode().ExtendedSelection)
        self.tag2page: dict[str, TagItem] = {}
        self.did2elem: dict[str, DrawableElements] = {}

        self.sm.dataChanged.connect(self.handle_item_data_changed)

    # def on_hide_tag(self):
    #     pass

    def handle_item_data_changed(self, top_left_index, bottom_right_index, roles):
        if self._block_selection_signal:
            return
        if Qt.CheckStateRole in roles:
            # Iterate through the changed items if multiple items changed
            for row in range(top_left_index.row(), bottom_right_index.row() + 1):
                for column in range(top_left_index.column(), bottom_right_index.column() + 1):
                    item = self.sm.item(row, column)
                    if item and item.isCheckable():
                        current_state = item.checkState()
                        checked = current_state == Qt.Checked
                        if isinstance(item, DrawableElements):
                            item.visible = checked
                            self.drawable_visibility_changed.emit(item.did, checked)
                        elif checked:
                            self.show_tags.emit(item.tag)
                        else:
                            self.hide_tags.emit(item.tag)

    def update_cls_list(self, cls_list):
        self.clearPages()
        for tag in cls_list:
            self.addPage(tag)
        self.addPage('None')
    
    def update_drawable_lst(self, drawable_id_lst, drawable_tag_lst,
                            drawable_visible_lst=None, drawable_name_lst=None):
        self._block_selection_signal = True
        tag2ids = {}
        for tag, page in self.tag2page.items():
            if page.rowCount() > 0:
                page.removeRows(0, page.rowCount())
            tag2ids[tag] = []

        visible_lst = drawable_visible_lst or [True] * len(drawable_id_lst)
        name_lst = drawable_name_lst or [None] * len(drawable_id_lst)

        for did, tag, vis, name in zip(drawable_id_lst, drawable_tag_lst, visible_lst, name_lst):
            if tag not in self.tag2page:
                tag = 'None'
            tag2ids[tag].append((did, vis, name))

        for tag, dids in tag2ids.items():
            page = self.tag2page[tag]
            for did, vis, name in dids:
                elem = DrawableElements(did, tag, visible=vis, name=name)
                page.appendRow(elem)
                self.did2elem[did] = elem

        self._block_selection_signal = False


    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection):
        super().selectionChanged(selected, deselected)
        if self._block_selection_signal:
            return
        
        
        tag_selection = []
        drawable_selection = []
        for ii, sel in enumerate(selected.indexes()):
            sel: DrawableElements = self.sm.itemFromIndex(sel)
            if isinstance(sel, DrawableElements):
                drawable_selection.append(sel)
            elif isinstance(sel, TagItem):
                tag_selection.append(sel)

        if len(drawable_selection) > 0:
            n_selected = len(drawable_selection)
            for ii, sel in enumerate(drawable_selection):
                is_last = ii + 1 == n_selected
                self.drawable_selection_changed.emit(sel.did, True, is_last)
        elif len(tag_selection) > 0:
            self.tag_selection_changed.emit(tag_selection[-1].tag)
                # 
        for sel in deselected.indexes():
            sel: DrawableElements = self.sm.itemFromIndex(sel)
            if isinstance(sel, DrawableElements):
                self.drawable_selection_changed.emit(sel.did, False, False)

    def update_tag(self, dids, tag_list: str):
        self._block_selection_signal = True
        sel_ids = self.selectionModel().selectedIndexes()
        sel_item_ids = []
        for idx in sel_ids:
            item = self.sm.itemFromIndex(idx)
            if not isinstance(item, DrawableElements):
                continue
            sel_item_ids.append(item.did)
        sel_item_ids = set(sel_item_ids)
        items: list[DrawableElements] = []
        for did in dids:
            items.append(self.did2elem[did])
        for elem, tag in zip(items, tag_list):
            if tag is None:
                tag = 'None'
            page = self.tag2page[tag]
            if elem.tag != tag:
                selected = elem.did in sel_item_ids
                did = elem.did
                self.tag2page[elem.tag].removeRow(elem.row())
                elem = DrawableElements(did, tag)
                self.did2elem[did] = elem
                page.appendRow(elem)
                if selected:
                    self.selectionModel().select(elem.index(), QItemSelectionModel.SelectionFlag.Select)
        self._block_selection_signal = False

    def addPage(self, tag: str) -> TagItem:
        prst = TagItem(tag)
        self.root_item.appendRow(prst)
        self.tag2page[tag] = prst
        return prst

    def clearPages(self):
        self._block_selection_signal = True
        self.tag2page.clear()
        self.did2elem.clear()
        rc = self.root_item.rowCount()
        if rc > 0:
            self.root_item.removeRows(0, rc)
        self._block_selection_signal = False

    def update_seleciton(self, did, selected, block_signal=True):
        if block_signal:
            self._block_selection_signal = True
        elem = self.did2elem.get(did)
        if elem is None:
            # albero non ancora sincronizzato (es. parti appena applicate)
            LOGGER.debug('tag_tree: did %s non presente nella tree', did)
            if block_signal:
                self._block_selection_signal = False
            return
        sel_model = self.selectionModel()
        flag = QItemSelectionModel.SelectionFlag.Select if selected else QItemSelectionModel.SelectionFlag.Deselect
        sel_model.select(elem.index(), flag)
        if block_signal:
            self._block_selection_signal = False

    def clear_drawable_selection(self, block_signal=True):
        if block_signal:
            self._block_selection_signal = True

        sel_model = self.selectionModel()
        sel_ids = self.selectionModel().selectedIndexes()

        for idx in sel_ids:
            item = self.sm.itemFromIndex(idx)
            if not isinstance(item, DrawableElements):
                continue
            sel_model.select(item.index(), QItemSelectionModel.SelectionFlag.Deselect)

        if block_signal:
            self._block_selection_signal = False

    def contextMenuEvent(self, e: QContextMenuEvent):
        global_pos = e.globalPos()
        e.setAccepted(True)
        self.create_context_menu(global_pos)

    def create_context_menu(self, pos):
        menu = QMenu()

        set_tag_menu = QMenu(title='Set Tag')
        tag_actions = []
        tg_lst = shared.cls_list + ['None']
        for t in tg_lst:
            a = QAction(t, set_tag_menu)
            tag_actions.append(a)
        set_tag_menu.addActions(tag_actions)

        menu.addMenu(set_tag_menu)
        menu.addSeparator()

        has_selection = len(self.get_selected_drawable_ids()) > 0
        act_delete = QAction(self.tr('Elimina layer (Canc)'), menu)
        act_duplicate = QAction(self.tr('Duplica layer (Ctrl+D)'), menu)
        act_rename = QAction(self.tr('Rinomina layer (F2)'), menu)
        act_up = QAction(self.tr('Porta su (ordine disegno)'), menu)
        act_down = QAction(self.tr('Porta giù (ordine disegno)'), menu)
        act_show = QAction(self.tr('Mostra selezionati (H)'), menu)
        act_hide = QAction(self.tr('Nascondi selezionati'), menu)
        for a in (act_delete, act_duplicate, act_rename, act_up, act_down, act_show, act_hide):
            a.setEnabled(has_selection)
        menu.addAction(act_delete)
        menu.addAction(act_duplicate)
        menu.addAction(act_rename)
        menu.addSeparator()
        menu.addAction(act_up)
        menu.addAction(act_down)
        menu.addSeparator()
        menu.addAction(act_show)
        menu.addAction(act_hide)
        menu.addSeparator()

        hide_tags = menu.addAction(self.hide_tag_action)
        show_tags = menu.addAction(self.show_tag_action)
        hide_non_selected = menu.addAction(self.hide_non_sel_tag_action)
        show_all_tags = menu.addAction(self.show_all_tag_action)

        menu.addSeparator()
        propagate_page = menu.addAction(self.tr('Propagate from previous page'))
        reveal_act = menu.addAction(self.tr('Reveal in File Explorer'))
        search_act = QAction(self.tr('Search drawables'), shortcut=QKeySequence('Ctrl+F'), parent=menu)
        menu.addAction(search_act)

        rst = menu.exec_(pos)

        if rst == reveal_act:
            self.reveal_drawable.emit()
        elif rst == self.hide_tag_action:
            self.hide_tags.emit('')
        elif rst == self.show_tag_action:
            self.show_tags.emit('')
        elif rst == self.hide_non_sel_tag_action:
            self.hide_non_selected.emit()
        elif rst == self.show_all_tag_action:
            self.show_all_tags.emit()
        elif rst == propagate_page:
            self.propagate_page.emit()
        elif rst == search_act:
            self.search_page.emit()
        elif rst == act_delete:
            self.delete_drawables.emit()
        elif rst == act_duplicate:
            self.duplicate_drawables.emit()
        elif rst == act_rename:
            self.rename_drawable.emit()
        elif rst == act_up:
            self.move_drawable_up.emit()
        elif rst == act_down:
            self.move_drawable_down.emit()
        elif rst == act_show:
            self.set_drawables_visible(True)
        elif rst == act_hide:
            self.set_drawables_visible(False)
        elif rst in tag_actions:
            tag = tg_lst[tag_actions.index(rst)]
            self.set_selection_tag.emit(tag)


    def get_selected_drawable_ids(self):
        sel_ids = self.selectionModel().selectedIndexes()
        dids = []
        for index in sel_ids:
            elem: DrawableElements = self.sm.itemFromIndex(index)
            if not isinstance(elem, DrawableElements):
                continue
            dids.append(elem.did)
        return dids

    def set_drawables_visible(self, visible: bool):
        """Mostra/nascondi i drawable selezionati (emette il segnale per comando)."""
        for did in self.get_selected_drawable_ids():
            self.drawable_visibility_changed.emit(did, visible)

    def get_selected_tags(self, get_non_selected=False):
        sel_ids = self.selectionModel().selectedIndexes()
        tags = set()
        for index in sel_ids:
            elem: DrawableElements = self.sm.itemFromIndex(index)
            tags.add(elem.tag)
        if get_non_selected:
            _tags = []
            for t in shared.cls_list + ['None']:
                if t not in tags:
                    _tags.append(t)
            tags = _tags
        return list(tags)

    def setTagCheckstate(self, tags, checked):
        self.blockSignals(True)
        check_state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for t in tags:
            page = self.tag2page[t]
            page.setCheckState(check_state)
        self.blockSignals(False)

class PreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._pixmap = None
        # self.setMinimumHeight(270)
        # self.setMinimumWidth(200)

    # def setImage(self, image: Union[np.ndarray, QImage, QPixmap]):
    #     if isinstance(image, np.ndarray):
    #         image = ndarray2pixmap(image)
    #     elif isinstance(image, QImage):
    #         image = QPixmap(image)
    #     self.setPixmap(image)


    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        scaled_pixmap = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap) # Set the scaled pixmap to the label
    

    def resizeEvent(self, event):
        if self._pixmap:
            self.setPixmap(self._pixmap)
        super().resizeEvent(event)


class DrawablePreview(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lo = QVBoxLayout(self)
        self.tag_path_label = PreviewLabel(parent=self)
        self.tag_path_label.setMinimumHeight(400)
        
        self.drawable_label = PreviewLabel(parent=self)
        self.drawable_label.setMinimumHeight(100)
        self.drawable_label.setSizePolicy(self.tag_path_label.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Maximum)
        self.setMinimumHeight(500)
        lo.addWidget(self.drawable_label)
        lo.addWidget(self.tag_path_label)
        self.tag = 'None'
        pass

    def updateDrawable(self, drawable: Drawable, tag_img):
        if tag_img is None:
            return
        
        pixmap = ndarray2pixmap(drawable.get_img())
        self.drawable_label.setPixmap(pixmap)

        pixmap = ndarray2pixmap(tag_img)
        self.tag_path_label.setPixmap(pixmap)

        self.tag = drawable.tag

    def updateTag(self, tag, tag_img, clear_preview_mask=False):
        self.tag = tag
        if tag_img is not None:
            pixmap = ndarray2pixmap(tag_img)
        else:
            pixmap = QPixmap(1, 1)
            pixmap.fill(Qt.GlobalColor.transparent)
        self.tag_path_label.setPixmap(pixmap)
        if clear_preview_mask:
            pixmap = QPixmap(1, 1)
            pixmap.fill(Qt.GlobalColor.transparent)
            self.drawable_label.setPixmap(pixmap)

        # self.setPixmap(pixmap)