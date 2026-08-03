from qtpy.QtWidgets import QCheckBox, QLabel, QMenu, QStackedWidget, QHBoxLayout, QToolButton, QVBoxLayout
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QMouseEvent, QIcon, QKeyEvent, QIntValidator
from .widget import Widget
from .combobox import ComboBox, SmallComboBox, SmallSizeComboBox, SizeComboBox
from .label import SmallParamLabel, ParamNameLabel, SizeControlLabel, SmallSizeControlLabel
from .message import TaskProgressBar
# from utils.torch_utils import AVAILABLE_DEVICES
from .ui_config import SegModel, AVAILABLE_SEGMODELS, ProgramConfig, pcfg, save_config
from .inference_backend import AVAILABLE_PROVIDERS

AVAILABLE_DEVICES = ['cpu', 'cuda']


def combobox_with_label(param_name: str = None, size='small', options=None, parent=None, scrollWidget=None, label_alignment=None, vertical_layout=False, editable=False, combobox_cls=None):
    if combobox_cls is None:
        combobox_cls = SmallComboBox if size == 'small' else ComboBox
    combobox = combobox_cls(options=options, parent=parent, scrollWidget=scrollWidget)
    combobox.setEditable(editable)
    label_cls = SmallParamLabel if size == 'small' else ParamNameLabel
    label = label_cls(param_name=param_name, alignment=label_alignment)
    if vertical_layout:
        layout = QVBoxLayout()
    else:
        layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(combobox)
    return combobox, label, layout

def size_combobox_with_label(param_name: str, value_range, init_value=0., size='small', label_alignment=None, vertical_layout=False):
    combobox_cls = SmallSizeComboBox if size == 'small' else SizeComboBox
    combobox = combobox_cls(val_range=value_range, init_value=init_value)
    label_cls = SmallParamLabel if size == 'small' else ParamNameLabel
    label = label_cls(param_name=param_name, alignment=label_alignment)
    if vertical_layout:
        layout = QVBoxLayout()
    else:
        layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(combobox)
    return combobox, label, layout


def checkbox_with_label(param_name: str = None, size='small', options=None, parent=None, scrollWidget=None, label_alignment=None, vertical_layout=False):
    combobox_cls = SmallComboBox if size == 'small' else ComboBox
    combobox = combobox_cls(options=options, parent=parent, scrollWidget=scrollWidget)
    label_cls = SmallParamLabel if size == 'small' else ParamNameLabel
    label = label_cls(param_name=param_name, alignment=label_alignment)
    if vertical_layout:
        layout = QVBoxLayout()
    else:
        layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(combobox)
    return combobox, label, layout


class SegParamsWidget(Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.refine_checker = QCheckBox(self, text=self.tr('Refine'))
        self.device_selector, _, device_lo = combobox_with_label(self.tr('Device'), options=AVAILABLE_DEVICES, editable=False)
        self.confidence_thr, _, conf_lo = size_combobox_with_label(self.tr('Confidence Threshold'), value_range=(0.1, 1), init_value=0.3)
        layout = QHBoxLayout(self)
        layout.addWidget(self.refine_checker)
        layout.addLayout(device_lo)
        layout.addLayout(conf_lo)
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.addStretch(-1)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)


class TagCombobox(SmallComboBox):

    enter_pressed = Signal()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self.enter_pressed.emit()
        return super().keyPressEvent(e)
    
        
class RunButton(QToolButton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.running = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon);
        self.setRunState()

    def setRunState(self):
        self.running = False
        self.setText(self.tr('Run'))
        self.setIcon(QIcon('assets/icons/run-start.svg'))

    def setStopState(self):
        self.running = True
        self.setText(self.tr('Stop'))
        self.setIcon(QIcon('assets/icons/run-stop.svg'))
    

class TopArea(Widget):

    tag_changed = Signal(str)
    # Fase 2: run batch / tool di prompt
    run_requested = Signal(bool)
    prompt_tool = Signal(str)  # 'box' | 'point' | 'none'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.show_colormsk_checkbox = QCheckBox(self.tr('Color Mask'), parent=self)
        
        self.mask_opacity_box = SmallSizeComboBox([0, 100], 'Opacity', self, validator_cls=QIntValidator)
        self.mask_opacity_box.setToolTip(self.tr("Opacity"))
        # self.mask_opacity_box.param_changed.connect(self.mask_opacity_changed)
        self.mask_opacity_label = SmallSizeControlLabel(self, direction=0, text='Opacity', alignment=Qt.AlignmentFlag.AlignCenter)
        self.mask_opacity_label.size_ctrl_changed.connect(lambda x: self.mask_opacity_box.changeByDelta(x, multiplier=1))
        # self.mask_opacity_label.btn_released.connect(self.mask_opacity_changed)
        mask_opacity_layout = QHBoxLayout()
        mask_opacity_layout.addWidget(self.mask_opacity_label)
        mask_opacity_layout.addWidget(self.mask_opacity_box)

        self.show_contour_checkbox = QCheckBox(self.tr('Contour'), parent=self)

        self.cls_list_combobox, _, cls_list_combobox_lo = combobox_with_label(options=AVAILABLE_SEGMODELS, param_name=self.tr('Tag'), size='small', editable=True, combobox_cls=TagCombobox)
        # self._block_tag_change_signal = False
        self.cls_list_combobox.currentTextChanged.connect(self.on_tag_changed)
        self.cls_list_combobox.enter_pressed.connect(self.on_tag_changed)

        self.search_label = QLabel(text=self.tr('Search Drawables with Keyword'))

        self.valid_checkbox = QCheckBox(self.tr('Valid'), parent=self)
        self.incomplete_checkbox = QCheckBox(self.tr('Is Incomplete'), parent=self)
        # self.parsing_combobox, _, parsing_combobox_lo = combobox_with_label(options=[], param_name=self.tr('Parsing File'), size='small', editable=False)
        # self.parsing_combobox.setMinimumWidth(150)
        # self.seg_params_widget = SegParamsWidget()

        self.model_stack_widget = QStackedWidget(parent=self)
        self.seg_params_widget = SegParamsWidget()
        self.model_stack_widget.addWidget(self.seg_params_widget)

        self.run_btn = RunButton()
        self.run_btn.pressed.connect(self.on_run_pressed)

        self.provider_selector, _, provider_lo = combobox_with_label(
            self.tr('Engine'), options=AVAILABLE_PROVIDERS, size='small', editable=False)
        self.provider_selector.currentTextChanged.connect(self.on_provider_changed)

        from .inference_backend import SAMProvider
        self.sam_size_selector, _, samsize_lo = combobox_with_label(
            self.tr('SAM'), options=list(SAMProvider.SIZES.keys()),
            size='small', editable=False)
        self.sam_size_selector.currentTextChanged.connect(self.on_sam_size_changed)

        self.box_tool_check = QCheckBox(self.tr('Box'), parent=self)
        self.point_tool_check = QCheckBox(self.tr('Point'), parent=self)
        self.box_tool_check.toggled.connect(self.on_prompt_tool)
        self.point_tool_check.toggled.connect(self.on_prompt_tool)

        # feedback di stato: messaggi su run/segmentazione/candidati
        self.status_label = QLabel(self.tr('Pronto — scegli Engine e premi Run, oppure W=Box / P=Point'), parent=self)
        self.status_label.setObjectName('StatusLabel')
        self.status_label.setWordWrap(False)
        self.status_label.setFixedHeight(20)
        self.status_label.setContentsMargins(6, 0, 6, 0)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.show_colormsk_checkbox)
        model_layout.addLayout(mask_opacity_layout)
        model_layout.addWidget(self.show_contour_checkbox)
        model_layout.addWidget(self.run_btn)
        model_layout.addLayout(cls_list_combobox_lo)
        model_layout.addLayout(provider_lo)
        model_layout.addLayout(samsize_lo)
        model_layout.addWidget(self.box_tool_check)
        model_layout.addWidget(self.point_tool_check)
        model_layout.addWidget(self.model_stack_widget)
        model_layout.addWidget(self.status_label)
        model_layout.addStretch(-1)
        model_layout.addWidget(self.incomplete_checkbox)
        model_layout.addWidget(self.valid_checkbox)
        # model_layout.addLayout(parsing_combobox_lo)
        # tool_layout = QHBoxLayout()

        layout = QHBoxLayout(self)
        layout.addLayout(model_layout)
        # layout.addLayout(tool_layout)
        margin = layout.contentsMargins().left()
        layout.setContentsMargins(margin, 0, margin, 0)
        self.setMaximumHeight(64)

    def set_status(self, msg: str):
        self.status_label.setText(msg)
        self.status_label.setToolTip(msg)

    # ---------- Fase 2: handler run/prompt ----------
    def on_provider_changed(self, name: str):
        if pcfg.inference_provider != name:
            pcfg.inference_provider = name
            save_config()
        self.sam_size_selector.setEnabled(name == 'sam')
        if name == 'dummy':
            self.set_status('Engine \'dummy\': candidati fittizi per provare il flusso — scegli SAM per la segmentazione reale.')
        elif name == 'sam':
            self.set_status('Engine SAM: premi Run per segmentare tutte le pagine, oppure W=Box / P=Point.')
        else:
            self.set_status(f'Engine \'{name}\': premi Run per iniziare.')

    def on_sam_size_changed(self, size: str):
        if pcfg.sam_model_size != size:
            pcfg.sam_model_size = size
            save_config()

    def on_run_pressed(self):
        # flip dello stato al click: Run->Stop (parte il batch) / Stop->Run (stop)
        if self.run_btn.running:
            self.run_btn.setRunState()
        else:
            self.run_btn.setStopState()
        self.run_requested.emit(self.run_btn.running)

    def on_prompt_tool(self):
        box = self.box_tool_check.isChecked()
        point = self.point_tool_check.isChecked()
        self.point_tool_check.blockSignals(True)
        self.box_tool_check.blockSignals(True)
        if box:
            self.point_tool_check.setChecked(False)
        elif point:
            self.box_tool_check.setChecked(False)
        self.point_tool_check.blockSignals(False)
        self.box_tool_check.blockSignals(False)
        if point:
            self.prompt_tool.emit('point')
        elif box:
            self.prompt_tool.emit('box')
        else:
            self.prompt_tool.emit('none')

    def set_running(self, running: bool):
        if running:
            self.run_btn.setStopState()
        else:
            self.run_btn.setRunState()

    def setupConfig(self, config: ProgramConfig):
        segparams = self.seg_params_widget
        segparams.refine_checker.setChecked(config.segmentation_refine)
        segparams.device_selector.setCurrentText(config.segmentation_device)
        segparams.confidence_thr.setValue(config.segmentation_conf_thr)
        self.provider_selector.blockSignals(True)
        self.provider_selector.setCurrentText(config.inference_provider)
        self.provider_selector.blockSignals(False)
        self.sam_size_selector.blockSignals(True)
        self.sam_size_selector.setCurrentText(config.sam_model_size)
        self.sam_size_selector.blockSignals(False)
        self.sam_size_selector.setEnabled(config.inference_provider == 'sam')

    # def block_tag_change_signal(self, block: bool):
    #     self._block_tag_change_signal = block

    def on_tag_changed(self):
        # if self._block_tag_change_signal:
        #     return
        self.tag_changed.emit(self.cls_list_combobox.currentText())


    def update_cls_list(self, cls_list):
        self.cls_list_combobox.blockSignals(True)
        self.cls_list_combobox.clear()
        self.cls_list_combobox.addItems(cls_list + ['None'])
        self.cls_list_combobox.blockSignals(False)

    def set_tag(self, tag):
        self.cls_list_combobox.blockSignals(True)
        self.cls_list_combobox.setCurrentText(tag)
        self.cls_list_combobox.blockSignals(False)

    def set_valid(self, valid):
        self.valid_checkbox.blockSignals(True)
        self.valid_checkbox.setChecked(valid)
        self.valid_checkbox.blockSignals(False)

    def set_incomplete(self, incomplete):
        self.incomplete_checkbox.blockSignals(True)
        self.incomplete_checkbox.setChecked(incomplete)
        self.incomplete_checkbox.blockSignals(False)
