import sys
import subprocess
import json
from pathlib import Path
import os.path as osp
import traceback
import shutil
import os

from PIL import Image
import numpy as np
from qtpy.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem, QApplication, QListWidget, QMenu, QStackedWidget, QHBoxLayout, QSplitter, QVBoxLayout, QShortcut, QPushButton, QInputDialog, QWidget, QAbstractItemView, QCheckBox
from qtpy.QtCore import Signal, QSize, Qt, QRectF
from qtpy.QtGui import QGuiApplication, QContextMenuEvent, QIcon, QCloseEvent, QKeySequence
import py7zr

from utils.io_utils import get_all_segcls
from . import shared_widget as SW
from .canvas import Canvas, DrawableItem
from .misc import parse_stylesheet, QKEY
from .framelesswindow import FramelessWindow
from .mainwindowbars import LeftBar, TitleBar, BottomBar
from .commands import (DeleteInstancesCommand, DuplicateInstancesCommand,
                       MoveInstanceCommand, SetInstancesVisibleCommand,
                       RenameInstanceCommand, ReorderInstancesCommand,
                       PasteInstancesCommand)
from .tag_tree import TagTree, DrawablePreview
from .io_thread import ProjSaveThread
from .message import FrameLessMessageBox, MessageBox
from .proj import ProjSeg
from .commands import SetDrawableTagCommand, CommonCommand, AddInstancesCommand, CreateDrawablesCommand, EditMaskCommand, SplitInstanceCommand, MergeInstancesCommand
from .top_area import TopArea
from .widget import Widget
from .run_thread import SegmentationThread
from .inference_backend import get_provider
from .ui_config import ProgramConfig, pcfg, save_config, EditMode, SegModel
from . import shared
from .logger import create_error_dialog, create_info_dialog
from .logger import logger as LOGGER



class PageListView(QListWidget):

    reveal_file = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setIconSize(QSize(shared.PAGELIST_THUMBNAIL_SIZE, shared.PAGELIST_THUMBNAIL_SIZE))
        self.setMinimumWidth(200)

    # def sizeHint(self) -> QSize:
    #     size = super().sizeHint()
    #     size.setWidth(360)
    #     return size

    def contextMenuEvent(self, e: QContextMenuEvent):
        menu = QMenu()
        reveal_act = menu.addAction(self.tr('Reveal in File Explorer'))
        rst = menu.exec_(e.globalPos())

        if rst == reveal_act:
            self.reveal_file.emit()

        return super().contextMenuEvent(e)


class MainWindow(FramelessWindow):

    restart_signal = Signal()
    proj = ProjSeg()
    create_errdialog = Signal(str, str, str)
    create_infodialog = Signal(dict, str)
    close_infodialog = Signal(str)
    
    def __init__(self, app: QApplication, config: ProgramConfig) -> None:
        super().__init__()
        self.app = app
        self.config = config
        self.save_on_page_changed = True
        self.opening_dir = False
        self.setupThread()
        self.setupUI()
        self.setupConfig()
        self.setupSignalSlots()
        self.setupShortcuts()

        self.showMaximized()
        self.setAcceptDrops(True)
        self.download_model_msg = self.tr('Downloading models...')

        shared.create_errdialog_in_mainthread = self.create_errdialog.emit
        self.create_errdialog.connect(self.on_create_errdialog)
        shared.create_infodialog_in_mainthread = self.create_infodialog.emit
        self.create_infodialog.connect(self.on_create_infodialog)
        shared.close_infodialog = self.close_infodialog.emit
        self.close_infodialog.connect(self.on_close_infodialog)

        if pcfg.open_recent_on_startup:
            if len(self.leftBar.recent_proj_list) > 0:
                proj_dir = self.leftBar.recent_proj_list[0]
                if osp.exists(proj_dir):
                    self.openProj(proj_dir)

    def setupSignalSlots(self):
        '''
        after setup config
        '''
        # self.topArea.seg_params_widget.device_selector.currentIndexChanged.connect(self.on_cartoonseg_device_changed)
        # self.topArea.seg_params_widget.confidence_thr.param_changed.connect(self.on_cartoonseg_confthr_changed)
        # self.topArea.seg_params_widget.refine_checker.stateChanged.connect(self.on_cartoonseg_refine_changed)
        self.topArea.tag_changed.connect(self.on_tag_changed)
        self.topArea.show_colormsk_checkbox.checkStateChanged.connect(self.on_show_colormsk)
        self.topArea.show_contour_checkbox.checkStateChanged.connect(self.on_show_contour)
        self.topArea.mask_opacity_label.btn_released.connect(self.on_set_colormsk_opacity)
        self.topArea.mask_opacity_box.param_changed.connect(self.on_set_colormsk_opacity)
        self.topArea.valid_checkbox.checkStateChanged.connect(self.on_set_page_valid)
        self.topArea.incomplete_checkbox.checkStateChanged.connect(self.on_set_page_incomplete)

        # Fase 2: inferenza interattiva (batch/box/point) + candidati
        self.topArea.run_requested.connect(self.on_run_requested)
        self.topArea.prompt_tool.connect(self.on_prompt_tool)
        self.canvas.point_prompted.connect(self.on_point_prompted)
        self.canvas.end_create_rect.connect(self.on_end_create_rect)
        self.canvas.export_cutout.connect(self.export_cutout)
        self.run_thread.progress.connect(self.on_run_progress)
        self.run_thread.status_msg.connect(self.on_run_status)
        self.topArea.anim_requested.connect(self.on_anim_requested)
        self.topArea.settings_requested.connect(self.on_settings_requested)
        self.candidates_list.itemDoubleClicked.connect(self.on_candidate_activate)
        self.apply_candidates_btn.clicked.connect(self.apply_candidates)
        self.merge_sel_btn.clicked.connect(self.on_merge_selected)
        self.show_cand_check.toggled.connect(self.on_show_candidates_toggled)
        self.on_show_candidates_toggled(self.show_cand_check.isChecked())
        
        # self.topArea.show_colormask.connect(self.on_show_colormask)
        self.titleBar.undo_trigger.connect(self.canvas.undo)
        self.titleBar.redo_trigger.connect(self.canvas.redo)
        self.titleBar.page_search_trigger.connect(self.on_show_page_search)

        # self.leftBar.rectTool.stateChanged.connect(self.on_recttool_state_changed)
        self.leftBar.export_result.connect(self.on_export_result)

        # self.canvas.instance_preview_area.contextmenu_requested.connect(self.canvas.on_create_contextmenu)
        self.bottomBar.scaleEditor.edit_value_changed.connect(self.on_edit_scale_changed)
        self.canvas.scalefactor_changed.connect(self.on_canvas_scale_changed)
        self.canvas.drawable_selection_changed.connect(self.on_canvas_selection_changed)
        self.canvas.search_widget.search.connect(self.on_page_search)
        self.canvas.context_menu_requested.connect(self.tagtree.create_context_menu)

        self.tagtree.drawable_selection_changed.connect(self.on_tagtree_selection_changed)
        self.tagtree.tag_selection_changed.connect(self.on_tag_selection_changed)
        self.tagtree.set_selection_tag.connect(lambda x: self.on_tag_changed(x, set_combobox=True))
        self.tagtree.reveal_drawable.connect(self.on_reveal_drawable)
        self.tagtree.hide_tags.connect(self.on_hide_tags)
        self.tagtree.show_tags.connect(self.on_show_tags)
        self.tagtree.hide_non_selected.connect(self.on_hide_non_selected_tags)
        self.tagtree.show_all_tags.connect(self.on_show_all_tags)
        self.tagtree.propagate_page.connect(self.on_propagate_page)
        self.tagtree.search_page.connect(self.on_show_page_search)

        # Fase 8: operazioni layer (menu contestuale + shortcut tree)
        self.tagtree.delete_drawables.connect(self.on_delete_drawables)
        self.tagtree.duplicate_drawables.connect(self.on_duplicate_drawables)
        self.tagtree.rename_drawable.connect(self.on_rename_drawable)
        self.tagtree.move_drawable_up.connect(lambda: self.on_reorder_selected(-1))
        self.tagtree.move_drawable_down.connect(lambda: self.on_reorder_selected(1))
        self.tagtree.drawable_visibility_changed.connect(self.on_drawable_visibility_changed)

    def on_show_page_search(self):
        if not self.canvas.gv.isVisible():
            return
        if self.canvas.search_widget.isHidden():
            self.canvas.search_widget.show()
        self.canvas.search_widget.focus_to_editor()
        # self.on_page_search()

    # def on_page_search(self):

    def on_page_search(self):
        sw = self.canvas.search_widget
        sw.clearSearchResult()
        self.tagtree.clear_drawable_selection(block_signal=False)

        if not sw.isVisible():
            return
    
        if not self.proj.model_valid:
            return

        text = sw.search_editor.toPlainText()
        if text == '':
            sw.updateCounterText()
            return

        selected_tags = self.tagtree.get_selected_tags()
        search_all_tags = len(selected_tags) == 0

        l2model = self.proj.l2dmodel
        pattern = sw.get_regex_pattern()
        matched_dids = []
        for d in l2model.valid_drawables():
            if not search_all_tags and d.tag not in selected_tags:
                continue
            if pattern.search(d.did):
                self.tagtree.update_seleciton(d.did, selected=True, block_signal=False)
                matched_dids.append(d.did)

    def on_propagate_page(self):
        prev_page_valid = True
        if self.proj.current_idx == 0:
            prev_page_valid = False
        else:
            prev_page = self.proj.idx2pagename(self.proj.current_idx - 1)
            prev_page_valid = osp.dirname(self.proj.current_model) == osp.dirname(prev_page)
            pass
        if not prev_page_valid:
            create_info_dialog(f'Previous page is not valid!')
            return
        
        from live2d.scrap_model import match_drawable_to_tag_voting, get_tag_voting_from_lmodel
        pre_path = osp.join(self.proj.directory, prev_page)
        lmodel, dir2tag, did2tag = get_tag_voting_from_lmodel(pre_path, seg_type=pcfg.seg_type, parsing_src=pcfg.parsing_src)
        src_matching, rst_matching, (id_matched, dir_matched) = match_drawable_to_tag_voting(self.proj.l2dmodel, dir2tag, did2tag, check_area=True)
        if len(rst_matching) > 0:
            self.canvas.push_undo_command(
                CommonCommand(
                    func=self.update_tag,
                    redo_kwargs={'did2tag': rst_matching},
                    undo_kwargs={'did2tag': src_matching}
                )
            )

            self.set_page_valid(lmodel._body_parsing['metadata'].get('is_valid', True))
            self.set_page_incomplete(lmodel._body_parsing['metadata'].get('is_incomplete', False))
            if len(dir_matched) > 0:
                recheck = {}
                for did in dir_matched:
                    tag = rst_matching[did]
                    if tag in recheck:
                        dlist = recheck[tag]
                    else:
                        dlist = []
                        recheck[tag] = dlist
                    dlist.append(did)
                recheck_info = 'following drawables might require recheck: \n'
                for tag, lst in recheck.items():
                    if tag is None:
                        tag = 'None'
                    recheck_info += tag + ':\n' + '\n'.join(lst) + '\n'
                create_info_dialog(recheck_info[:-1])
        else:
            create_info_dialog(f'No valid matching found!')


    def on_set_page_valid(self):
        valid = self.topArea.valid_checkbox.isChecked()
        self.canvas.push_undo_command(
            CommonCommand(
                redo_kwargs={'valid': valid},
                undo_kwargs={'valid': not valid},
                func=self.set_page_valid
            )
        )

    def on_set_page_incomplete(self):
        incomplete = self.topArea.incomplete_checkbox.isChecked()
        self.canvas.push_undo_command(
            CommonCommand(
                redo_kwargs={'incomplete': incomplete},
                undo_kwargs={'incomplete': not incomplete},
                func=self.set_page_incomplete
            )
        )

    def set_page_valid(self, valid):
        self.topArea.set_valid(valid)
        self.proj.is_current_page_valid = valid

    def set_page_incomplete(self, incomplete):
        self.topArea.set_incomplete(incomplete)
        self.proj.is_incomplete = incomplete

    def on_hide_tags(self, tag: str = None):
        if tag == '':
            tag = None
        if tag is None:
            tags = self.tagtree.get_selected_tags()
        else:
            tags = [tag]
        self.canvas.setTagsVisible(tags, False)
        self.tagtree.setTagCheckstate(tags, False)

    def on_show_tags(self, tag: str = None):
        if tag == '':
                tag = None
        if tag is None:
            tags = self.tagtree.get_selected_tags()
        else:
            tags = [tag]
        self.canvas.setTagsVisible(tags, True)
        self.tagtree.setTagCheckstate(tags, True)

    def on_hide_non_selected_tags(self):
        tags = self.tagtree.get_selected_tags(get_non_selected=True)
        self.canvas.setTagsVisible(tags, False)
        self.tagtree.setTagCheckstate(tags, False)
        tags = self.tagtree.get_selected_tags()
        if len(tags) > 0:
            self.canvas.setTagsVisible(tags, True)
            self.tagtree.setTagCheckstate(tags, True)

    def on_export_result(self):
        tmp_dir = osp.join(self.proj.directory, osp.basename(self.proj.directory) + '_cleaned')
        if osp.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        for page in self.proj.pages:
            srcp = osp.join(self.proj.directory, page, pcfg.parsing_src)
            if not osp.exists(srcp):
                continue
            saved = osp.join(tmp_dir, page)
            os.makedirs(saved, exist_ok=True)
            shutil.copy(srcp, saved)
        savep = osp.abspath(osp.join(self.proj.directory, osp.splitext(osp.basename(self.proj.proj_path))[0] + '.7z'))
        current_dir = os.getcwd()
        os.chdir(self.proj.directory)
        with py7zr.SevenZipFile(savep, 'w') as archive:
            archive.writeall(osp.basename(tmp_dir))
        os.chdir(current_dir)
        create_info_dialog(f'Archive exported to {savep}')
        shutil.rmtree(tmp_dir)

    def on_show_all_tags(self):
        tags = shared.cls_list + ['None']
        self.canvas.setTagsVisible(tags, True)
        self.tagtree.setTagCheckstate(tags, True)

    def on_candidates_menu(self, pos):
        item = self.candidates_list.itemAt(pos)
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu()
        act_apply = menu.addAction(self.tr('Apply as part'))
        act_tag = menu.addAction(self.tr('Set tag...'))
        act_edit = menu.addAction(self.tr('Edit mask...'))
        act_split = menu.addAction(self.tr('Split into parts'))
        act_layered = menu.addAction(self.tr('Export layered parts...'))
        act_del = menu.addAction(self.tr('Delete candidate'))
        act_export = menu.addAction(self.tr('Export cutout PNG'))
        act = menu.exec_(self.candidates_list.mapToGlobal(pos))
        if act == act_apply:
            self.apply_candidates()
        elif act == act_tag:
            self.on_candidate_tag_dialog(idx)
        elif act == act_edit:
            self.on_edit_mask(idx)
        elif act == act_split:
            self.on_split_instance(idx)
        elif act == act_layered:
            self.export_layered()
        elif act == act_del:
            self.on_candidate_delete(idx)
        elif act == act_export:
            self.export_candidate_cutout(idx)

    def on_show_candidates_toggled(self, show: bool):
        self.canvas.set_candidates_overlay(show)

    def _refresh_after_assembly(self):
        """Aggiorna modello/canvas/lista/tagtree dopo comandi che toccano le istanze."""
        self.proj.rebuild_applied_drawables()
        self.canvas.refreshDrawableItems()
        self.refresh_candidates()
        self._refresh_tagtree()

    def _refresh_tagtree(self):
        """Risincronizza la tag tree con le drawables correnti (dopo Apply/Edit/Split/Merge/layer ops)."""
        if not self.proj.model_valid:
            return
        dids, tag_list = self.proj.get_did_tag_pairs(pcfg.seg_type)
        ins_by_did = {f'inst://{self.proj.current_model}/{i.idx}': i
                      for i in self.proj.current_instance_list}
        visible_lst = [ins_by_did.get(d, None).visible if ins_by_did.get(d) else True for d in dids]
        name_lst = [ins_by_did.get(d).name if ins_by_did.get(d) else None for d in dids]
        self.tagtree.update_drawable_lst(dids, tag_list, visible_lst, name_lst)

    def _find_instance(self, idx):
        for i in self.proj.current_instance_list:
            if i.idx == idx:
                return i
        return None

    # ---------------- Fase 8: operazioni sui livelli ----------------
    def _instances_from_dids(self, dids):
        by_idx = {i.idx: i for i in self.proj.current_instance_list}
        out = []
        for did in dids:
            if not did or not did.startswith('inst://'):
                continue
            try:
                idx = int(did.rsplit('/', 1)[1])
            except (ValueError, IndexError):
                continue
            if idx in by_idx:
                out.append(by_idx[idx])
        return out

    def _selected_instances(self):
        """Istanza selezionate: dalla tag tree e/o dal canvas."""
        dids = list(self.tagtree.get_selected_drawable_ids())
        for it in self.canvas.selected_drawable_items():
            dids.append(it.drawable.did)
        return self._instances_from_dids(list(dict.fromkeys(dids)))

    def _after_layer_mutation(self):
        """Refresh completo dopo comandi layer (canvas + tree + salvataggio)."""
        self.canvas.refreshDrawableItems()
        self._refresh_tagtree()

    def on_delete_drawables(self):
        ins = self._selected_instances()
        if not ins:
            return
        self.canvas.push_undo_command(
            DeleteInstancesCommand(self.proj, ins, refresh_cb=self._after_layer_mutation))

    def on_duplicate_drawables(self):
        ins = self._selected_instances()
        if not ins:
            return
        self.canvas.push_undo_command(
            DuplicateInstancesCommand(self.proj, ins, refresh_cb=self._after_layer_mutation))

    def on_rename_drawable(self):
        ins = self._selected_instances()
        if not ins:
            return
        ins = ins[0]
        cur = ins.name or f'inst://{self.proj.current_model}/{ins.idx}'
        name, ok = QInputDialog.getText(self, self.tr('Rinomina layer'), self.tr('Nome layer:'), text=cur)
        if not ok or not name.strip():
            return
        self.canvas.push_undo_command(
            RenameInstanceCommand(self.proj, ins, name.strip(), refresh_cb=self._after_layer_mutation))

    def on_reorder_selected(self, delta):
        ins = self._selected_instances()
        if not ins:
            return
        for i in ins:
            self.canvas.push_undo_command(
                ReorderInstancesCommand(self.proj, i, delta, refresh_cb=self._after_layer_mutation))

    def on_drawable_visibility_changed(self, did, visible):
        ins = self._instances_from_dids([did])
        if not ins:
            return
        self.canvas.push_undo_command(
            SetInstancesVisibleCommand(self.proj, ins, visible, refresh_cb=self._after_layer_mutation))

    # ---------------- Fase 8: clipboard (copy/cut/paste) ----------------
    def copy_selected(self):
        ins = self._selected_instances()
        if not ins:
            return
        import copy as _copy
        self._clipboard = [(_copy.deepcopy(i), self.proj.current_model) for i in ins]
        self.on_run_status(self.tr(f'Copiati {len(ins)} layer negli appunti'))

    def cut_selected(self):
        self.copy_selected()
        self.on_delete_drawables()

    def paste_clipboard(self):
        if not getattr(self, '_clipboard', None):
            self.on_run_status(self.tr('Appunti vuoti: copia prima un layer (Ctrl+C)'))
            return
        templates = [c for c, _ in self._clipboard]
        self.canvas.push_undo_command(
            PasteInstancesCommand(self.proj, templates, refresh_cb=self._after_layer_mutation))
        self.on_run_status(self.tr(f'Incollati {len(templates)} layer'))

    def on_edit_mask(self, idx):
        ins = self._find_instance(idx)
        if ins is None:
            return
        img = self.proj.current_image
        if img is None:
            create_info_dialog('Immagine di pagina non disponibile.')
            return
        from .mask_editor import run_mask_editor
        ok, results = run_mask_editor(img, ins, self)
        if not ok or not results:
            return
        if len(results) == 1:
            m, b = results[0]
            self.canvas.push_undo_command(EditMaskCommand(self.proj, ins, m, b))
        else:
            pieces = [m for m, _ in results]
            self.canvas.push_undo_command(SplitInstanceCommand(self.proj, ins, pieces))
        self._refresh_after_assembly()

    def on_split_instance(self, idx):
        ins = self._find_instance(idx)
        if ins is None:
            return
        from .mask_ops import split_components
        pieces = split_components(ins.mask)
        if len(pieces) < 2:
            create_info_dialog('L\'istanza ha una sola componente connessa: niente da dividere.')
            return
        self.canvas.push_undo_command(SplitInstanceCommand(self.proj, ins, pieces))
        self._refresh_after_assembly()

    def on_merge_selected(self):
        items = self.candidates_list.selectedItems()
        if len(items) < 2:
            create_info_dialog('Seleziona almeno 2 candidati (Ctrl+click) e riprova.')
            return
        idxs = [it.data(Qt.ItemDataRole.UserRole) for it in items]
        ins_list = [i for i in self.proj.current_instance_list if i.idx in idxs]
        img = self.proj.current_image
        if img is None:
            return
        from .mask_ops import merge_masks
        H, W = img.shape[:2]
        m, b = merge_masks([i.mask for i in ins_list],
                           [i.bbox for i in ins_list], H, W)
        if not m.any():
            create_info_dialog('Maschere vuote: impossibile fondere.')
            return
        self.canvas.push_undo_command(MergeInstancesCommand(self.proj, ins_list, m, b))
        self._refresh_after_assembly()

    def on_candidate_tag_dialog(self, idx):
        vocab = get_all_segcls(pcfg.cls_path)
        items = sorted(vocab) if vocab else []
        items = ['unknown'] + items
        cur = 'unknown'
        for i in self.proj.current_instance_list:
            if i.idx == idx:
                cur = i.tag or 'unknown'
                break
        tag, ok = QInputDialog.getItem(self, self.tr('Set tag'),
                                       self.tr('Tag parte:'), items, 0, False)
        if ok and tag:
            self.retag_candidate(idx, tag)

    def apply_candidates(self):
        """Voting tag per overlap e promozione a Drawable (undo-able)."""
        instances = [i for i in self.proj.current_instance_list if not i.applied]
        if not instances:
            create_info_dialog('Nessun candidato da applicare (o gia\' applicati).')
            return
        if not self.proj.model_valid:
            create_info_dialog('Apri una pagina con un modello valido per il voting.')
            return
        from .assembly import vote_tags
        from .commands import CreateDrawablesCommand
        img = self.proj.current_image
        votes = vote_tags(self.proj.l2dmodel, instances, img,
                          min_iou=pcfg.assembly_min_iou)
        tags = [t for _, t, _, _ in votes]
        self.canvas.push_undo_command(
            CreateDrawablesCommand(self.proj, instances, tags,
                                   self.proj.current_model, img))
        self.canvas.updateCanvas()
        self.refresh_candidates()
        self._refresh_tagtree()

    def retag_candidate(self, idx, tag):
        from .commands import SetCandidateTagsCommand
        ins = None
        for i in self.proj.current_instance_list:
            if i.idx == idx:
                ins = i
                break
        if ins is None:
            return
        self.canvas.push_undo_command(
            SetCandidateTagsCommand(self.proj, [ins], [tag]))
        self.refresh_candidates()


    def export_candidate_cutout(self, idx):
        ins = None
        for i in self.proj.current_instance_list:
            if i.idx == idx:
                ins = i
                break
        if ins is None:
            return
        d = self.proj.instance_dir()
        savep = QFileDialog.getSaveFileName(self, self.tr('Save Cutout...'),
                                            osp.join(d, f'candidate_{idx}.png'), 'PNG (*.png)')
        if isinstance(savep, tuple):
            savep = savep[0]
        if not savep:
            return
        cutout = ins.get_cutout(self.proj.current_image)
        if cutout is None:
            create_info_dialog('Cutout non generabile (mask/immagine mancante).')
            return
        if not savep.lower().endswith('.png'):
            savep += '.png'
        Image.fromarray(cutout).save(savep)
        create_info_dialog(f'Cutout salvato in {savep}')

    def on_reveal_drawable(self):
        sel_dids = self.tagtree.get_selected_drawable_ids()
        if len(sel_dids) > 0:
            drawable = self.proj.l2dmodel.did2drawable.get(sel_dids[-1])
            if drawable is None or not drawable.src_path:
                return
            src_path = drawable.src_path
            if sys.platform == 'win32':
                # qprocess seems to fuck up with "\""
                p = "\""+str(Path(src_path))+"\""
                subprocess.Popen("explorer.exe /select,"+p, shell=True)
            elif sys.platform == 'darwin':
                p = "\""+src_path+"\""
                subprocess.Popen("open -R "+p, shell=True)

    def on_show_colormask(self, show):
        self.canvas.setShowTagMask(show)
        # pass

    # def on_recttool_state_changed(self):
    #     if self.leftBar.rectTool.isChecked():
    #         pcfg.edit_mode = EditMode.RectInference
    #         save_config()
    #     else:
    #         if pcfg.edit_mode == EditMode.RectInference:
    #             pcfg.edit_mode = EditMode.NONE
    #             self.canvas.rect_tool.hide()
    #             save_config()

    def on_tag_changed(self, tag, set_combobox=False):
        sel_dids = self.tagtree.get_selected_drawable_ids()
        if len(sel_dids) == 0:
            return
        sel_src_tags = [self.proj.l2dmodel.did2drawable[d].tag for d in sel_dids
                        if d in self.proj.l2dmodel.did2drawable]
        sel_tgt_tags = [tag] * len(sel_dids)
        self.canvas.push_undo_command(
            SetDrawableTagCommand(
            update_tag=lambda x: self.update_tag(dids=sel_dids, tag_list=x, set_combobox=set_combobox), src_tags=sel_src_tags, tgt_tags=sel_tgt_tags))

    def update_tag(self, dids=None, tag_list=None, set_combobox=False, did2tag=None):
        if did2tag is not None:
            dids = []
            tag_list = []
            for k, v in did2tag.items():
                dids.append(k)
                tag_list.append(v)
        else:
            assert dids is not None and tag_list is not None
        self.tagtree.update_tag(dids, tag_list)
        self.canvas.update_tag(dids, tag_list)
        tag = self.drawable_preview.tag
        self.drawable_preview.updateTag(tag, self.canvas.get_tagitem(tag).img)
        if set_combobox and len(set(tag_list)) == 1:
            self.topArea.cls_list_combobox.blockSignals(True)
            self.topArea.cls_list_combobox.setCurrentText(tag)
            self.topArea.cls_list_combobox.blockSignals(False)

    def on_show_colormsk(self):
        show = self.topArea.show_colormsk_checkbox.isChecked()
        self.canvas.setShowTagMask(show)
        if pcfg.show_colorcode != show:
            pcfg.show_colorcode = show
            save_config()

    def on_show_contour(self):
        show = self.topArea.show_contour_checkbox.isChecked()
        self.canvas.setShowContour(show)
        if pcfg.show_contour != show:
            pcfg.show_contour = show
            save_config()

    def on_set_colormsk_opacity(self, *args, **kwargs):
        value = int(self.topArea.mask_opacity_box.value())
        self.canvas.setColormskOpacity(value)
        if pcfg.mask_opacity != value:
            pcfg.mask_opacity = value
            save_config()

    def setupThread(self):
        self.proj_save_thred = ProjSaveThread()
        self.proj_save_thred.progress.connect(self.on_update_proj_progress)
        self.proj_save_thred.early_stop_signal.connect(self.on_projsave_early_stopped)
        self.proj_save_thred.export_sucess.connect(self.on_export_sucess)
        self.run_thread = SegmentationThread()
        self.run_thread.page_finished.connect(self.on_page_finished)
        self.run_thread.manual_inference_finished.connect(self.manual_inference_finished)
        self.run_thread.early_stop_signal.connect(self.on_runthread_early_stopped)
    
    def on_edit_scale_changed(self):
        value = self.bottomBar.scaleEditor.get_value() / 100
        self.canvas.scaleImage(value, emit_changed=False, scale_to=True)

    def on_canvas_scale_changed(self):
        self.bottomBar.scaleEditor.set_value(int(self.canvas.scale_factor * 100))

    def on_canvas_selection_changed(self, did, selected):
        # self.tagtree.update_selections(sel_ids)
        self.tagtree.update_seleciton(did, selected)
        if selected:
            sels = [item for item in self.canvas.selectedItems() if isinstance(item, DrawableItem)]
            if sels[-1].drawable.did == did:
                d = sels[-1].drawable
                self.drawable_preview.updateDrawable(d, self.canvas.get_tagitem(d.tag).img)
                self.topArea.set_tag(d.tag)

    def on_tag_selection_changed(self, tag):
        clear_preview_mask = len(self.canvas.selected_drawable_items()) == 0
        self.drawable_preview.updateTag(tag, self.canvas.get_tagitem(tag).img, clear_preview_mask)

    def on_tagtree_selection_changed(self, did, selected, is_last):
        self.canvas.update_drawable_selection(did, selected, ensure_visible=is_last)
        if selected and is_last:
            d = self.proj.l2dmodel.did2drawable.get(did)
            if d is not None:
                self.drawable_preview.updateDrawable(d, self.canvas.get_tagitem(d.tag).img)

    def on_export_sucess(self, msg: str):
        create_info_dialog(msg)

    def on_projsave_early_stopped(self, msg: str):
        LOGGER.debug(msg)
        self.bottomBar.progress_bar.updateProgress(0)

    def on_runthread_early_stopped(self, msg: str):
        LOGGER.debug(msg)
        self.bottomBar.progress_bar.updateProgress(0)
        self.bottomBar.progress_bar.hide()
        self.topArea.set_running(False)
        self.topArea.set_status(msg)
        if 'interrotto' not in msg.lower():
            create_info_dialog(msg)

    def on_run_status(self, msg: str):
        self.topArea.set_status(msg)
    
    def on_update_proj_progress(self, progress):
        if self.topArea.isVisible():
            self.bottomBar.progress_bar.updateProgress(progress)
        if progress == 100:
            self.bottomBar.progress_bar.hide()

    def manual_inference_finished(self, instances):
        """Nuove istanze prodotte da un prompt box/point (undo-able)."""
        instances = list(instances or [])
        if not instances:
            self.topArea.set_running(False)
            self.topArea.set_status('Nessun candidato prodotto — prova con un box/point più grande.')
            return
        self.canvas.push_undo_command(AddInstancesCommand(self.proj, instances))
        self.refresh_candidates()
        self.topArea.set_status(
            f'+{len(instances)} candidato/i — selezionalo e premi \'Apply as parts\' per creare la parte.')

    # ---------- Fase 2: run batch / box / point ----------
    def on_run_progress(self, value: int):
        self.bottomBar.progress_bar.updateProgress(value)
        if value >= 100:
            self.bottomBar.progress_bar.hide()
            self.topArea.set_running(False)
            self.topArea.set_status('Batch completato — apri ogni pagina e premi \'Apply as parts\'.')

    def on_run_requested(self, running: bool):
        if running:
            if not self.proj.model_valid:
                create_info_dialog('Apri prima un progetto con almeno una pagina.')
                self.topArea.set_running(False)
                return
            try:
                provider = get_provider(pcfg.inference_provider,
                                        device=pcfg.segmentation_device)
            except Exception as e:  # noqa: BLE001
                create_error_dialog(e, 'Backend di inferenza non disponibile')
                self.topArea.set_running(False)
                return
            self.topArea.set_status(
                f'Avvio segmentazione con \'{pcfg.inference_provider}\'…')
            self.run_thread.runSegmentation(self.proj, provider)
            self.topArea.set_running(True)
            self.bottomBar.progress_bar.show()
            self.bottomBar.progress_bar.updateProgress(0)
        else:
            self.run_thread.set_stop_flag()
            self.topArea.set_running(False)

    def on_prompt_tool(self, mode: str):
        if mode == 'box':
            pcfg.edit_mode = EditMode.RectInference
            self.canvas.set_point_prompt_mode(False)
        elif mode == 'point':
            pcfg.edit_mode = EditMode.PointInference
            self.canvas.set_point_prompt_mode(True)
        else:
            self.canvas.set_point_prompt_mode(False)
            pcfg.edit_mode = EditMode.NONE

    def on_point_prompted(self, points, labels):
        if not self.proj.model_valid:
            return
        try:
            provider = get_provider(pcfg.inference_provider,
                                    device=pcfg.segmentation_device)
        except Exception as e:  # noqa: BLE001
            create_error_dialog(e, 'Provider non disponibile')
            return
        self.run_thread.runSegmentation(self.proj, provider,
                                        points=points, labels=labels)

    def on_end_create_rect(self, rect, idx):
        if pcfg.edit_mode != EditMode.RectInference:
            return
        box = [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
        try:
            provider = get_provider(pcfg.inference_provider,
                                    device=pcfg.segmentation_device)
        except Exception as e:  # noqa: BLE001
            create_error_dialog(e, 'Provider non disponibile')
            return
        self.run_thread.runSegmentation(self.proj, provider, boxes=[box])

    def on_settings_requested(self):
        from .settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        dlg.exec_()

    def on_anim_requested(self):
        """Apre il dialogo Cycle Animation sulle parti della pagina corrente."""
        from .anim_dialog import AnimationDialog
        dlg = AnimationDialog(self.proj, self)
        dlg.exec_()

    # ---------- Pannello candidati ----------
    def refresh_candidates(self):
        self.candidates_list.blockSignals(True)
        self.candidates_list.clear()
        for ins in self.proj.current_instance_list:
            x, y, w, h = ins.bbox
            tag = ins.tag if ins.tag else 'unknown'
            mark = f'\u2713 {tag}' if ins.applied else f'[{tag}]'
            it = QListWidgetItem(
                f'#{ins.idx}  {mark}  {ins.score:.2f}  [{x},{y} {w}x{h}]')
            it.setData(Qt.ItemDataRole.UserRole, ins.idx)
            self.candidates_list.addItem(it)
        self.candidates_list.blockSignals(False)
        if self.show_cand_check.isChecked():
            self.canvas.refresh_candidate_overlay()

    def on_candidate_activate(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        for ins in self.proj.current_instance_list:
            if ins.idx == idx:
                x, y, w, h = ins.bbox
                self.canvas.gv.ensureVisible(QRectF(x, y, w, h))
                break

    def on_candidate_delete(self, idx):
        cur = self.proj.current_instance_list
        removed = [i for i in cur if i.idx == idx]
        self.proj._cur_instances = [i for i in cur if i.idx != idx]
        self.proj.save_current_instances()
        if removed and removed[0].applied and self.proj.model_valid:
            # la drawable applicata va rimossa insieme all'istanza
            did = f'inst://{self.proj.current_model}/{idx}'
            self.proj.l2dmodel.drawables = [
                d for d in self.proj.l2dmodel.drawables if d.did != did]
            self.canvas.refreshDrawableItems()
            self._refresh_tagtree()
        self.canvas.setProjSaveState(True)
        self.refresh_candidates()

    def on_page_finished(self, page_index: int):
        if page_index != self.pageList.currentIndex().row():
            self.pageList.setCurrentRow(page_index)
        else:
            self.proj.set_current_page_byidx(page_index)
            self.canvas.updateCanvas()
        n_cand = len(self.proj.current_instance_list)
        self.topArea.set_status(
            f'Pagina {page_index + 1}/{self.proj.num_pages} segmentata — {n_cand} candidato/i.')
        if self.proj.num_pages > 0:
            if page_index + 1 == self.proj.num_pages:
                self.bottomBar.progress_bar.hide()
            else:
                progress = (page_index + 1) / self.proj.num_pages * 100
                progress = int(round(progress))
                self.bottomBar.progress_bar.updateProgress(progress)


    def setupUI(self):
        screen_size = QGuiApplication.primaryScreen().geometry().size()
        self.setMinimumWidth(screen_size.width() // 2)

        self.leftBar = LeftBar(self)
        self.leftBar.showPageListLabel.clicked.connect(self.updatePageListVisibility)
        self.leftBar.open_proj.connect(self.openProj)

        self.pageList = PageListView()
        self.pageList.reveal_file.connect(self.on_reveal_file)
        self.pageList.currentItemChanged.connect(self.pageListCurrentItemChanged)
        
        self.centralStackWidget = QStackedWidget(self)
        
        self.titleBar = TitleBar(self, proj=self.proj)
        self.titleBar.closebtn_clicked.connect(self.on_closebtn_clicked)
        # self.titleBar.display_lang_changed.connect(self.on_display_lang_changed)

        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        mainHLayout.addWidget(self.centralStackWidget)
        mainHLayout.setContentsMargins(0, 0, 0, 0)
        mainHLayout.setSpacing(0)

        # set up canvas
        SW.canvas = self.canvas = Canvas()
        self.canvas.proj = self.proj
        self.canvas.gv.hide_canvas.connect(self.onHideCanvas)
        self.canvas.proj_savestate_changed.connect(self.on_savestate_changed)
        self.canvas.drop_open_folder.connect(self.dropOpenDir)

        self.topArea = TopArea(parent=self)
        # self.topArea.run_btn.clicked.connect(self.on_runbtn_clicked)
        # self.run_thread.finished.connect(self.topArea.run_btn.setRunState)

        self.centerWidget = Widget()
        centerLayout = QVBoxLayout(self.centerWidget)
        centerLayout.addWidget(self.topArea)
        centerLayout.addWidget(self.canvas.gv)
        centerLayout.setContentsMargins(0, 0, 0, 0)
        centerLayout.setSpacing(0)

        self.rightWidget = QSplitter(Qt.Orientation.Vertical)
        self.candidates_list = QListWidget()
        self.candidates_list.setMinimumHeight(120)
        self.candidates_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.candidates_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.candidates_list.customContextMenuRequested.connect(self.on_candidates_menu)
        self.apply_candidates_btn = QPushButton(self.tr('Apply as parts'))
        self.apply_candidates_btn.setToolTip(
            self.tr('Vota i tag per overlap e crea le parti editabili (undo-able)'))
        self.merge_sel_btn = QPushButton(self.tr('Merge sel.'))
        self.merge_sel_btn.setToolTip(self.tr('Fonde i candidati selezionati (Ctrl+click)'))
        self.show_cand_check = QCheckBox(self.tr('Show candidates'))
        self.show_cand_check.setChecked(True)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.apply_candidates_btn)
        btn_row.addWidget(self.merge_sel_btn)
        btn_row.addWidget(self.show_cand_check)
        cand_header = QWidget()
        cand_lo = QVBoxLayout(cand_header)
        cand_lo.setContentsMargins(4, 2, 4, 2)
        cand_lo.addLayout(btn_row)
        cand_lo.addWidget(self.candidates_list)
        self.tagtree = TagTree(parent=self)
        self.drawable_preview = DrawablePreview(parent=self)
        self.rightWidget.addWidget(cand_header)
        self.rightWidget.addWidget(self.tagtree)
        self.rightWidget.addWidget(self.drawable_preview)
        
        self.rightWidget.setStretchFactor(0, 3)
        self.rightWidget.setStretchFactor(1, 7)
        self.rightWidget.setStretchFactor(2, 1)
        # rightLayout = QVBoxLayout(self.rightWidget)
        # rightLayout.addWidget(self.tagtree)
        # rightLayout.adds
        # rightLayout.addWidget(self.drawable_preview)

        self.mainWindowSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.mainWindowSplitter.addWidget(self.pageList)
        self.mainWindowSplitter.addWidget(self.centerWidget)
        self.mainWindowSplitter.addWidget(self.rightWidget)
        # self.mainWindowSplitter.addWidget(self.rightComicTransStackPanel)

        self.centralStackWidget.addWidget(self.mainWindowSplitter)

        self.bottomBar = BottomBar(self)

        mainVBoxLayout = QVBoxLayout(self)
        mainVBoxLayout.addWidget(self.titleBar)
        mainVBoxLayout.addLayout(mainHLayout)
        mainVBoxLayout.addWidget(self.bottomBar)
        mainVBoxLayout.setContentsMargins(0, 0, 0, 0)
        mainVBoxLayout.setSpacing(0)

        self.mainvlayout = mainVBoxLayout
        self.mainWindowSplitter.setStretchFactor(0, 1)
        self.mainWindowSplitter.setStretchFactor(1, 7)
        self.mainWindowSplitter.setStretchFactor(2, 3)
        self.resetStyleSheet()

    def setupConfig(self):
        self.leftBar.initRecentProjMenu(pcfg.recent_proj_list)
        self.leftBar.showPageListLabel.setChecked(pcfg.show_page_list)
        # if pcfg.edit_mode == EditMode.RectInference:
        #     self.leftBar.rectTool.setChecked(True)
        if not pcfg.show_page_list:
            self.pageList.setHidden(True)
        self.titleBar.darkModeAction.setChecked(pcfg.darkmode)
        self.topArea.setupConfig(pcfg)
        shared.cls_list = get_all_segcls(pcfg.cls_path)
        self.update_cls_list()

        self.topArea.show_colormsk_checkbox.blockSignals(True)
        self.topArea.show_colormsk_checkbox.setChecked(pcfg.show_colorcode)
        self.topArea.show_colormsk_checkbox.blockSignals(False)
        self.canvas.setShowTagMask(pcfg.show_colorcode)

        self.topArea.show_contour_checkbox.blockSignals(True)
        self.topArea.show_contour_checkbox.setChecked(pcfg.show_contour)
        self.topArea.show_contour_checkbox.blockSignals(False)
        self.canvas.setShowContour(pcfg.show_contour)

        self.topArea.mask_opacity_box.blockSignals(True)
        self.topArea.mask_opacity_box.setValue(pcfg.mask_opacity)
        self.topArea.mask_opacity_box.blockSignals(False)
        self.canvas.setColormskOpacity(pcfg.mask_opacity)


    def update_cls_list(self):
        self.tagtree.update_cls_list(shared.cls_list)
        self.canvas.update_cls_list(shared.cls_list)
        self.topArea.update_cls_list(shared.cls_list)

    def setupShortcuts(self):
        self.titleBar.darkmode_trigger.connect(self.on_darkmode_triggered)
        self.leftBar.save_proj.connect(self.conditional_manual_save)

        shortcutA = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        shortcutA.activated.connect(self.shortcutBefore)
        shortcutPageUp = QShortcut(QKeySequence(QKEY.Key_Left), self)
        shortcutPageUp.activated.connect(self.shortcutBefore)

        shortcutD = QShortcut(QKeySequence("D"), self)
        shortcutD.activated.connect(self.shortcutNext)
        shortcutPageDown = QShortcut(QKeySequence(QKEY.Key_Right), self)
        shortcutPageDown.activated.connect(self.shortcutNext)

        # Fase 2: tool box/point prompt
        shortcutBox = QShortcut(QKeySequence("W"), self)
        shortcutBox.activated.connect(self.shortcutBoxTool)
        shortcutPoint = QShortcut(QKeySequence("P"), self)
        shortcutPoint.activated.connect(self.shortcutPointTool)

        # Fase 8: clipboard + delete globali (Ctrl+C/V/X, Canc, Ctrl+D)
        sc_copy = QShortcut(QKeySequence.StandardKey.Copy, self)
        sc_copy.activated.connect(self.copy_selected)
        sc_cut = QShortcut(QKeySequence.StandardKey.Cut, self)
        sc_cut.activated.connect(self.cut_selected)
        sc_paste = QShortcut(QKeySequence.StandardKey.Paste, self)
        sc_paste.activated.connect(self.paste_clipboard)
        sc_del = QShortcut(QKeySequence.StandardKey.Delete, self)
        sc_del.activated.connect(self.on_delete_drawables)
        sc_dup = QShortcut(QKeySequence('Ctrl+D'), self)
        sc_dup.activated.connect(self.on_duplicate_drawables)
        sc_save = QShortcut(QKeySequence.StandardKey.Save, self)
        sc_save.activated.connect(self.conditional_manual_save)
        sc_selall = QShortcut(QKeySequence.StandardKey.SelectAll, self)
        sc_selall.activated.connect(self.canvas.selectAll)
        self._clipboard = []

    def shortcutBoxTool(self):
        self.topArea.box_tool_check.setChecked(not self.topArea.box_tool_check.isChecked())

    def shortcutPointTool(self):
        self.topArea.point_tool_check.setChecked(not self.topArea.point_tool_check.isChecked())

        # shortcutRectTool = QShortcut(QKeySequence("W"), self)
        # shortcutRectTool.activated.connect(self.shortcutRectTool)

    # def shortcutRectTool(self):
    #     self.leftBar.rectTool.setChecked(not self.leftBar.rectTool.isChecked())

    def shortcutNext(self):
        index = self.pageList.currentIndex()
        page_count = self.pageList.count()
        if index.isValid():
            row = index.row()
            row = (row + 1) % page_count
            self.pageList.setCurrentRow(row)

    def shortcutBefore(self):
        index = self.pageList.currentIndex()
        page_count = self.pageList.count()
        if index.isValid():
            row = index.row()
            row = (row - 1 + page_count) % page_count
            self.pageList.setCurrentRow(row)

    def shortcutSelectAll(self):
        self.canvas.selectAll()

    def updatePageListVisibility(self):
        show = self.leftBar.showPageListLabel.isChecked()
        if show:
            if self.pageList.isHidden():
                self.pageList.show()
        else:
            self.pageList.hide()
        pcfg.show_page_list = show
        save_config()

    def resetStyleSheet(self, reverse_icon: bool = False):
        theme = 'eva-dark' if pcfg.darkmode else 'eva-light'
        self.setStyleSheet(parse_stylesheet(theme, reverse_icon))

    def on_closebtn_clicked(self):
        if self.proj_save_thred.isRunning():
            self.proj_save_thred.finished.connect(self.close)
            mb = FrameLessMessageBox()
            mb.setText(self.tr('Waiting for saving process to be completed...'))
            self.proj_save_thred.finished.connect(mb.close)
            mb.exec()
            return
        self.close()

    def on_reveal_file(self):
        current_model_path = self.proj.current_model_path()
        if sys.platform == 'win32':
            # qprocess seems to fuck up with "\""
            p = "\""+str(Path(current_model_path))+"\""
            subprocess.Popen("explorer.exe /select,"+p, shell=True)
        elif sys.platform == 'darwin':
            p = "\""+current_model_path+"\""
            subprocess.Popen("open -R "+p, shell=True)

    def onHideCanvas(self):
        pass
    
    def on_savestate_changed(self, unsaved: bool):
        save_state = self.tr('unsaved') if unsaved else self.tr('saved')
        self.titleBar.setTitleContent(save_state=save_state)

    def dropOpenDir(self, directory: str):
        if isinstance(directory, str) and osp.exists(directory):
            self.openProj(directory)

    def openProj(self, p: str):
        try:
            self.opening_dir = True
            self.proj.load(p)
            self.canvas.clearDrawableItems()
            self.titleBar.setTitleContent(osp.basename(p))
            self.updatePageList()
            self.opening_dir = False
            self.leftBar.updateRecentProjList(self.proj.proj_path)
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project ') + p)
            return

    def conditional_manual_save(self):
        if not self.opening_dir:
            self.saveCurrentPage()

    def saveCurrentPage(self):
        self.proj.save_current_page()
        self.canvas.setProjSaveState(False)
        self.canvas.update_saved_undostep()

    def pageListCurrentItemChanged(self):
        item = self.pageList.currentItem()
        self.page_changing = True
        if item is not None:
            if self.save_on_page_changed:
                self.conditional_manual_save()
            self.proj.set_current_page(item.text())
            self.topArea.set_valid(self.proj.is_current_page_valid)
            self.topArea.set_incomplete(self.proj.is_incomplete)
            self.canvas.clear_undostack()
            self.canvas.updateCanvas()
            dids, tag_list = self.proj.get_did_tag_pairs(pcfg.seg_type)
            self.tagtree.update_drawable_lst(dids, tag_list)
            tag = self.drawable_preview.tag
            self.drawable_preview.updateTag(tag, tag_img=self.canvas.get_tagitem(tag).img, clear_preview_mask=True)
            self.titleBar.setTitleContent(page_name=self.proj.current_model)
            self.refresh_candidates()
            # self.module_manager.handle_page_changed()
        self.page_changing = False

    def update_parsing_list(self):
        self.proj.set_current_page

    def updatePageList(self):
        if self.pageList.count() != 0:
            self.pageList.clear()
        if len(self.proj.pages) >= shared.PAGELIST_THUMBNAIL_MAXNUM:
            item_func = lambda imgname: QListWidgetItem(imgname)
        else:
            item_func = lambda imgname:\
                QListWidgetItem(QIcon(osp.join(self.proj.directory, imgname)), imgname)
        for imgname in self.proj.pages:
            lstitem =  item_func(imgname)
            self.pageList.addItem(lstitem)
            if imgname == self.proj.current_model:
                self.pageList.setCurrentItem(lstitem)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.conditional_manual_save()
        self.canvas.prepareClose()
        save_config()
        if not self.proj.is_empty:
            self.proj.save()
        return super().closeEvent(event)
    
    def on_darkmode_triggered(self):
        pcfg.darkmode = self.titleBar.darkModeAction.isChecked()
        from .misc import apply_theme_palette
        apply_theme_palette(self.app, dark=pcfg.darkmode)
        self.resetStyleSheet(reverse_icon=True)
        save_config()

    def export_cutout(self, export_mask=False):
        """Esporta i cutout dei drawable selezionati (PNG full-canvas con alpha)."""
        if not self.proj.model_valid:
            create_info_dialog('Apri un modello con parti da esportare.')
            return
        items = self.canvas.selected_drawable_items()
        if not items:
            create_info_dialog('Seleziona almeno una parte sul canvas (click).')
            return
        d = self.proj.instance_dir()
        H, W = self.proj.current_image.shape[:2]
        for item in items:
            dr = item.drawable
            tag = dr.tag or 'unknown'
            if export_mask:
                crop = np.asarray(dr.visible_mask, dtype=np.uint8) * 255
                full = np.zeros((H, W), dtype=np.uint8)
            else:
                crop = np.asarray(dr.get_img(), dtype=np.uint8)
                full = np.zeros((H, W, 4), dtype=np.uint8)
            ch, cw = crop.shape[:2]
            x, y = int(dr.x), int(dr.y)
            x2, y2 = min(x + cw, W), min(y + ch, H)
            if x2 > x and y2 > y:
                if export_mask:
                    full[y:y2, x:x2] = crop[:y2 - y, :x2 - x]
                else:
                    full[y:y2, x:x2] = crop[:y2 - y, :x2 - x]
            savep = QFileDialog.getSaveFileName(
                self, self.tr('Save Cutout...'),
                osp.join(d, f'{tag}__{Path(dr.did or str(dr.idx)).name}.png'),
                'PNG (*.png)')
            if isinstance(savep, tuple):
                savep = savep[0]
            if not savep:
                continue
            if not savep.lower().endswith('.png'):
                savep += '.png'
            Image.fromarray(full).save(savep)
        create_info_dialog('Cutout esportati.')

    def export_layered(self):
        """Esporta tutte le parti (originali + applicate) come PNG separati
        + manifest.json + composite.png, per l'uso a valle (atlas/training)."""
        if not self.proj.model_valid:
            create_info_dialog('Apri un modello con parti da esportare.')
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, self.tr('Export layered parts...'),
            self.proj.instance_dir())
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        H, W = self.proj.current_image.shape[:2]
        manifest = []
        for order, dr in enumerate(self.proj.l2dmodel.valid_drawables()):
            tag = dr.tag or 'unknown'
            fname = f'{order:03d}__{tag}.png'
            try:
                crop = np.asarray(dr.get_img(), dtype=np.uint8)
            except Exception as e:  # noqa: BLE001
                LOGGER.warning(f'export layered: skip {dr.did}: {e}')
                continue
            full = np.zeros((H, W, 4), dtype=np.uint8)
            ch, cw = crop.shape[:2]
            x, y = int(dr.x), int(dr.y)
            x2, y2 = min(x + cw, W), min(y + ch, H)
            if x2 > x and y2 > y:
                full[y:y2, x:x2] = crop[:y2 - y, :x2 - x]
            Image.fromarray(full).save(osp.join(out_dir, fname))
            manifest.append({
                'file': fname, 'did': dr.did, 'tag': tag,
                'idx': dr.idx, 'draw_order': order,
                'bbox': [x, y, cw, ch],
                'area': int(np.count_nonzero(full[..., 3]))
            })
        # composita: parti applicate sopra le originali, in draw_order
        comp = np.zeros((H, W, 4), dtype=np.uint8)
        for dr in self.proj.l2dmodel.valid_drawables():
            try:
                crop = np.asarray(dr.get_img(), dtype=np.uint8)
            except Exception:  # noqa: BLE001
                continue
            ch, cw = crop.shape[:2]
            x, y = int(dr.x), int(dr.y)
            x2, y2 = min(x + cw, W), min(y + ch, H)
            if x2 > x and y2 > y:
                roi = crop[:y2 - y, :x2 - x]
                a = roi[..., 3:4].astype(np.float32) / 255.0
                dst = comp[y:y2, x:x2]
                dst[..., :3] = (roi[..., :3] * a + dst[..., :3] * (1 - a)).astype(np.uint8)
                dst[..., 3] = np.maximum(dst[..., 3], roi[..., 3])
        Image.fromarray(comp).save(osp.join(out_dir, 'composite.png'))
        with open(osp.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump({'image': f'{self.proj.current_model}', 'parts': manifest,
                       'size': [W, H]}, f, indent=2, ensure_ascii=False)
        create_info_dialog(f'Export layered completato: {out_dir}\n'
                           f'{len(manifest)} parti + manifest.json + composite.png')

    def on_create_errdialog(self, error_msg: str, detail_traceback: str = '', exception_type: str = ''):
        try:
            if exception_type != '':
                shared.showed_exception.add(exception_type)
            err = QMessageBox()
            err.setText(error_msg)
            err.setDetailedText(detail_traceback)
            err.exec()
            if exception_type != '':
                shared.showed_exception.remove(exception_type)
        except:
            if exception_type in shared.showed_exception:
                shared.showed_exception.remove(exception_type)
            LOGGER.error('Failed to create error dialog')
            LOGGER.error(traceback.format_exc())

    def on_create_infodialog(self, info_dict: dict, info_type: str = None):
        if info_type is not None:
            if info_type in shared.info_widget_set:
                return
            if info_type == 'DownloadModel':
                info_dict['info_msg'] = self.download_model_msg
        
        dialog = MessageBox(**info_dict)
        shared.add_to_info_widget_set(dialog, info_type=info_type)
        dialog.show()   # exec_ will block main thread

    def on_close_infodialog(self, info_type: str):
        if info_type is not None and info_type != '' and info_type in shared.info_widget_set:
            w = shared.info_widget_set.pop(info_type)
            w.close()