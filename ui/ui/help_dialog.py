# -*- coding: utf-8 -*-
"""Dialogo Aiuto: scorciatoie da tastiera + mini-guida del flusso di lavoro."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                            QPushButton, QTextBrowser, QFrame)


_SHORTCUTS = [
    ('Navigazione', [
        ('← / →', 'Pagina precedente / successiva'),
        ('W', 'Tool box (segmentazione)'),
        ('P', 'Tool point (segmentazione)'),
    ]),
    ('Modifica layer', [
        ('Ctrl+C / Ctrl+X / Ctrl+V', 'Copia / taglia / incolla parti selezionate'),
        ('Canc / Backspace', 'Elimina parte selezionata'),
        ('Ctrl+D', 'Duplica parte'),
        ('F2', 'Rinomina parte'),
        ('H', 'Nascondi / mostra parte'),
        ('Ctrl+Z / Ctrl+Y', 'Annulla / Ripeti'),
        ('Ctrl+A', 'Seleziona tutto sul canvas'),
    ]),
    ('File', [
        ('Ctrl+S', 'Salva pagina corrente'),
        ('F1', 'Questo aiuto'),
    ]),
]

_FLOW = (
    'Flusso di lavoro:\n'
    '1. Apri un progetto (cartella con final.png per personaggio).\n'
    '2. Run: segmenta la pagina con SAM (candidati in basso).\n'
    '3. Applica i candidati: Apply as parts (crea le parti come layer).\n'
    '4. Correggi: seleziona una parte e usa Modifica maschera (pennello, '
    'gomma, split/merge) — tutto undo-able.\n'
    '5. Tagga le parti (testa, torso, braccia, gambe…) dal menu contestuale '
    'o dal combo Tag.\n'
    '6. Anim: apri il dialogo Cycle Animation per idle/walk/run; i parametri '
    'vengono salvati in anim.json e ripristinati.\n'
    '7. Esporta: sprite atlas animato + manifest.json (formato sprite-gen), '
    'layer PNG o PSD (un layer per parte + sfondo).\n'
)


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Aiuto — Scorciatoie e guida')
        self.setMinimumSize(520, 560)
        lay = QVBoxLayout(self)

        title = QLabel('<b>Studio See-Through — guida rapida</b>', self)
        lay.addWidget(title)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        html = ['<h3>Flusso di lavoro</h3>',
                '<pre>' + _FLOW + '</pre>']
        for section, rows in _SHORTCUTS:
            html.append(f'<h3>{section}</h3><table cellpadding="4">')
            for k, d in rows:
                html.append(f'<tr><td><b>{k}</b></td><td>{d}</td></tr>')
            html.append('</table>')
        browser.setHtml('<br>'.join(html))
        lay.addWidget(browser, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton('Chiudi', self)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)


def show_help(parent=None):
    HelpDialog(parent).exec_()
