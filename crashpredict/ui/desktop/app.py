"""Application desktop PySide6 — 5 modules + backtesting + rapport.

Même logique métier que la version Streamlit. Les graphiques sont rendus via
matplotlib (optionnel) ; sans matplotlib, seules les tables sont affichées.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ... import db
from ...config import CONFIG
from ...prediction import make_prediction, strategy_table
from ...prediction import current_analysis
from ...stats import dashboard_stats, prediction_history, performance_by_target
from ...stats.dashboard import last_metrics_run
from ...validation import close_pending_predictions
from ..theme import pyside_stylesheet
from .widgets import Badge, KpiCard, card_row, heading, muted, table

PAGES = ["Dashboard", "Prédiction", "Historique", "Entraînement", "Données", "Rapport"]


class TrainWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, targets, models, initial_train, step, max_folds):
        super().__init__()
        self.params = dict(targets=targets, models=models, initial_train=initial_train,
                           step=step, max_folds=max_folds)

    def run(self) -> None:
        try:
            from ...training import run_training

            summary = run_training(**self.params)
            self.finished_ok.emit(summary)
        except Exception as exc:  # pragma: no cover - UI
            self.failed.emit(repr(exc))


class DashboardPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.addWidget(heading("Dashboard"))
        analysis = current_analysis()
        stats = dashboard_stats(backtest=True)
        if analysis.get("error"):
            lay.addWidget(muted("Base vide : importez des données (onglet Données)."))
            return
        lay.addWidget(
            card_row([
                KpiCard("Manches vérifiées", f"{analysis['n_rounds']:,}".replace(",", " "),
                        f"dernière : {analysis['last_multiplier']:.2f}x"),
                KpiCard("Prédictions évaluées", f"{stats['n_predictions']:,}".replace(",", " "),
                        f"{stats['n_success']} succès / {stats['n_failure']} échecs"),
                KpiCard("Taux de réussite", f"{(stats.get('success_rate') or 0)*100:.1f} %",
                        f"modèle : {stats.get('deployed_model') or 'n/d'}", "#00B894"),
                KpiCard("Cible optimisée (80 %)", f"{(analysis['optimized'].get('target') or 0):.2f}x",
                        f"probabilité {(analysis['optimized'].get('probability') or 0)*100:.1f} %", "#FDCB6E"),
            ])
        )
        lay.addWidget(heading("Derniers coefficients", 15))
        recent = db.get_rounds(limit=25, ascending=False)
        lay.addWidget(table(["Manche", "Coefficient", "Horodatage", "Origine"],
                            [[r["round_id"], f"{r['multiplier']:.2f}x", r["ts"], r["ts_source"]] for r in recent]))
        lay.addWidget(heading("Stratégies", 15))
        lay.addWidget(table(["Stratégie", "Cible", "Probabilité", "Niveau"],
                            [[s["stratégie"], f"{s['cible']:.2f}x", f"{s['probabilité']*100:.1f} %", s["niveau"]]
                             for s in strategy_table()], height=220))
        lay.addStretch(1)


class PredictionPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.addWidget(heading("Prédiction"))
        lay.addWidget(muted(
            "Le robot propose un coefficient cible et sa probabilité de validation sur les 3 manches suivantes. "
            "La probabilité est une fréquence historique, pas une garantie."
        ))
        row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["Cible fixe", "Taux de réussite visé"])
        self.value = QDoubleSpinBox()
        self.value.setRange(1.01, 1000.0)
        self.value.setSingleStep(0.05)
        self.value.setValue(2.0)
        self.mode.currentIndexChanged.connect(self._sync_mode)
        btn = QPushButton("Lancer l'analyse")
        btn.clicked.connect(self.run_prediction)
        row.addWidget(QLabel("Stratégie"))
        row.addWidget(self.mode)
        row.addWidget(QLabel("Valeur"))
        row.addWidget(self.value)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        lay.addWidget(self.output, 1)
        self._sync_mode()

    def _sync_mode(self) -> None:
        if self.mode.currentText() == "Cible fixe":
            self.value.setRange(1.01, 1000.0)
            self.value.setValue(2.0)
        else:
            self.value.setRange(0.5, 0.99)
            self.value.setSingleStep(0.01)
            self.value.setValue(0.8)

    def run_prediction(self) -> None:
        close_pending_predictions()
        try:
            if self.mode.currentText() == "Cible fixe":
                pred = make_prediction(target=self.value.value())
            else:
                pred = make_prediction(desired_rate=self.value.value())
        except RuntimeError as exc:
            QMessageBox.warning(self, "Prédiction impossible", str(exc))
            return
        loi = pred["explanation"]["loi_marginale"]
        self.output.setPlainText(
            f"Identifiant       : {pred['id']}\n"
            f"Créée le          : {pred['created_at']}\n"
            f"Cible             : {pred['target']:.2f}x\n"
            f"Probabilité       : {pred['probability']*100:.1f} % ({pred['confidence']})\n"
            f"Méthode           : {pred['model_name']}\n"
            f"Ancrage           : manche {pred['anchor_round_id']}\n"
            f"Manches validées  : {pred['window_round_ids']}\n"
            f"Statut            : EN ATTENTE\n\n"
            f"Calcul : P(1 manche ≥ cible) = {loi['P(1 manche >= cible)']*100:.2f} % sur "
            f"{loi['n_manches_utilisees']} manches\n"
            f"         P(validation 3 manches) = {loi['P(validation 3 manches)']*100:.2f} %\n\n"
            f"{pred['explanation']['avertissement']}"
        )


class HistoryPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.addWidget(heading("Historique"))
        close_pending_predictions()
        df = prediction_history(limit=300, backtest=False)
        rows = []
        for _, r in df.iterrows():
            rows.append([r["id"], r["created_at"], f"{r['target']:.2f}x",
                         f"{(r['probability'] or 0)*100:.1f} %", Badge.text_for(r["status"]),
                         r["anchor_round_id"], str(r["window_round_ids"]), str(r["window_values"])])
        lay.addWidget(table(["ID", "Date", "Cible", "Probabilité", "Statut", "Ancrage", "Manches", "Valeurs"], rows))
        lay.addWidget(muted("Les prédictions passent automatiquement de EN ATTENTE à SUCCÈS/ÉCHEC dès que les "
                            "3 manches suivant l'ancrage sont connues."))


class TrainingPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.addWidget(heading("Entraînement"))
        ctl = QHBoxLayout()
        self.folds = QSpinBox()
        self.folds.setRange(1, 40)
        self.folds.setValue(3)
        self.initial = QSpinBox()
        self.initial.setRange(5_000, 80_000)
        self.initial.setSingleStep(5_000)
        self.initial.setValue(CONFIG.backtest.initial_train)
        btn = QPushButton("Lancer l'entraînement + backtest")
        btn.clicked.connect(self.train)
        ctl.addWidget(QLabel("Folds"))
        ctl.addWidget(self.folds)
        ctl.addWidget(QLabel("Entraînement initial"))
        ctl.addWidget(self.initial)
        ctl.addWidget(btn)
        ctl.addStretch(1)
        lay.addLayout(ctl)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        lay.addWidget(self.results, 1)
        self.worker = None
        self.refresh()

    def refresh(self) -> None:
        run_id, metrics, skill, summary = last_metrics_run()
        if not run_id:
            return
        text = [f"Dernier run {run_id}", f"Modèle déployé : {summary.get('deployed_model')}",
                f"Folds : {summary.get('n_folds')} · durée {summary.get('duration_seconds')} s", ""]
        if not skill.empty:
            piv = skill.pivot_table(index=["model_name", "target"], columns="metric", values="value").reset_index()
            text.append(piv.to_string(index=False))
        self.results.setPlainText("\n".join(text))

    def train(self) -> None:
        self.log.setPlainText("Entraînement en cours (walk-forward, réentraînement par fold)…")
        self.worker = TrainWorker(
            targets=CONFIG.targets.fixed,
            models=["logistic", "random_forest", "lightgbm"],
            initial_train=self.initial.value(),
            step=CONFIG.backtest.step,
            max_folds=self.folds.value(),
        )
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(lambda msg: self.log.setPlainText(f"Échec : {msg}"))
        self.worker.start()

    def _done(self, summary: dict) -> None:
        self.log.setPlainText(
            f"Terminé en {summary.get('duration_seconds')} s · {summary.get('n_folds')} folds · "
            f"modèle déployé : {summary.get('deployed_model')}"
        )
        self.refresh()


class DataPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.addWidget(heading("Données"))
        fp = db.data_fingerprint()
        lay.addWidget(card_row([
            KpiCard("Manches", f"{fp['n']:,}".replace(",", " "), f"{fp['first']} → {fp['last']}"),
            KpiCard("Trous", str(fp["gaps"]), "manches manquantes", "#00B894" if not fp["gaps"] else "#E17055"),
            KpiCard("Chaîne SHA256", f"{fp['chain_ok']:,}".replace(",", " "), f"échecs : {fp['chain_bad']}"),
            KpiCard("Doublons", str(fp["duplicates"]), "dédoublonnage sur round_id"),
        ]))
        lay.addWidget(heading("Dernières manches", 15))
        rows = db.get_rounds(limit=60, ascending=False)
        lay.addWidget(table(["Manche", "Coefficient", "Horodatage", "Origine", "Chaîne"],
                            [[r["round_id"], f"{r['multiplier']:.2f}x", r["ts"], r["ts_source"],
                              {1: "OK", 0: "INVALIDE", None: "—"}[r["chain_ok"]]] for r in rows]))
        lay.addWidget(muted(
            "Source : table de vérification provably-fair BC.Game. La chaîne SHA256 prouve l'intégrité et l'ordre. "
            "L'horodatage est estimé pour l'historique et réel (live) pour les manches collectées en continu."
        ))


class ReportPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.addWidget(heading("Rapport"))
        run_id, metrics, skill, summary = last_metrics_run()
        stats = dashboard_stats(backtest=True)
        fp = db.data_fingerprint()
        text = [
            "DONNÉES",
            f"  {fp['n']:,} manches, {fp['gaps']} trou, {fp['duplicates']} doublon, "
            f"{fp['chain_ok']:,} transitions SHA256 vérifiées, {fp['chain_bad']} échec.",
            "",
            "RÉSULTATS",
            f"  Prédictions évaluées : {stats['n_predictions']:,}",
            f"  Taux observé : {(stats.get('success_rate') or 0)*100:.2f} %",
            f"  Brier : {stats.get('brier')}",
            f"  Modèle déployé : {stats.get('deployed_model')}",
        ]
        if skill is not None and not skill.empty:
            piv = skill.pivot_table(index=["model_name", "target"], columns="metric", values="value").reset_index()
            text += ["", "MODÈLES vs BASELINE", piv.to_string(index=False)]
            text += ["", "CONCLUSION", "  Un modèle n'est déployé que s'il bat la baseline de façon significative",
                     "  (correction de Holm). Sinon le robot annonce clairement l'absence d'avantage."]
        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText("\n".join(str(t) for t in text))
        lay.addWidget(view, 1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crash Game AI Predictor")
        self.resize(1280, 820)
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        side_lay = QVBoxLayout(sidebar)
        brand = QLabel("📈 Crash AI")
        brand.setObjectName("Brand")
        side_lay.addWidget(brand)
        self.nav = QListWidget()
        self.nav.setStyleSheet("QListWidget { background: transparent; border: none; }")
        for name in PAGES:
            item = QListWidgetItem(name)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.nav.addItem(item)
        self.nav.currentRowChanged.connect(self._switch)
        side_lay.addWidget(self.nav, 1)
        side_lay.addWidget(muted("Outil d'analyse statistique.\nAucun gain garanti."))
        layout.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.pages = [DashboardPage(self), PredictionPage(self), HistoryPage(self),
                      TrainingPage(self), DataPage(self), ReportPage(self)]
        for p in self.pages:
            self.stack.addWidget(p)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.nav.setCurrentRow(0)

    def _switch(self, index: int) -> None:
        if index >= 0:
            self.stack.setCurrentIndex(index)


def run() -> int:
    db.init_db()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(pyside_stylesheet())
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
