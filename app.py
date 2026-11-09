"""AquaCNN · Laboratorio 6 — punto de entrada de la aplicación."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Permite ejecutar `python app.py` desde cualquier directorio.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_excepthook(app) -> None:
    from PySide6.QtWidgets import QMessageBox

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(detail)
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("AquaCNN · error inesperado")
            box.setText("Se produjo un error inesperado, pero la aplicación puede continuar.")
            box.setInformativeText(str(exc))
            box.setDetailedText(detail)
            box.exec()
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


def _set_app_user_model_id(app_id: str = "AquaCNN.Laboratorio6") -> None:
    """Hace que Windows use el icono de AquaCNN en la barra de tareas."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:  # noqa: BLE001
        pass


def _install_tooltip_guard(app):
    """Silencia los tooltips durante el arranque y los oculta a los 4 s.

    Al construir la ventana, el cursor suele estar sobre la pantalla y Qt puede
    mostrar y cerrar ventanas de tooltip en cadena. El filtro se desarma justo
    después de mostrar la ventana principal. Después, cualquier tooltip se
    cierra solo tras unos segundos para no tapar la interfaz.
    """
    from PySide6.QtCore import QEvent, QObject, QTimer
    from PySide6.QtWidgets import QToolTip

    from ui import preferences

    class TooltipGuard(QObject):
        def __init__(self) -> None:
            super().__init__()
            self.armed = True
            self._hide = QTimer(self)
            self._hide.setSingleShot(True)
            self._hide.timeout.connect(QToolTip.hideText)

        def eventFilter(self, obj, event) -> bool:  # noqa: N802
            if event.type() == QEvent.Type.ToolTip:
                if self.armed or not preferences.get("tooltips", True):
                    return True
                self._hide.start(4000)
            return False

    guard = TooltipGuard()
    app.installEventFilter(guard)
    return guard


def main() -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from core.config import APP_NAME, APP_TITLE
    from ui import preferences
    from ui import icons
    from ui.theme import apply_theme, load_saved_palette

    _set_app_user_model_id()
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_TITLE)
    app.setOrganizationName("AquaCNN")
    app.setOrganizationDomain("aquacnn.local")

    # La paleta se configura ANTES de importar las páginas para que sus estilos
    # en línea usen los colores elegidos desde el primer fotograma.
    load_saved_palette()
    apply_theme(app)
    _install_excepthook(app)
    app.setWindowIcon(icons.app_icon())

    guard = _install_tooltip_guard(app)

    from ui.main_window import MainWindow

    window = MainWindow()
    if preferences.get("start_maximized", True):
        window.showMaximized()
    else:
        window.showNormal()

    QTimer.singleShot(1400, lambda: setattr(guard, "armed", False))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
