"""Prueba exhaustiva de los interruptores de activación/desactivación por etapa del pipeline visual."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configurar modo offscreen para Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtWidgets import QApplication

from core.config import PIPELINE_STAGES, PipelineConfig
from core.pipeline import analyze_image
from ui.main_window import MainWindow


def run_tests():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win.goto_page(1)
    app.processEvents()
    explorer = win.explorer

    print("1. Verificando que todas las tarjetas tengan su interruptor...")
    expected_stages = [s["key"] for s in PIPELINE_STAGES]
    for key in expected_stages:
        assert key in explorer.stage_cards, f"Falta tarjeta para etapa '{key}'"
        card = explorer.stage_cards[key]
        assert hasattr(card, "switch"), f"Tarjeta '{key}' no tiene switch"
        assert hasattr(card, "off_badge"), f"Tarjeta '{key}' no tiene off_badge"
        if key in ("read", "net"):
            assert not card.can_toggle, f"Tarjeta '{key}' no debe ser conmutable"
            assert not card.switch.isEnabled()
        else:
            assert card.can_toggle, f"Tarjeta '{key}' debe ser conmutable"
            assert card.switch.isEnabled()
    print("   -> Todas las 12 tarjetas y sus interruptores verificados OK.")

    print("2. Verificando sincronización bidireccional tarjeta <-> controles...")
    toggleable = [s["key"] for s in PIPELINE_STAGES if not s.get("always", False)]
    for key in toggleable:
        card = explorer.stage_cards[key]
        ctrl_sw = explorer.controls.stage_switches[key]

        # Apagar desde la tarjeta
        card.switch.setChecked(False)
        app.processEvents()
        assert not card.switch.isChecked()
        assert not ctrl_sw.isChecked(), f"Control '{key}' no se sincronizó a False"
        assert not explorer.cfg.stage_enabled(key), f"explorer.cfg['{key}'] no se actualizó a False"
        assert not card.off_badge.isHidden(), f"Badge de 'DESACTIVADA' no visible para '{key}'"

        # Encender desde la tarjeta
        card.switch.setChecked(True)
        app.processEvents()
        assert card.switch.isChecked()
        assert ctrl_sw.isChecked(), f"Control '{key}' no se sincronizó a True"
        assert explorer.cfg.stage_enabled(key), f"explorer.cfg['{key}'] no se actualizó a True"
        assert card.off_badge.isHidden(), f"Badge de 'DESACTIVADA' visible tras reactivar '{key}'"

        # Apagar desde el panel de controles
        ctrl_sw.setChecked(False)
        app.processEvents()
        assert not card.switch.isChecked(), f"Tarjeta '{key}' no se sincronizó al apagar desde controles"
        assert not card.off_badge.isHidden()

        # Encender desde el panel de controles
        ctrl_sw.setChecked(True)
        app.processEvents()
        assert card.switch.isChecked(), f"Tarjeta '{key}' no se sincronizó al encender desde controles"
        assert card.off_badge.isHidden()
    print("   -> Sincronización bidireccional para las 10 etapas conmutables OK.")

    print("3. Verificando ejecución del pipeline con etapas desactivadas individualmente y en conjunto...")
    sample_path = win.dataset_info.samples[0].path if win.dataset_info and win.dataset_info.samples else None
    if not sample_path:
        # Usar una imagen dummy si el dataset no estuviera cargado
        sample_path = Path("Tema/peces/trucha/trucha_01.jpg")

    assert Path(sample_path).exists(), f"No existe imagen de prueba: {sample_path}"

    # Probar apagando cada etapa individualmente
    for key in toggleable:
        cfg = PipelineConfig()
        cfg.set_stage(key, False)
        assert not cfg.stage_enabled(key)
        res = analyze_image(sample_path, cfg)
        assert res is not None
        assert res.stage_states[key] is False, f"La etapa '{key}' no figura como desactivada en el resultado"
        assert res.net_input.shape == (cfg.net_input_size, cfg.net_input_size)

    # Probar apagando múltiples etapas a la vez (por ejemplo: sin ROI, sin ecualización, sin conv, sin pool)
    multi_cfg = PipelineConfig()
    multi_cfg.set_stage("roi", False)
    multi_cfg.set_stage("enhance", False)
    multi_cfg.set_stage("conv", False)
    multi_cfg.set_stage("pool", False)
    multi_res = analyze_image(sample_path, multi_cfg)
    assert multi_res.stage_states["roi"] is False
    assert multi_res.stage_states["enhance"] is False
    assert multi_res.stage_states["conv"] is False
    assert multi_res.stage_states["pool"] is False
    assert multi_res.net_input.shape == (multi_cfg.net_input_size, multi_cfg.net_input_size)
    print("   -> analyze_image funcionó correctamente con etapas desactivadas individual y conjuntamente.")

    print("4. Verificando persistencia entre múltiples imágenes...")
    # Apagamos 'conv' y 'roi' en el explorador
    explorer.stage_cards["conv"].switch.setChecked(False)
    explorer.stage_cards["roi"].switch.setChecked(False)
    assert not explorer.cfg.stage_enabled("conv")
    assert not explorer.cfg.stage_enabled("roi")

    # Cambiamos a diferentes imágenes del dataset
    samples = win.dataset_info.samples[:3] if win.dataset_info and len(win.dataset_info.samples) >= 3 else []
    for s in samples:
        explorer.set_sample(str(s.path))
        assert not explorer.cfg.stage_enabled("conv"), "La configuración perdió el estado de 'conv'"
        assert not explorer.cfg.stage_enabled("roi"), "La configuración perdió el estado de 'roi'"
        assert not explorer.stage_cards["conv"].switch.isChecked()
        assert not explorer.stage_cards["roi"].switch.isChecked()

    print("   -> Persistencia entre imágenes verificada OK.")

    # Restaurar
    explorer.stage_cards["conv"].switch.setChecked(True)
    explorer.stage_cards["roi"].switch.setChecked(True)

    print("\nTODAS LAS PRUEBAS DE INTERRUPTORES PASARON SATISFACTORIAMENTE (100%).")


if __name__ == "__main__":
    run_tests()
