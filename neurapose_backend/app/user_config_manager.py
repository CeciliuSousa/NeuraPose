import json
from pathlib import Path
from typing import Dict, Any
import config_master as cm


CONFIG_FILE = Path(__file__).resolve().parent.parent / "user_settings.json"

class UserConfigManager:
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Carrega as configurações do arquivo JSON ou retorna as configurações do config_master."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ERROR] Falha ao carregar user_settings.json: {e}")
        
        # Se não existir ou falhar, retorna o que está no config_master (via RUNTIME_CONFIG se possível)
        return UserConfigManager.get_default_config()

    @staticmethod
    def save_config(config: Dict[str, Any]):
        """Salva as configurações no arquivo JSON."""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"[OK] Configurações salvas em {CONFIG_FILE}")
        except Exception as e:
            print(f"[ERROR] Falha ao salvar user_settings.json: {e}")

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Extrai as configurações padrão do config_master."""
        # Esta lista deve ser mantida sincronizada com o que queremos expor no front
        return {
            "DETECTION_CONF": cm.DETECTION_CONF,
            "POSE_CONF_MIN": cm.POSE_CONF_MIN,
            "EMA_ALPHA": cm.EMA_ALPHA,
            "PROCESSING_DATASET": cm.PROCESSING_DATASET,
            "OSNET_MODEL": cm.OSNET_MODEL,
            "YOLO_MODEL": cm.YOLO_MODEL,
            "RTMPOSE_MODEL": cm.RTMPOSE_MODEL,
            "DEVICE": cm.DEVICE,
            "CLASSE1": cm.CLASSE1,
            "CLASSE2": cm.CLASSE2,
            "CLASSE2_THRESHOLD": cm.CLASSE2_THRESHOLD,
            "BATCH_SIZE": cm.BATCH_SIZE,
            "EPOCHS": cm.EPOCHS,
            "LEARNING_RATE": cm.LEARNING_RATE,
            "TIME_STEPS": cm.TIME_STEPS,
            # BotSORT
            "TRACK_HIGH_THRESH": cm.BOT_SORT_CONFIG.get("track_high_thresh", 0.5),
            "TRACK_LOW_THRESH": cm.BOT_SORT_CONFIG.get("track_low_thresh", 0.1),
            "NEW_TRACK_THRESH": cm.BOT_SORT_CONFIG.get("new_track_thresh", 0.6),
            "TRACK_BUFFER": cm.BOT_SORT_CONFIG.get("track_buffer", 30),
            "MATCH_THRESH": cm.BOT_SORT_CONFIG.get("match_thresh", 0.8),
            "PROXIMITY_THRESH": cm.BOT_SORT_CONFIG.get("proximity_thresh", 0.5),
            "APPEARANCE_THRESH": cm.BOT_SORT_CONFIG.get("appearance_thresh", 0.25),
        }

    @staticmethod
    def reset_to_defaults():
        """Remove o arquivo de configurações do usuário para voltar ao config_master."""
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
            print("[OK] user_settings.json removido. Restaurando padrões de config_master.")
