import json
from pathlib import Path
from typing import Dict, Any
import config_master as cm


CONFIG_FILE = Path(__file__).resolve().parent.parent / "user_settings.json"

class UserConfigManager:
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Carrega as configurações do arquivo JSON ou retorna as configurações do config_master."""
        defaults = UserConfigManager.get_default_config()
        
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved_config = json.load(f)
                    # Merge: valores salvos sobrescrevem os padrões
                    defaults.update(saved_config)
                    return defaults
            except Exception as e:
                print(f"[ERROR] Falha ao carregar user_settings.json: {e}")
        
        return defaults

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
        return {
            # Modelos de IA
            "YOLO_MODEL": cm.YOLO_MODEL,
            "OSNET_MODEL": cm.OSNET_MODEL,
            "RTMPOSE_MODEL": cm.RTMPOSE_MODEL,
            "TEMPORAL_MODEL": getattr(cm, "TEMPORAL_MODEL", "lstm"),
            
            # Classes de Detecção
            "CLASSE1": cm.CLASSE1,
            "CLASSE2": cm.CLASSE2,
            "CLASSE2_THRESHOLD": cm.CLASSE2_THRESHOLD,
            
            # Configurações YOLO
            "DETECTION_CONF": cm.DETECTION_CONF,
            "YOLO_IMGSZ": getattr(cm, "YOLO_IMGSZ", "1280"),
            
            # Configurações de Pose
            "POSE_CONF_MIN": cm.POSE_CONF_MIN,
            "EMA_ALPHA": cm.EMA_ALPHA,
            "EMA_MIN_CONF": getattr(cm, "EMA_MIN_CONF", 0.3),
            
            # BoTSORT - usando chaves minúsculas como o frontend espera
            "track_high_thresh": cm.BOT_SORT_CONFIG.get("track_high_thresh", 0.5),
            "track_low_thresh": cm.BOT_SORT_CONFIG.get("track_low_thresh", 0.1),
            "new_track_thresh": cm.BOT_SORT_CONFIG.get("new_track_thresh", 0.6),
            "match_thresh": cm.BOT_SORT_CONFIG.get("match_thresh", 0.8),
            "track_buffer": cm.BOT_SORT_CONFIG.get("track_buffer", 30),
            "proximity_thresh": cm.BOT_SORT_CONFIG.get("proximity_thresh", 0.5),
            "appearance_thresh": cm.BOT_SORT_CONFIG.get("appearance_thresh", 0.25),
            
            # Parâmetros de Treinamento
            "TIME_STEPS": cm.TIME_STEPS,
            "BATCH_SIZE": cm.BATCH_SIZE,
            "LEARNING_RATE": cm.LEARNING_RATE,
            "EPOCHS": cm.EPOCHS,
            
            # Parâmetros de Sequência
            "MAX_FRAMES_PER_SEQUENCE": getattr(cm, "MAX_FRAMES_PER_SEQUENCE", 1800),
            "MIN_FRAMES_PER_ID": getattr(cm, "MIN_FRAMES_PER_ID", 30),
            
            # Outros
            "DEVICE": cm.DEVICE,
            "PROCESSING_DATASET": cm.PROCESSING_DATASET,
        }

    @staticmethod
    def reset_to_defaults():
        """Remove o arquivo de configurações do usuário para voltar ao config_master."""
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
            print("[OK] user_settings.json removido. Restaurando padrões de config_master.")
