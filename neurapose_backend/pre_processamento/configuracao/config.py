# ==============================================================
# neusapose-backend/pre_processamento/configuracao/config.py
# ==============================================================

# Importa TUDO do config_master.py
from neurapose_backend.config_master import (
    # Parametros de processamento

    POSE_CONF_MIN,
    CLAMP_MARGIN,
    EMA_ALPHA,
    EMA_MIN_CONF,
    TRACKER_NAME,
    
    # Dimensoes RTMPose
    SIMCC_W,
    SIMCC_H,
    SIMCC_SPLIT_RATIO,
    
    # Normalizacao
    MEAN,
    STD,
    
    # Visualizacao
    PAIRS,
    
    # Paths de pre-processamento
    PROCESSING_DATASET,
    PROCESSING_INPUT_DIR,
    PROCESSING_OUTPUT_DIR,
    PROCESSING_VIDEOS_DIR,
    PROCESSING_PREDS_DIR,
    PROCESSING_JSONS_DIR,
    PROCESSING_ANNOTATIONS_DIR,
    RTMPOSE_PREPROCESSING_PATH,
    
    # Parametros de sequencia (CRITICOS - devem ser iguais em proc/treino/teste)
    MAX_FRAMES_PER_SEQUENCE,
    MIN_FRAMES_PER_ID,
    FPS_TARGET,
    FRAME_DISPLAY_W,
    FRAME_DISPLAY_H,
    
    # Classes
    CLASS_NAMES,
    CLASSE1,
    CLASSE2,
    
    # Dataset de saida
    TEST_DATASET,
) 
