import time
import os
import psutil
from neurapose_backend import config_master as cm

def throttle(interval=None):
    """
    Pausa a execução para permitir que a CPU esfrie.
    Útil em loops infinitos ou processamento pesado.
    
    Args:
        interval: Tempo em segundos. Se None, usa config_master.LOOP_SLEEP_INTERVAL.
    """
    if not getattr(cm, 'SAFE_MODE', True):
        return

    if interval is None:
        interval = getattr(cm, 'LOOP_SLEEP_INTERVAL', 0.01)
        
    time.sleep(interval)

def get_optimal_threads():
    """
    Retorna o número ideal de threads baseado no modo de segurança.
    """
    if getattr(cm, 'SAFE_MODE', True):
        return getattr(cm, 'MAX_CPU_WORKERS', 2)
    
    return os.cpu_count() or 4

def set_process_priority(high=True):
    """
    Ajusta prioridade do processo.
    Safe Mode = Normal (evita travar PC).
    Performance Mode = Alta.
    """
    try:
        p = psutil.Process(os.getpid())
        if getattr(cm, 'SAFE_MODE', True):
            # Normal Priority
            if os.name == 'nt':
                p.nice(psutil.NORMAL_PRIORITY_CLASS)
            else:
                p.nice(0)
        else:
            # High Priority (se solicitado)
            if high:
                if os.name == 'nt':
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                else:
                    p.nice(-10) # Pode falhar sem sudo
    except Exception as e:
        print(f"[CPU] Falha ao ajustar prioridade: {e}")
