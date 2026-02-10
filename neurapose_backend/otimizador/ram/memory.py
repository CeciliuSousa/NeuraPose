import gc
import psutil
import os
from colorama import Fore
from neurapose_backend import config_master as cm

def get_ram_usage_mb():
    """Retorna uso de RAM do processo atual em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def force_gc(verbose=False):
    """
    Força coleta de lixo completa.
    Retorna quanto foi liberado (estimado, pois GC do Python é complexo).
    """
    before = get_ram_usage_mb()
    gc.collect()
    after = get_ram_usage_mb()
    freed = before - after
    
    if verbose or freed > 100: # Log apenas se liberar muito (>100MB)
        print(Fore.GREEN + f"[RAM] GC Liberado: {freed:.1f} MB (Atual: {after:.1f} MB)")
        
    return freed

def smart_cleanup(frame_idx=None):
    """
    Limpeza inteligente baseada em intervalos configurados.
    Pode ser chamado dentro de loops de processamento.
    """
    if not getattr(cm, 'SAFE_MODE', True):
        return

    interval = getattr(cm, 'GC_COLLECT_INTERVAL', 50)
    
    # Se frame_idx for None, força limpeza (fim de processo)
    if frame_idx is None:
        force_gc()
        return

    if frame_idx > 0 and frame_idx % interval == 0:
        force_gc(verbose=False)
