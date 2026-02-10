import torch
import gc
from typing import Dict, Tuple
from contextlib import contextmanager
import neurapose_backend.config_master as cm


class GPUManager:
    """
    Gerenciador de recursos GPU para RTMPose/RTMDet.
    Otimizações: Memory pooling, async processing, mixed precision.
    
    Big-O (init): O(1) - apenas configurações
    """

    def __init__(self, device: str = cm.DEVICE, memory_fraction: float = None) -> None:
        """
        Inicializa gerenciador GPU com alocação eficiente.
        
        Args:
            device: 'cuda:0', 'cuda:1', etc. ou 'cpu'
            memory_fraction: Fração da memória GPU a usar (0.0-1.0)
        
        Big-O: O(1) - configuração de variáveis
        """
        if memory_fraction is None:
            memory_fraction = getattr(cm, 'MAX_VRAM_USAGE_RATIO', 0.85)

        self.device: str = device
        self.is_cuda: bool = 'cuda' in device and torch.cuda.is_available()
        
        if self.is_cuda:
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
            torch.cuda.set_device(gpu_id)
            torch.cuda.reset_peak_memory_stats()
            
            # Aloca fração máxima de memória GPU
            try:
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device=gpu_id)
            except RuntimeError:
                print(f"Aviso: Não foi possível limitar memória GPU a {memory_fraction*100}%")

    def update_device(self, device: str) -> None:
        """
        Atualiza o dispositivo ativo (ex: troca CPU/GPU em tempo de execução).
        IMPORTANTE: Chamar isso se cm.DEVICE mudar.
        """
        self.device = device
        self.is_cuda = 'cuda' in device and torch.cuda.is_available()
        
        if self.is_cuda:
            try:
                gpu_id = int(device.split(':')[1]) if ':' in device else 0
                torch.cuda.set_device(gpu_id)
            except:
                pass

    def enable_mixed_precision(self) -> None:
        """
        Ativa automático mixed precision (float32 + float16).
        Aumenta throughput em ~2x com mínima perda de precisão.
        """
        if self.is_cuda:
            torch.set_float32_matmul_precision('high') # 'medium' for even more speed
            print(f"Mixed precision ativado na {self.device}")

    def enable_cudnn_benchmarking(self) -> None:
        """
        Ativa CUDNN auto-tuning (melhor kernel selection).
        Recomendado se input shape é fixo (recomendado em vídeos).
        """
        if self.is_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print(f"CUDNN benchmarking ativado")

    def compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Compila modelo com torch.compile (Torch 2.0+).
        Modo: 'reduce-overhead' para minimizar latência de python/cuda calls.
        """
        if hasattr(torch, 'compile') and self.is_cuda:
            print("Compilando modelo com torch.compile (reduce-overhead)...")
            try:
                # 'reduce-overhead' usa CUDA Graphs (ótimo para loops pequenos)
                return torch.compile(model, mode='reduce-overhead')
            except Exception as e:
                print(f"Aviso: Falha ao compilar modelo: {e}")
                return model
        return model

    @contextmanager
    def inference_mode(self):
        """
        Context manager para inferência otimizada.
        Desabilita gradientes e view tracking.
        """
        if torch.cuda.is_available():
             with torch.inference_mode():
                 yield
        else:
             with torch.no_grad():
                 yield

    def warmup(self, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> None:
        """
        Aquece a GPU executando 10 iterações dummy.
        Evita latência no primeiro frame real.
        """
        if self.is_cuda:
            print("Aquecendo GPU...")
            dummy_input = torch.randn(input_shape, device=self.device)
            # stream = torch.cuda.Stream()
            # with torch.cuda.stream(stream):
            for _ in range(10):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            print("GPU aquecida!")

    def clear_cache(self) -> None:
        """
        Limpa cache GPU e garbage collection Python.
        """
        if self.is_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Retorna estatísticas de memória GPU.
        Returns: Dict com allocated, reserved, free (em MB)
        """
        if not self.is_cuda:
            return {'allocated_mb': 0, 'reserved_mb': 0, 'free_mb': 0}

        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e6
        free = total - allocated

        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'total_mb': total,
            'utilization_percent': (allocated / total * 100) if total > 0 else 0
        }

    def synchronize(self) -> None:
        """Sincroniza GPU (aguarda conclusão de operações)."""
        if self.is_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def profile_memory(self, name: str = "Operation"):
        """Context manager para profile memória GPU."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        yield
        
        if self.is_cuda:
            torch.cuda.synchronize()

    def reset_peak_memory_stats(self) -> None:
        """Reseta estatísticas de pico de memória."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def get_peak_memory_stats(self) -> float:
        """Retorna o pico de memória alocada desde o último reset (em MB)."""
        if self.is_cuda:
            return torch.cuda.max_memory_allocated() / 1e6
        return 0.0


def enable_cudnn_auto_tuning() -> None:
    """Ativa CUDNN auto-tuning global."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def clear_gpu_cache() -> None:
    """Limpa cache GPU + Python garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def check_gpu_memory(min_free_mb=None, margin_mb=None):
    """
    Verifica memória livre na GPU de forma leve (sem alocação de teste).
    Retorna (ok, msg, free_mb).
    Smart Check: Usa mem_get_info driver call (custo zero).
    """
    if not torch.cuda.is_available():
        return False, "GPU não disponível", 0

    if min_free_mb is None:
        min_free_mb = cm.RTMPOSE_BATCH_SIZE * 2  # Estimativa
    
    if margin_mb is None:
        margin_mb = getattr(cm, 'MEMORY_SAFETY_MARGIN_MB', 1024)

    try:
        # Usa torch.cuda.mem_get_info() que é leve (driver call)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024)
        
        needed_mb = min_free_mb + margin_mb
        
        if free_mb < needed_mb:
            return False, f"VRAM Insuficiente: {free_mb:.0f}MB livres (Precisa {needed_mb:.0f}MB)", free_mb
            
        return True, f"VRAM OK: {free_mb:.0f}MB livres de {total_mb:.0f}MB", free_mb

    except Exception as e:
        return False, f"Erro check VRAM: {str(e)}", 0


class GPUMemoryPool():
    """
    Pool de alocação de tensores GPU pré-alocados (memory pooling).
    reduz fragmentação e overhead de alocação.
    """

    def __init__(self, device: str = 'cuda:0', pool_size_mb: int = 512) -> None:
        """
        Inicializa pool de memória.
        
        Args:
            device: Dispositivo CUDA
            pool_size_mb: Tamanho pré-alocado em MB
        """
        self.device = device
        self.is_cuda = 'cuda' in device and torch.cuda.is_available()
        self.pool_tensors = []
        
        if self.is_cuda:
            # Pré-aloca pool
            num_tensors = max(1, pool_size_mb // 256)
            for _ in range(num_tensors):
                t = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
                self.pool_tensors.append(t)
            print(f"Pool GPU alocado: {pool_size_mb}MB em {num_tensors} tensores")

    def clear_pool(self) -> None:
        """Libera pool."""
        self.pool_tensors.clear()
        if self.is_cuda:
            torch.cuda.empty_cache()


# Singleton Global Instance
# Permite importacao direta: from neurapose_backend.cuda.gpu_utils import gpu_manager
gpu_manager = GPUManager(device=cm.DEVICE)
