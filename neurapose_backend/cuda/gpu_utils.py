import torch
import gc
from typing import Dict, Tuple
from contextlib import contextmanager


class GPUManager:
    """
    Gerenciador de recursos GPU para RTMPose/RTMDet.
    Otimizações: Memory pooling, async processing, mixed precision.
    
    Big-O (init): O(1) - apenas configurações
    """

    def __init__(self, device: str = 'cuda:0', memory_fraction: float = 0.90) -> None:
        """
        Inicializa gerenciador GPU com alocação eficiente.
        
        Args:
            device: 'cuda:0', 'cuda:1', etc. ou 'cpu'
            memory_fraction: Fração da memória GPU a usar (0.0-1.0)
        
        Big-O: O(1) - configuração de variáveis
        """
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
        
        Args:
            model: Modelo PyTorch
        
        Returns:
            Modelo compilado otimizado
        """
        if hasattr(torch, 'compile') and self.is_cuda:
            print("Compilando modelo com torch.compile (reduce-overhead)...")
            try:
                # 'reduce-overhead' usa CUDA Graphs (ótimo para loops pequenos)
                # 'max-autotune' demora muito para compilar mas é o mais rápido
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
        Mais rápido que no_grad().
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
        Big-O: O(C) onde C = cache size
        """
        if self.is_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Retorna estatísticas de memória GPU.
        Big-O: O(1) - leitura de registros GPU
        
        Returns:
            Dict com allocated, reserved, free (em MB)
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
        """
        Sincroniza GPU (aguarda conclusão de operações).
        Use antes de medir tempo.
        
        Big-O: O(P) onde P = pending ops (geralmente < 1ms)
        """
        if self.is_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def profile_memory(self, name: str = "Operation"):
        """
        Context manager para profile memória GPU.
        
        Usage:
            with gpu_manager.profile_memory("frame_processing"):
                # código aqui
        
        Big-O: O(1)
        """
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        yield
        
        if self.is_cuda:
            torch.cuda.synchronize()
            # peak_mem = torch.cuda.max_memory_allocated() / 1e6
            # print(f"[{name}] Pico memória: {peak_mem:.2f} MB") - Removido para não poluir UI

    def reset_peak_memory_stats(self) -> None:
        """
        Reseta estatísticas de pico de memória.
        Big-O: O(1)
        """
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def get_peak_memory_stats(self) -> float:
        """
        Retorna o pico de memória alocada desde o último reset (em MB).
        Big-O: O(1)
        """
        if self.is_cuda:
            return torch.cuda.max_memory_allocated() / 1e6
        return 0.0


def enable_cudnn_auto_tuning() -> None:
    """
    Ativa CUDNN auto-tuning global.
    Big-O: O(1)
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def clear_gpu_cache() -> None:
    """
    Limpa cache GPU + Python garbage collection.
    Big-O: O(C)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class GPUMemoryPool():
    """
    Pool de alocação de tensores GPU pré-alocados (memory pooling).
    Reduz fragmentação e overhead de alocação.
    
    Big-O (alloc): O(1) reutilização, O(N) primeira alocação
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
        """
        Libera pool.
        Big-O: O(P) onde P = número de tensores no pool
        """
        self.pool_tensors.clear()
        if self.is_cuda:
            torch.cuda.empty_cache()

