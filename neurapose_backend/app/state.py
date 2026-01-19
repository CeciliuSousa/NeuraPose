# ==============================================================
# neurapose-backend/app/state.py
# ==============================================================
# DEPRECATED: Este arquivo agora re-exporta de global/state.py
# Mantido para compatibilidade com imports existentes

from neurapose_backend.global.state import state, ProcessingState

__all__ = ['state', 'ProcessingState']
