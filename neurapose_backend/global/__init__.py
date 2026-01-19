# ==============================================================
# neurapose_backend/global/__init__.py
# ==============================================================
# Módulos globais compartilhados entre app e pre_processamento

from .state import state, ProcessingState

__all__ = ['state', 'ProcessingState']
