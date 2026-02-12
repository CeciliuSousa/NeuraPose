# Routes package initialization
# This file exposes the routers for easier importing

from .system import router as system_router
from .config import router as config_router
from .processing import router as processing_router
from .training import router as training_router
from .testing import router as testing_router
# from .reid import router as reid_router
from .annotation import router as annotation_router
from .dataset import router as dataset_router
