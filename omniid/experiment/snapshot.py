import platform
import sys
import subprocess
from typing import Optional
from omniid.experiment.models import EnvironmentSnapshot

def get_git_commit() -> Optional[str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit
    except Exception:
        return None

def capture_environment() -> EnvironmentSnapshot:
    pytorch_version = None
    cuda_available = False
    
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    return EnvironmentSnapshot(
        os=platform.system(),
        platform=platform.platform(),
        python_version=sys.version,
        pytorch_version=pytorch_version,
        cuda_available=cuda_available,
        git_commit=get_git_commit()
    )
