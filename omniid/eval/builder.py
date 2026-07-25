from omniid.eval.registry import EVALUATOR_REGISTRY
from omniid.eval.base import BaseEvaluator

def build_evaluator(name: str, **kwargs) -> BaseEvaluator:
    """
    Instantiates an evaluation suite by name.
    """
    # Trigger registrations
    import omniid.eval.tasks.retrieval
    import omniid.eval.tasks.verification
    import omniid.eval.tasks.clustering
    
    return EVALUATOR_REGISTRY.build(name, **kwargs)
