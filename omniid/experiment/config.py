import hashlib
import yaml
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List

def fingerprint_config(cfg: DictConfig) -> str:
    """
    Computes a deterministic SHA256 fingerprint for a given OmegaConf config.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    yaml_str = yaml.dump(container, sort_keys=True)
    return hashlib.sha256(yaml_str.encode('utf-8')).hexdigest()

def diff_configs(cfg1: DictConfig, cfg2: DictConfig) -> Dict[str, Any]:
    """
    Computes a simplified diff between two OmegaConf configurations.
    Returns a dictionary of differing keys with their respective values.
    """
    dict1 = OmegaConf.to_container(cfg1, resolve=True)
    dict2 = OmegaConf.to_container(cfg2, resolve=True)
    
    # Flatten dicts for easier diffing
    def flatten(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    flat1 = flatten(dict1)
    flat2 = flatten(dict2)
    
    diff = {}
    all_keys = set(flat1.keys()).union(set(flat2.keys()))
    
    for k in all_keys:
        v1 = flat1.get(k)
        v2 = flat2.get(k)
        if v1 != v2:
            diff[k] = {"old": v1, "new": v2}
            
    return diff
