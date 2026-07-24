from dataclasses import dataclass, field
from typing import List, Any, Dict

@dataclass
class ValidationResult:
    """
    Collects validation results for a dataset processing run.
    Ensures the pipeline can continue processing without crashing on the first invalid sample.
    """
    accepted: List[Any] = field(default_factory=list)
    rejected: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, sample_id: str, error_msg: str):
        self.errors.append(f"[{sample_id}] ERROR: {error_msg}")

    def add_warning(self, sample_id: str, warning_msg: str):
        self.warnings.append(f"[{sample_id}] WARNING: {warning_msg}")

    def merge(self, other: 'ValidationResult'):
        self.accepted.extend(other.accepted)
        self.rejected.extend(other.rejected)
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors[:100],  # Limit to first 100 for report
            "warnings": self.warnings[:100]
        }
