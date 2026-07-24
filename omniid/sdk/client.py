import logging
from typing import Optional, Dict, Any
from omniid.data_engine.ingestion.parser import IngestionParser
from omniid.data_engine.validators.validator import DatasetValidator
from omniid.data_engine.observability.quality import QualityAssessor
from omniid.data_engine.normalization.normalizer import DatasetNormalizer
from omniid.data_engine.normalization.config import NormalizationConfig
from omniid.data_engine.integrity.fingerprint import DatasetFingerprinter
from omniid.data_engine.manifests.builder import ManifestBuilder

logger = logging.getLogger(__name__)

class DatasetClient:
    """
    Data Engine SDK Client.
    Fluid API to ingest, validate, assess, normalize, fingerprint, and manifest datasets.
    """
    def __init__(self):
        self._raw_path = None
        self._ingested_data = None
        self._validation_result = None
        self._quality_result = None
        self._normalized_data = None
        self._fingerprint = None
        self._manifest_path = None
        
        self.config = NormalizationConfig()

    def ingest(self, dataset_path: str) -> 'DatasetClient':
        logger.info(f"Ingesting dataset from {dataset_path}")
        self._raw_path = dataset_path
        parser = IngestionParser(dataset_path)
        self._ingested_data = parser.parse()
        return self

    def validate(self) -> 'DatasetClient':
        if self._ingested_data is None:
            raise ValueError("Cannot validate before ingestion.")
        logger.info("Validating dataset contracts...")
        validator = DatasetValidator()
        self._validation_result = validator.validate(self._ingested_data)
        return self

    def assess_quality(self) -> 'DatasetClient':
        if self._validation_result is None:
            raise ValueError("Cannot assess quality before validation.")
        logger.info("Assessing dataset quality (resolution, blur)...")
        assessor = QualityAssessor()
        self._quality_result = assessor.assess(self._validation_result.accepted)
        return self

    def normalize(self, output_dir: str = "./artifacts/normalized") -> 'DatasetClient':
        if self._quality_result is None:
            raise ValueError("Cannot normalize before quality assessment.")
        logger.info("Normalizing dataset...")
        normalizer = DatasetNormalizer(self.config, output_dir)
        self._normalized_data = normalizer.normalize(self._quality_result.accepted)
        return self

    def fingerprint(self) -> 'DatasetClient':
        if self._normalized_data is None:
            raise ValueError("Cannot fingerprint before normalization.")
        logger.info("Generating dataset fingerprint...")
        fingerprinter = DatasetFingerprinter()
        self._fingerprint = fingerprinter.compute(self._normalized_data, self.config.model_dump())
        return self

    def generate_manifest(self, dataset_name: str = "identity-dataset", output_dir: str = "./artifacts") -> 'DatasetClient':
        if self._fingerprint is None:
            raise ValueError("Cannot generate manifest before fingerprinting.")
        logger.info("Generating dataset manifest...")
        builder = ManifestBuilder(dataset_name)
        
        # Merge validation and quality results for the full report
        full_validation = self._validation_result
        full_validation.merge(self._quality_result)
        
        self._manifest_path = builder.generate(self._normalized_data, self._fingerprint, full_validation, output_dir)
        return self

    def publish(self, publish_dir: str) -> 'DatasetClient':
        """
        Mock publish step. In a real system, uploads to a feature store or blob storage.
        """
        logger.info(f"Published dataset to {publish_dir}. Manifest: {self._manifest_path}")
        return self
