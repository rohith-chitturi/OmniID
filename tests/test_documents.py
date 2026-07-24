import pytest
from omniid.models.registry import BACKBONE_REGISTRY
from omniid.models.builder import build_encoder
from omniid.models.document_encoders.layoutlmv3 import LayoutLMv3Encoder
from omniid.models.document_encoders.donut import DonutEncoder
from omniid.documents.pipeline import DocumentPreprocessingPipeline
from omniid.documents.ocr import MockOCRProvider

def test_document_registry():
    llm3 = build_encoder("layoutlmv3")
    donut = build_encoder("donut")
    
    assert isinstance(llm3, LayoutLMv3Encoder)
    assert isinstance(donut, DonutEncoder)

def test_layoutlmv3_schema():
    encoder = build_encoder("layoutlmv3")
    assert encoder.metadata.supports_ocr
    assert encoder.metadata.supports_layout
    
    spec = encoder.preprocess
    assert spec.bounding_box_scale == 1000
    
    pipeline = DocumentPreprocessingPipeline(spec, ocr_provider=MockOCRProvider())
    doc_input = pipeline("mock_image")
    
    # 2 mock boxes [10,10,50,20] & [60,10,120,20] -> scaled by 10 (1000/100)
    assert doc_input["layout_boxes"][0] == [100, 100, 500, 200]
    
    output = encoder.encode(doc_input)
    assert list(output.document_embedding.shape) == [1, 768]
    assert list(output.token_embeddings.shape) == [1, 512, 768]

def test_donut_schema():
    encoder = build_encoder("donut")
    assert encoder.metadata.supports_generation
    assert not encoder.metadata.supports_ocr
    
    spec = encoder.preprocess
    assert spec.bounding_box_scale is None
    
    pipeline = DocumentPreprocessingPipeline(spec)
    doc_input = pipeline("mock_image")
    
    output = encoder.encode(doc_input)
    assert list(output.document_embedding.shape) == [1, 1024]
    assert output.text == "<s_receipt><s_company>OmniCorp</s_company></s_receipt>"
