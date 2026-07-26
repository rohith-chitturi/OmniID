try:
    from fastapi import FastAPI, HTTPException
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from omniid.runtime.types import InferenceRequest, InferenceResponse
from omniid.runtime.pipeline import InferencePipeline

class FastAPIServer:
    """
    Thin serving layer wrapping the InferencePipeline.
    """
    def __init__(self, model_name: str):
        if not HAS_FASTAPI:
            raise ImportError("FastAPI is required for the serving layer. Run 'pip install fastapi uvicorn'.")
            
        self.app = FastAPI(title=f"OmniID Inference: {model_name}")
        self.pipeline = InferencePipeline.from_model(model_name)
        
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.post("/v1/predict", response_model=InferenceResponse)
        async def predict(request: InferenceRequest):
            try:
                response = self.pipeline.predict(request)
                return response
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
                
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "model_version": self.pipeline.manifest.version}
