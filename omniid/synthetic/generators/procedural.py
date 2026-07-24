import os
import random
from PIL import Image, ImageDraw
from typing import Dict, Any
from omniid.synthetic.generators.base import (
    BaseFaceGenerator, BaseDocumentGenerator, 
    BaseSignatureGenerator, BaseVoiceGenerator
)

class ProceduralFaceGenerator(BaseFaceGenerator):
    def generate(self, persona: Dict[str, Any], seed: int, output_path: str) -> str:
        random.seed(seed)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', (256, 256), color=color)
        d = ImageDraw.Draw(img)
        d.text((50, 120), persona.get("name", "Unknown"), fill=(255, 255, 255))
        img.save(output_path)
        return output_path

class ProceduralDocumentGenerator(BaseDocumentGenerator):
    def generate(self, persona: Dict[str, Any], template: str, seed: int, output_path: str) -> str:
        random.seed(seed)
        img = Image.new('RGB', (600, 400), color=(200, 200, 200))
        d = ImageDraw.Draw(img)
        d.text((20, 20), f"Template: {template}", fill=(0, 0, 0))
        d.text((20, 60), f"Name: {persona.get('name')}", fill=(0, 0, 0))
        d.text((20, 100), f"DOB: {persona.get('dob')}", fill=(0, 0, 0))
        img.save(output_path)
        return output_path

class ProceduralSignatureGenerator(BaseSignatureGenerator):
    def generate(self, name: str, seed: int, output_path: str) -> str:
        random.seed(seed)
        img = Image.new('RGB', (300, 100), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.line([(20, 50), (280, 50)], fill=(0, 0, 0), width=2)
        d.text((100, 40), name, fill=(0, 0, 0))
        img.save(output_path)
        return output_path

class PlaceholderVoiceGenerator(BaseVoiceGenerator):
    def generate(self, text: str, seed: int, output_path: str) -> str:
        # Create a dummy valid wav file (very basic header)
        import wave
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b'\x00' * 1600) # 0.1s of silence
        return output_path
