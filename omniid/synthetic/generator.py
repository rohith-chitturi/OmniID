import os
import json
import random
from typing import Dict, Any, Optional
from omniid.synthetic.generators.procedural import (
    ProceduralFaceGenerator, ProceduralDocumentGenerator, 
    ProceduralSignatureGenerator, PlaceholderVoiceGenerator
)

class SyntheticIdentityGenerator:
    """
    Universal Synthetic Identity Engine (USIE).
    Scenario-driven generator creating multi-modal identities and ground truth labeling.
    Feeds directly into the Data Engine.
    """
    def __init__(self):
        self.face_gen = ProceduralFaceGenerator()
        self.doc_gen = ProceduralDocumentGenerator()
        self.sig_gen = ProceduralSignatureGenerator()
        self.voice_gen = PlaceholderVoiceGenerator()

    def generate(self, scenario: str, count: int, seed: int, output_dir: str, profile: str = "clean"):
        random.seed(seed)
        os.makedirs(output_dir, exist_ok=True)
        
        for mod in ["face", "document", "signature", "voice"]:
            os.makedirs(os.path.join(output_dir, mod), exist_ok=True)
            
        metadata = {}
        
        for i in range(count):
            identity_id = f"synth_{scenario}_{profile}_{i:04d}"
            
            # 1. Persona Generation
            persona = {
                "name": f"Person_{i}",
                "dob": f"199{i%10}-01-01",
                "nationality": "GENERIC"
            }
            
            # Ground Truth
            ground_truth = {
                "duplicate": False,
                "fraud": scenario == "document_fraud" and random.random() < 0.5,
                "same_identity_group": identity_id
            }
            
            # Determine modalities based on scenario
            skip_voice = (scenario == "missing_modalities" and random.random() < 0.3)
            
            # 2. Artifact Generation
            face_path = os.path.join(output_dir, "face", f"{identity_id}.jpg")
            self.face_gen.generate(persona, seed + i, face_path)
            
            doc_path = os.path.join(output_dir, "document", f"{identity_id}.jpg")
            self.doc_gen.generate(persona, "generic_passport", seed + i + 1, doc_path)
            
            sig_path = os.path.join(output_dir, "signature", f"{identity_id}.jpg")
            self.sig_gen.generate(persona["name"], seed + i + 2, sig_path)
            
            if not skip_voice:
                voice_path = os.path.join(output_dir, "voice", f"{identity_id}.wav")
                self.voice_gen.generate("Hello world", seed + i + 3, voice_path)

            # 3. Compile Metadata
            metadata[identity_id] = {
                "identity_id": identity_id,
                "persona": persona,
                "ground_truth": ground_truth
            }

        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Generated Scenario: {scenario} | Profile: {profile} | Seed: {seed}")
        print(f"Faces: {count} | Documents: {count} | Signatures: {count}")
        print(f"Output to: {output_dir}")
