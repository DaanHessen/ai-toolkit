import gradio as gr
import torch
import os
import uuid
import shutil
from PIL import Image
from lose_train import train_lose
from toolkit.models.lose_adapter import LoSEAdapter, patch_flux_with_lose
from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein9BModel
from toolkit.config_modules import ModelConfig

# --- Utilities ---

def extract_face_embed(image_path, app):
    # Non-differentiable face extraction for inference/preprocessing
    import cv2
    import numpy as np
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    # Return the first face's embedding
    return torch.from_numpy(faces[0].normed_embedding).float()

class LoseInference:
    def __init__(self):
        self.flux_model = None
        self.lose_adapter = None
        self.face_analysis = None

    def load_models(self, flux_path, lose_path):
        # Load FLUX Klein 9B
        m_config = ModelConfig(name_or_path=flux_path, arch="flux2_klein_9b")
        self.flux_model = Flux2Klein9BModel(device="cuda", model_config=m_config)
        self.flux_model.load_model()
        
        # Load LoSE Adapter
        self.lose_adapter = LoSEAdapter(self.flux_model.unet)
        self.lose_adapter.load_weights(lose_path)
        self.lose_adapter.to("cuda")
        
        # Patch Flux
        patch_flux_with_lose(self.flux_model.unet, self.lose_adapter)
        
        # Load InsightFace for live ID extraction
        from insightface.app import FaceAnalysis
        self.face_analysis = FaceAnalysis(name='buffalo_l', root='./insightface_models')
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    def generate(self, ref_image, prompt, steps=20, guidance=3.5):
        if self.face_analysis is None:
            return "Models not loaded."
            
        # 1. Extract ID
        temp_path = "temp_ref.jpg"
        ref_image.save(temp_path)
        id_embed = extract_face_embed(temp_path, self.face_analysis)
        if id_embed is None:
            return "No face detected in reference image."
        id_embed = id_embed.unsqueeze(0).to("cuda")
        
        # 2. Run Inference
        # This is a simplified call - would use the toolkit's pipeline in production
        with torch.no_grad():
            output_image = self.flux_model.pipeline(
                prompt=prompt,
                id_embed=id_embed, # Passed to patched transformer
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=512,
                width=512
            ).images[0]
        return output_image

# --- UI Layout ---

inf_engine = LoseInference()

with gr.Blocks(title="LoSE: Load ID-Residual Adapter for FLUX.2") as demo:
    gr.Markdown("# LoSE: Load ID-Residual Adapter for FLUX.2")
    gr.Markdown("Custom identity preservation for FLUX.2 Klein 9B via Facial Identity Loss.")
    
    with gr.Tabs():
        # --- Train Tab ---
        with gr.TabItem("Train"):
            with gr.Row():
                with gr.Column():
                    dataset_images = gr.File(label="Upload 512x512 Face Crops", file_count="multiple")
                    arcface_model = gr.Textbox(label="ArcFace Model Path", value="./models/buffalo_l/model.safetensors")
                    flux_base = gr.Textbox(label="Base FLUX Path", value="black-forest-labs/FLUX.1-schnell")
                with gr.Column():
                    lr = gr.Number(label="Learning Rate", value=1e-4)
                    steps = gr.Number(label="Steps", value=1000)
                    batch = gr.Number(label="Batch Size", value=1)
                    train_btn = gr.Button("Start Training", variant="primary")
            
            output_log = gr.Textbox(label="Training Status")
            
            def run_train(images, arcface, base, lr_val, steps_val, batch_val):
                # Create a temporary dataset folder
                train_id = str(uuid.uuid4())
                dset_path = f"datasets/{train_id}"
                os.makedirs(dset_path, exist_ok=True)
                for img in images:
                    shutil.copy(img.name, dset_path)
                
                # Call training function
                train_lose(
                    model_path=base,
                    dataset_path=dset_path,
                    arcface_path=arcface,
                    lr=lr_val,
                    steps=int(steps_val),
                    batch_size=int(batch_val)
                )
                return "Training Complete! LoSE adapter saved to ./output_lose"

            train_btn.click(run_train, inputs=[dataset_images, arcface_model, flux_base, lr, steps, batch], outputs=output_log)

        # --- Inference Tab ---
        with gr.TabItem("Inference"):
            with gr.Row():
                with gr.Column():
                    model_load_base = gr.Textbox(label="Base FLUX Path", value="black-forest-labs/FLUX.1-schnell")
                    model_load_lose = gr.Textbox(label="LoSE Adapter Path", value="./output_lose/lose_final.safetensors")
                    load_btn = gr.Button("Load Models")
                    
                    ref_id_img = gr.Image(label="Reference Face Image", type="pil")
                    prompt = gr.Textbox(label="Prompt", placeholder="A photo of the person in a space suit")
                    gen_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output_img = gr.Image(label="Generated Result")

            def load_wrapper(base, lose):
                inf_engine.load_models(base, lose)
                return "Models Loaded Successfully!"

            load_btn.click(load_wrapper, inputs=[model_load_base, model_load_lose], outputs=output_log)
            gen_btn.click(inf_engine.generate, inputs=[ref_id_img, prompt], outputs=output_img)

if __name__ == "__main__":
    demo.launch(share=True)
