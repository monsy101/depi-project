#!/usr/bin/env python3
"""
Streamlit web interface for Stable Diffusion with LoRA support
"""
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from io import BytesIO
from PIL import Image
import os
import time

# Page configuration
st.set_page_config(
    page_title="Depi — Text to Image",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "pipe" not in st.session_state:
    st.session_state.pipe = None
if "lora_available" not in st.session_state:
    st.session_state.lora_available = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

@st.cache_resource
def load_models():
    """Load Stable Diffusion model and LoRA adapter"""
    pipe = None
    lora_available = False
    
    try:
        # Check if model exists locally first
        local_model_paths = [
            "./models",
            "./model",
            os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5")
        ]
        
        model_path = None
        use_local = False
        
        for path in local_model_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "model_index.json")):
                model_path = path
                use_local = True
                break
        
        if use_local:
            st.info(f"📦 Found local model at {model_path}. Loading from disk...")
            status_text = "Loading model from local disk (faster)..."
        else:
            st.warning("🌐 Model not found locally. Downloading from Hugging Face (~4GB, may take 5-15 minutes)...")
            model_path = "sd-legacy/stable-diffusion-v1-5"
            status_text = "Downloading model from Hugging Face... This may take 5-15 minutes depending on your internet speed."
        
        with st.spinner(status_text):
            # Load base model
            try:
                # Try loading with local_files_only first if we found a local model
                if use_local:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=True
                    )
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
            except Exception as e:
                # If local load fails, try downloading
                if use_local:
                    st.warning(f"Local model load failed: {str(e)}. Falling back to download...")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "sd-legacy/stable-diffusion-v1-5",
                        torch_dtype=torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                else:
                    raise
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
            if use_local:
                st.success(f"✅ Base model loaded from local disk on {device.upper()}")
            else:
                st.success(f"✅ Base model downloaded and loaded on {device.upper()}")
            
            # Check for LoRA adapter in multiple possible locations
            lora_paths = [
                "lora/stable_diffusion_finetune_v1",
                "lora_adapter",
                "deployment/models/stable_diffusion_finetune_v1"
            ]
            
            lora_loaded = False
            for lora_path in lora_paths:
                if os.path.exists(lora_path):
                    # Check for different LoRA file formats
                    lora_files = [
                        os.path.join(lora_path, "pytorch_lora_weights.bin"),
                        os.path.join(lora_path, "pytorch_lora_weights.safetensors"),
                        os.path.join(lora_path, "diffusion_pytorch_model.safetensors")
                    ]
                    
                    has_lora_file = any(os.path.exists(f) for f in lora_files)
                    
                    if has_lora_file:
                        try:
                            st.info(f"Loading LoRA adapter from {lora_path}...")
                            
                            # Try PeftModel method first
                            try:
                                pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
                                lora_available = True
                                lora_loaded = True
                                st.success(f"✅ LoRA adapter loaded successfully from {lora_path}!")
                                break
                            except Exception as peft_error:
                                # If PeftModel fails, try load_lora_weights (diffusers method)
                                st.info(f"Trying alternative loading method...")
                                try:
                                    # For diffusers format LoRA
                                    if hasattr(pipe, 'load_lora_weights'):
                                        pipe.load_lora_weights(lora_path)
                                        lora_available = True
                                        lora_loaded = True
                                        st.success(f"✅ LoRA adapter loaded (diffusers format) from {lora_path}!")
                                        break
                                    else:
                                        raise peft_error
                                except Exception as load_error:
                                    # If both methods fail, show the original error
                                    st.warning(f"Failed to load LoRA from {lora_path}: {str(peft_error)}")
                                    continue
                        except Exception as e:
                            st.warning(f"Failed to load LoRA from {lora_path}: {str(e)}")
                            continue
            
            if not lora_loaded:
                st.info("No LoRA adapter found. Using base model only.")
        
        return pipe, lora_available
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

def generate_image(pipe, prompt, negative_prompt="", num_steps=20, guidance_scale=7.5, 
                   width=512, height=512, seed=None):
    """Generate image from prompt"""
    try:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                width=int(width),
                height=int(height),
                generator=generator
            ).images[0]
        
        return image, None
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Depi — Text to Image</h1>', unsafe_allow_html=True)
    st.markdown("### Generate images using Stable Diffusion with LoRA fine-tuning")
    
    # Sidebar for model management
    with st.sidebar:
        st.header("Model Settings")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"**Device:** {device.upper()}")
        
        # Check if models directory exists
        models_exist = os.path.exists("./models") and os.path.exists("./models/model_index.json")
        if models_exist:
            st.success("✅ Local models found!")
        else:
            with st.expander("💡 Tip: Download Model First"):
                st.markdown("""
                **To avoid long download times:**
                
                Run this command in terminal first:
                ```bash
                python prepare_model.py
                ```
                
                This will download the model (~4GB) once, 
                then future loads will be much faster!
                """)
        
        if st.button("Load Model", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                pipe, lora_available = load_models()
                if pipe is not None:
                    st.session_state.pipe = pipe
                    st.session_state.lora_available = lora_available
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
        
        # Model status
        st.markdown("---")
        st.subheader("Model Status")
        if st.session_state.model_loaded:
            st.success("Model Loaded")
            if st.session_state.lora_available:
                st.success("LoRA Adapter Active")
            else:
                st.info("Base Model Only")
        else:
            st.warning("Model Not Loaded")
            st.info("Click 'Load Model' to start")
        
        # Clear cache button
        st.markdown("---")
        if st.button("Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.pipe = None
            st.session_state.model_loaded = False
            st.rerun()
    
    # Main content area
    if not st.session_state.model_loaded:
        st.warning(" Please load the model first using the sidebar button.")
        st.info("""
        **Instructions:**
        1. Click the " Load Model" button in the sidebar
        2. Wait for the model to load (this may take a few minutes)
        3. Once loaded, you can start generating images!
        """)
        
        # Show example prompts
        with st.expander("Example Prompts (click to view)"):
            examples = [
                "abstract geometric patterns in vibrant colors, artistic composition",
                "colorful gradient background with flowing shapes, modern art",
                "minimalist line art, geometric forms, clean design",
                "a beautiful painting of an ancient Egyptian queen, oil painting",
                "futuristic cityscape at sunset, cyberpunk style, neon lights"
            ]
            for i, example in enumerate(examples, 1):
                st.text(f"{i}. {example}")
    else:
        # Create two columns for input and output
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Input Parameters")
            
            # Initialize example prompt state
            if "example_prompt" not in st.session_state:
                st.session_state.example_prompt = "abstract geometric patterns in vibrant colors, artistic composition"
            if "example_negative" not in st.session_state:
                st.session_state.example_negative = ""
            
            # Prompt input
            prompt = st.text_area(
                "Prompt",
                value=st.session_state.example_prompt,
                height=100,
                help="Describe the image you want to generate",
                key="prompt_input"
            )
            
            # Negative prompt
            negative_prompt = st.text_area(
                "Negative Prompt (optional)",
                value=st.session_state.example_negative,
                height=60,
                help="Describe what you want to avoid in the image",
                key="negative_prompt_input"
            )
            
            # Advanced settings
            with st.expander("Advanced Settings", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    num_steps = st.slider(
                        "Inference Steps",
                        min_value=10,
                        max_value=50,
                        value=20,
                        step=5,
                        help="More steps = better quality but slower"
                    )
                    
                    guidance_scale = st.slider(
                        "Guidance Scale",
                        min_value=1.0,
                        max_value=20.0,
                        value=7.5,
                        step=0.5,
                        help="How closely to follow the prompt"
                    )
                
                with col_b:
                    width = st.selectbox(
                        "Width",
                        options=[256, 384, 512, 640, 768, 1024],
                        index=2,
                        help="Image width in pixels"
                    )
                    
                    height = st.selectbox(
                        "Height",
                        options=[256, 384, 512, 640, 768, 1024],
                        index=2,
                        help="Image height in pixels"
                    )
                
                seed = st.number_input(
                    "Seed (optional)",
                    min_value=None,
                    max_value=None,
                    value=None,
                    help="Set a seed for reproducible results. Leave empty for random."
                )
            
            # Generate button
            generate_button = st.button(
                "Generate Image",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            st.header("Generated Image")
            
            if generate_button:
                if not prompt.strip():
                    st.error("Please enter a prompt!")
                else:
                    with st.spinner("Generating image... This may take 30-60 seconds."):
                        start_time = time.time()
                        image, error = generate_image(
                            st.session_state.pipe,
                            prompt,
                            negative_prompt,
                            num_steps,
                            guidance_scale,
                            width,
                            height,
                            seed
                        )
                        generation_time = time.time() - start_time
                        
                        if error:
                            st.error(f"Generation failed: {error}")
                        elif image:
                            st.image(image, caption=f"Generated in {generation_time:.1f}s", use_container_width=True)
                            
                            # Save to session state
                            st.session_state.generated_images.append({
                                "image": image,
                                "prompt": prompt,
                                "time": generation_time,
                                "timestamp": time.time()
                            })
                            
                            # Download button
                            buf = BytesIO()
                            image.save(buf, format="PNG")
                            st.download_button(
                                "💾 Download Image",
                                data=buf.getvalue(),
                                file_name=f"generated_{int(time.time())}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            
                            # Show generation info
                            st.info(f"""
                            **Generation Info:**
                            - Steps: {num_steps}
                            - Guidance Scale: {guidance_scale}
                            - Size: {width}x{height}
                            - Time: {generation_time:.1f}s
                            - Seed: {seed if seed is not None else 'Random'}
                            """)
            
            # Show recent images
            if st.session_state.generated_images:
                st.markdown("---")
                st.subheader("Recent Generations")
                
                # Show last 3 images
                for i, img_data in enumerate(reversed(st.session_state.generated_images[-3:]), 1):
                    with st.expander(f"Image {i} - {img_data['prompt'][:50]}..."):
                        st.image(img_data["image"], use_container_width=True)
                        st.caption(f"Generated in {img_data['time']:.1f}s")
                        buf = BytesIO()
                        img_data["image"].save(buf, format="PNG")
                        st.download_button(
                            f"Download Image {i}",
                            data=buf.getvalue(),
                            file_name=f"generated_{i}_{int(img_data['timestamp'])}.png",
                            mime="image/png",
                            key=f"download_{i}"
                        )
        
        # Example prompts section
        st.markdown("---")
        st.header("Example Prompts")
        
        example_cols = st.columns(3)
        examples = [
            ("Abstract Art", "abstract geometric patterns in vibrant colors, artistic composition", ""),
            ("Modern Art", "colorful gradient background with flowing shapes, modern art", "blurry, low quality"),
            ("Minimalist", "minimalist line art, geometric forms, clean design", "complex, busy"),
            ("Egyptian Style", "a beautiful painting of an ancient Egyptian queen, oil painting", ""),
            ("Cyberpunk", "futuristic cityscape at sunset, cyberpunk style, neon lights", ""),
            ("Nature", "serene mountain landscape at dawn, misty atmosphere, photorealistic", "cartoon, illustration")
        ]
        
        for i, (title, example_prompt, example_negative) in enumerate(examples):
            with example_cols[i % 3]:
                st.markdown(f"**{title}**")
                if st.button(f"Use This Prompt", key=f"example_{i}", use_container_width=True):
                    st.session_state.example_prompt = example_prompt
                    st.session_state.example_negative = example_negative
                    st.rerun()

if __name__ == "__main__":
    main()
