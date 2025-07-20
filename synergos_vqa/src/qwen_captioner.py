import os
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface_cache"

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import numpy as np
import torch
import re

logger = logging.getLogger(__name__)

MAX_TOTAL_TOKENS = 12288
MAX_PROMPT_NUM = 5

class InternVLCaptioner:
    def __init__(self, model_path="OpenGVLab/InternVL3-8B", device="cuda", max_retries=3):
        self.max_retries = max_retries
        self.model_path = model_path
        self.device = device
        self.generation_config = {
            "max_new_tokens": 120,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.92,
            "repetition_penalty": 1.2
        }
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Initializing InternVLCaptioner with model: {model_path} (Attempt {attempt + 1}/{self.max_retries})")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
                logger.info("InternVLCaptioner initialized successfully")
                return
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    logger.info(f"Waiting {wait_time} seconds before next attempt...")
                    time.sleep(wait_time)
                else:
                    logger.error("All attempts failed to initialize InternVLCaptioner")
                    raise

    def _prompt_templates(self, question):
        return [
            f"Describe the overall scene and setting to help answer: '{question}'. Focus on key visual elements and their relationships. Use 1-2 concise sentences.",
            f"Identify and describe the key objects, their attributes, and actions in the image for the question: '{question}'. Be specific about details that might help answer the question. Use 1-2 short sentences.",
            f"Provide any relevant reasoning, relationships, or background knowledge that could help answer: '{question}'. Consider both explicit and implicit information in the image. Use 1-2 concise sentences.",
            f"Based on the image and previous descriptions, what are the 3 most likely answers to: '{question}'? Start with 'The answer may be' and list them as specific words or short phrases separated by ' | '. For example, if the question is 'What color is the car?', respond with 'The answer may be red | blue | black'. Be direct and concise."
        ]

    def _preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            image = image.crop((left, top, left + min_side, top + min_side))
            image = image.resize((448, 448))
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise

    def generate_prompts(self, image_path, question, num_prompts=4):
        templates = self._prompt_templates(question)
        prompts = templates[:num_prompts]
        return prompts

    def postprocess_caption(self, caption):
        for phrase in [
            "Based on the image,", "In the image,", "The image shows", "The image depicts",
            "This image shows", "This image depicts", "Looking at the image,", "From the image,"
        ]:
            if caption.strip().startswith(phrase):
                caption = caption[len(phrase):].strip()
        
        caption = re.sub(r'\s+([.,!?])', r'\1', caption)
        caption = re.sub(r'([.,!?])\s+', r'\1 ', caption)
        
        if not caption[-1] in '.!?':
            caption = caption + '.'
            
        return caption.strip()

    def generate_caption(self, image_path, prompt):
        try:
            image_tensor = self._preprocess_image(image_path)
            response = self.model.chat(
                self.tokenizer, image_tensor, prompt, generation_config=self.generation_config
            )
            caption = self.postprocess_caption(response)
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return ""