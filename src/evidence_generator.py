# src/evidence_generator.py
import torch
import logging
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, DetrImageProcessor, DetrForObjectDetection
from scipy.spatial.distance import cdist
import numpy as np

logger = logging.getLogger(__name__)


class SynergisticEvidenceGenerator:
    def __init__(self, mllm_path="OpenGVLab/InternVL3-8B", detr_path="facebook/detr-resnet-50", device="cuda"):
        """
        Initializes a unified generator for all three evidence types.
        """
        self.device = device

        # Load the powerful MLLM for text generation
        print(f"Loading MLLM from: {mllm_path}")
        self.mllm_tokenizer = AutoTokenizer.from_pretrained(mllm_path, trust_remote_code=True)
        self.mllm = AutoModelForCausalLM.from_pretrained(mllm_path, trust_remote_code=True).to(device).eval()
        self.mllm_generation_config = {"max_new_tokens": 120, "do_sample": True, "temperature": 1.0, "top_p": 0.92}

        # Load the DETR model for object detection
        print(f"Loading DETR from: {detr_path}")
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_path)
        self.detr_model = DetrForObjectDetection.from_pretrained(detr_path).to(device).eval()

        logger.info(f"SynergisticEvidenceGenerator initialized successfully.")

    def _generate_text_from_image(self, image, prompt):
        """Helper function to run the MLLM."""
        try:
            with torch.no_grad():
                response = self.mllm.chat(self.mllm_tokenizer, image, prompt,
                                          generation_config=self.mllm_generation_config)
            return response.strip()
        except Exception as e:
            logger.error(f"Error during MLLM generation: {e}")
            return ""

    def generate_holistic_evidence(self, image, question):
        """
        Generates holistic evidence describing the overall scene[cite: 79].
        """
        prompt = f"Concisely describe the overall scene and setting in the image to help answer the question: '{question}'. Focus on key visual elements and their relationships."
        holistic_evidence = self._generate_text_from_image(image, prompt)
        return holistic_evidence

    def generate_structural_and_causal_evidence(self, image, question, prototype_library):
        """
        MERGED FUNCTION: Generates both Structural (Prototype CoT) and Causal evidence.
        """
        prototype_cot = "No distinct objects identified."
        causal_evidence = "Causal analysis could not be performed due to a lack of structural elements."

        try:
            # --- Structural Evidence Generation ---
            # 1. Detect objects using DETR [cite: 82]
            inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.detr_model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1][0]  # Features for all queries
                probas = outputs.logits.softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.9  # Confidence threshold

            if torch.sum(keep) > 0:
                object_features = last_hidden_state[keep, :].cpu().numpy()

                # 2. Match features to prototypes in the library [cite: 82]
                prototypes = prototype_library["prototypes"]
                cluster_to_label = prototype_library["cluster_to_label"]

                # Find the closest prototype for each detected object feature (using cosine distance)
                distances = cdist(object_features, prototypes, 'cosine')
                closest_prototype_indices = np.argmin(distances, axis=1)

                # 3. Form the Prototype CoT [cite: 82]
                detected_labels = [cluster_to_label[idx] for idx in closest_prototype_indices]
                prototype_cot = " -> ".join(sorted(list(set(detected_labels))))  # Get unique, sorted labels

            # --- Causal Evidence Generation ---
            # 4. Use the generated Prototype CoT to form a counterfactual prompt [cite: 83, 84]
            causal_prompt = f"Given the identified elements '{prototype_cot}', what is the most critical element for answering '{question}'? Provide a brief counterfactual analysis, for example: 'If the {detected_labels[0] if detected_labels else 'main object'} were different, the answer would change.'"
            causal_evidence = self._generate_text_from_image(image, causal_prompt)

        except Exception as e:
            logger.error(f"Error during structural/causal generation: {e}")

        return prototype_cot, causal_evidence

    def generate_all_evidence(self, image_path, question, prototype_library):
        """
        A convenient top-level method that generates all three evidence streams.
        This is the main function to be called from your training/testing loop.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Could not open image {image_path}: {e}")
            return {"holistic": "", "structural": "", "causal": ""}

        # Generate Holistic evidence
        holistic = self.generate_holistic_evidence(image, question)

        # Generate Structural and Causal evidence together
        structural, causal = self.generate_structural_and_causal_evidence(image, question, prototype_library)

        return {
            "holistic": holistic,
            "structural": structural,
            "causal": causal,
        }