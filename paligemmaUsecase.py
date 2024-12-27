from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch

model_id = "SkalskiP/paligemma2_latex_ocr_v5"
# MODEL_DIR = ""

url = "C:\\Users\\gbsibot-3\\Desktop\\remmitslm\\smolVLM implementation\\images\\Screenshot 2024-12-13 182051.png"
# image = load_image(url)
image = url

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

# model.save_pretrained(MODEL_DIR)

# Leaving the prompt blank for pre-trained models
# prompt = ""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": ("Please extract the following information from the remittance advices and present it in JSON format: "
                   "the company name (it will never be Mettler-Toledo), 9-digit invoice numbers (which may be single or multiple), "
                   "the total amount stated in the document, and the currency used in the document along with individual amounts for each invoice number,the tax amount,Make sure to not bring dummy values , and just the values which are there in the image."
)},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

model_inputs = processor(text=text, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

