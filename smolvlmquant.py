import torch
import os
import logging
from transformers import AutoProcessor, AutoModelForVision2Seq ,BitsAndBytesConfig
from transformers.image_utils import load_image
# Configure Logging
logging.basicConfig(
   filename="invoice_processing.log",
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s"
)
# Device Selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
logging.info(f"Running on device: {DEVICE}")
# Model and Processor Paths
MODEL_DIR = "./smolVLMmodelquant"
PROCESSOR_DIR = "./smolVLMprocquant"
# Initialize Processor
try:
   if os.path.exists(PROCESSOR_DIR):
       processor = AutoProcessor.from_pretrained(PROCESSOR_DIR)
       logging.info("Processor loaded from local directory.")
   else:
       processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
       processor.save_pretrained(PROCESSOR_DIR)
       logging.info("Processor downloaded and saved locally.")
except Exception as e:
   logging.error(f"Failed to load processor: {e}")
   raise
# Initialize Model
try:
   if os.path.exists(MODEL_DIR):
       model = AutoModelForVision2Seq.from_pretrained(
           MODEL_DIR,
        #    torch_dtype=torch.bfloat16,
           _attn_implementation="eager",
        #    quantization_config=quantization_config,
           low_cpu_mem_usage =True,
       )
       logging.info("Model loaded from local directory.")
       print(model.device)
   else:
       model = AutoModelForVision2Seq.from_pretrained(
           "HuggingFaceTB/SmolVLM-Instruct",
        #    torch_dtype=torch.bfloat16,
           _attn_implementation="eager",
           quantization_config=quantization_config,
           low_cpu_mem_usage =True,
       )
       model.save_pretrained(MODEL_DIR)
       logging.info("Model downloaded and saved locally.")
#    model.to(DEVICE)
except Exception as e:
   logging.error(f"Failed to load model: {e}")
   raise
# Load the test image (consider making this dynamic)
image_path = r""
try:
   image1 = load_image(image_path)
   logging.info(f"Loaded image from {image_path}")
except Exception as e:
   logging.error(f"Failed to load image: {e}")
   raise
# Create input messages
# messages = [
#    {
#        "role": "user",
#        "content": [
#            {"type": "image"},
#            {
#                "type": "text",
#                "text": (
#                    "Please extract the following information from the remittance advices and present it in JSON format: "
#                    "the company name (it will never be ), 9-digit invoice numbers (which may be single or multiple), "
#                    "the total amount stated in the document, and the currency used in the document along with individual amounts for each invoice number."
#                )
#            }
#        ]
#    },
# ]
messages = [
   {
       "role": "user",
       "content": [
           {"type": "image"},
           {
               "type": "text",
               "text": (
                   "You are an intelligent document extraction assistant. Analyze the provided remittance advice image and extract "
                   "the following information in JSON format, ensuring accuracy and completeness:\n\n"
                   "- **Company Name:** Extract the name of the company making the payment (it will never be '').\n"
                   "- **Invoice Numbers:** Extract all 9-digit invoice numbers (there may be one or multiple invoices).\n"
                   "- **Total Amount:** Extract the total payment amount from the document.\n"
                   "- **Currency:** Identify the currency used in the document.\n"
                   "- **Invoice Details:** For each extracted invoice number, provide its corresponding payment amount.\n\n"
                   "Format the output as a JSON object like this example:\n\n"
                   "{\n"
                   "  'company_name': 'ABC Corporation',\n"
                   "  'invoice_numbers': ['123456789', '987654321'],\n"
                   "  'total_amount': '1500.00',\n"
                   "  'currency': 'USD',\n"
                   "  'invoices': [\n"
                   "    {'invoice_number': '123456789', 'amount': '1000.00'},\n"
                   "    {'invoice_number': '987654321', 'amount': '500.00'}\n"
                   "  ]\n"
                   "}\n\n"
                   "Ensure that all values are extracted exactly as they appear on the remittance advice and use proper data types where applicable."
               )
           }
       ]
   }
]
# Prepare Inputs
try:
   prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
   inputs = processor(text=prompt, images=[image1], return_tensors="pt").to(DEVICE)
except Exception as e:
   logging.error(f"Failed to prepare inputs: {e}")
   raise
# Generate Outputs
try:
   generated_ids = model.generate(**inputs, max_new_tokens=500)
   generated_texts = processor.batch_decode(
       generated_ids, skip_special_tokens=True
   )
   # Print and log the result

   extracted_text = generated_texts[0]
   output =extracted_text
   # Find the index of "Assistant: {"
   start_index = output.find("Assistant: {")
   if start_index != -1:
        # Extract the substring starting from "Assistant: {"
     extracted_output = output[start_index:]
     print(extracted_output)
   else:
       print("The specified substring was not found.")
#    print(extracted_text)
   logging.info(f"Extraction successful: {extracted_output}")
except Exception as e:
   logging.error(f"Error during extraction: {e}")
