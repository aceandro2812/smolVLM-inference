import torch
import os
import logging
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


logging.basicConfig(
   filename="invoice_processing.log",
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s"
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.info(f"Running on this {DEVICE}")

MODEL_DIR = "./smolVLMmodel"

PROCESSOR_DIR = "./smolVLMproc"

try:
    if os.path.exists(PROCESSOR_DIR):
        processor = AutoProcessor.from_pretrained(PROCESSOR_DIR)
        logging.info("Processor loaded from local ")
    else:
        processor = AutoProcessor.from_pretrained("mjschock/SmolVLM-Instruct-SFT-LaTeX_OCR")    
        processor.save_pretrained(PROCESSOR_DIR)

except Exception as e:
    logging.error(f"Failed to load processor model{e}")
    raise

try:
    if os.path.exists(PROCESSOR_DIR):
        processor = AutoProcessor.from_pretrained(PROCESSOR_DIR)
        logging.info("Processor loaded from local ")
    else:
        processor = AutoProcessor.from_pretrained("mjschock/SmolVLM-Instruct-SFT-LaTeX_OCR")    
        processor.save_pretrained(PROCESSOR_DIR)

except Exception as e:
    logging.error(f"Failed to load processor model{e}")
    raise


try:
   if os.path.exists(MODEL_DIR):
       
       model = AutoModelForVision2Seq.from_pretrained(
           MODEL_DIR,
           torch_dtype=torch.bfloat16,
           _attn_implementation="eager",
       )
       logging.info("Model loaded from local directory.")
   else:
       model = AutoModelForVision2Seq.from_pretrained("mjschock/SmolVLM-Instruct-SFT-LaTeX_OCR",torch_dtype=torch.bfloat16,_attn_implementation="eager")
      #  model = PeftModel.from_pretrained(base_model, "HuggingFaceTB/SmolVLM-Instruct-DPO",torch_dtype=torch.bfloat16,_attn_implementation="eager",)
      #  model = AutoModelForVision2Seq.from_pretrained(
      #      "HuggingFaceTB/SmolVLM-Instruct-DPO",
      #      torch_dtype=torch.bfloat16,
      #      _attn_implementation="eager",
      #  )
       model.save_pretrained(MODEL_DIR)
       logging.info("Model downloaded and saved locally.")
   model.to(DEVICE)
except Exception as e:
   logging.error(f"Failed to load model: {e}")
   raise



image_folder = r"C:\\Users\\gbsibot-3\\Desktop\\remmitslm\\smolVLM implementation\\images\\"

for filename in os.listdir(image_folder):
    if filename.endswith(('.png','.jpg','.jpeg')):
        image_path = os.path.join(image_folder,filename)
        print(f"Founde Image: {image_path}")

        try:
          image1 = load_image(image_path)
          logging.info(f"Loaded image from {image_path}")
        except Exception as e:
            logging.error(f"Failed to load image: {e}")
            raise
# Create input messages
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Please extract the following information from the remittance advices and present it in JSON format: "
                        "the company name (it will never be Mettler-Toledo), 9-digit invoice numbers (which may be single or multiple), "
                        "the total amount stated in the document, and the currency used in the document along with individual amounts for each invoice number,the tax amount,Make sure to not bring dummy values , and just the values which are there in the image."
                    )
                }
            ]
        },
        ]

        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt , images = [image1] , return_tensors = "pt").to(DEVICE)
        except Exception as e:
            logging.error(f"Failed to prepare inputs: {e}")
            raise
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(
                generated_ids , skip_special_tokens = True
            )

            extracted_text = generated_texts[0]
            output = extracted_text

            start_index = output.find("Assistant: {")
            if start_index != -1:
                extracted_output = output[start_index:]
                print(extracted_output)
            else:
                print("the specified substring was not found.")

            logging.info(f"Extraction Successfull : {extracted_output}")

        
        except Exception as e:
            logging.error(f"Error during extarction: {e}")
        try:
            # Clean up the extracted_output to isolate the JSON part
            start_index = extracted_output.find("{")
            end_index = extracted_output.rfind("}") + 1  # Include the closing brace
            json_part = extracted_output[start_index:end_index]

            # Now manually parse the json_part string to extract needed values
            import ast

            # Use ast.literal_eval to safely evaluate the string representation of the dictionary
            parsed_output = ast.literal_eval(json_part.replace(":", ": ").replace(",", ", ").replace("'", "\""))

            # Extract each field into variables
            company = parsed_output["company"]
            invoice_numbers = parsed_output["invoice_numbers"]
            total_amount = parsed_output["total_amount"]
            currency = parsed_output["currency"]

            # Print the extracted variables
            print(f"Company: {company}")
            print(f"Invoice Numbers: {invoice_numbers}")
            print(f"Total Amount: {total_amount}")
            print(f"Currency: {currency}")
            
        except Exception as e:
            logging.error(f"Error during csv posting: {e}")


        