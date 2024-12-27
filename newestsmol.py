import os
import logging
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForVision2Seq
import ast
from PIL import Image
from io import BytesIO

import fitz  # PyMuPDF

class PDFInvoiceProcessor:
    def __init__(self, model_name="mjschock/SmolVLM-Instruct-SFT-LaTeX_OCR", 
                 pdf_folder=None, output_csv="invoice_extraction_results.csv"):
        """
        Initialize PDF Invoice Processor
        
        :param model_name: Hugging Face model name
        :param pdf_folder: Folder containing PDF files
        :param output_csv: Output CSV filename
        """
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="pdf_invoice_processing.log"
        )
        
        # Device configuration
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Running on {self.DEVICE}")
        
        # Model and processor directories
        self.MODEL_DIR = "./smolVLMmodel"
        self.PROCESSOR_DIR = "./smolVLMproc"
        
        # Input and output configurations
        self.pdf_folder = pdf_folder or r"C:\\Users\\gbsibot-3\\Desktop\\remmitslm\\smolVLM implementation\\images\\"
        self.output_csv = output_csv
        
        # Load model and processor
        self.load_model_and_processor(model_name)
    
    def load_model_and_processor(self, model_name):
        """Load model and processor with local caching"""
        try:
            # Processor loading
            if os.path.exists(self.PROCESSOR_DIR):
                self.processor = AutoProcessor.from_pretrained(self.PROCESSOR_DIR)
                logging.info("Processor loaded from local")
            else:
                self.processor = AutoProcessor.from_pretrained(model_name)    
                self.processor.save_pretrained(self.PROCESSOR_DIR)
            
            # Model loading
            if os.path.exists(self.MODEL_DIR):
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.MODEL_DIR,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="eager",
                )
                logging.info("Model loaded from local directory.")
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="eager"
                )
                self.model.save_pretrained(self.MODEL_DIR)
            
            self.model.to(self.DEVICE)
        
        except Exception as e:
            logging.error(f"Failed to load model or processor: {e}")
            raise
    
    def process_pdf(self, pdf_path):
        """
        Process a single PDF file
        
        :param pdf_path: Path to PDF file
        :return: Extracted invoice information
        """
        try:
            # Open the PDF file using PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Store results for multi-page PDFs
            all_extractions = []
            
            for idx, page in enumerate(doc):
                try:
                    # Render the page to a Pixmap
                    pixmap = page.get_pixmap()
                    
                    # Check for valid dimensions
                    if pixmap.width <= 0 or pixmap.height <= 0:
                        logging.error(f"Page {idx+1} of {pdf_path} has invalid dimensions.")
                        continue
                    
                    # Convert Pixmap to PIL Image
                    image_bytes = pixmap.tobytes()  # Use tobytes() instead of getImageData
                    image = Image.open(BytesIO(image_bytes)).convert('RGB')
                except Exception as e:
                    logging.error(f"Failed to convert page {idx+1} of {pdf_path}: {e}")
                    continue
                
                # Prepare image for processing
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
                                    "the total amount stated in the document, and the currency used in the document along with individual amounts for each invoice number, the tax amount, Make sure to not bring dummy values, and just the values which are there in the image."
                                )
                            }
                        ]
                    },
                ]
                
                try:
                    prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.DEVICE)
                except Exception as e:
                    logging.error(f"Failed to prepare inputs for page {idx+1}: {e}")
                    continue
                
                try:
                    generated_ids = self.model.generate(**inputs, max_new_tokens=500)
                    generated_texts = self.processor.batch_decode(
                    generated_ids , skip_special_tokens = True
                    )

                    extracted_text = generated_texts[0]
                    output = extracted_text

                    start_index = output.find("Assistant: {")
                    if start_index != -1:
                        extracted_output = output[start_index:]
                        print(extracted_output)
                        start_index
                        parsed_output = ast.literal_eval(json_part.replace(":", ": ").replace(",", ", ").replace("'", "\""))
                        parsed_output['page'] = idx + 1
                    
                        all_extractions.append(parsed_output)
                    else:
                        print("the specified substring was not found.")

                    # generated_ids = self.model.generate(**inputs, max_new_tokens=500)
                    # generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # # Extract JSON part
                    # output = generated_texts[0]
                    # start_index = output.find("{")
                    # end_index = output.rfind("}") + 1
                    # json_part = output[start_index:end_index]
                    
                    # # Parse JSON
                    # parsed_output = ast.literal_eval(json_part.replace(":", ": ").replace(",", ", ").replace("'", "\""))
                    # parsed_output['page'] = idx + 1
                    
                    # all_extractions.append(parsed_output)
                
                except Exception as e:
                    logging.error(f"Error processing page {idx+1}: {e}")
            
            return all_extractions
        
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def process_folder(self):
        """
        Process all PDFs in the specified folder
        """
        all_results = []
        
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder, filename)
                
                try:
                    pdf_results = self.process_pdf(pdf_path)
                    
                    # Prepare results with filename
                    for result in pdf_results:
                        result['filename'] = filename
                    
                    all_results.extend(pdf_results)
                    
                except Exception as e:
                    logging.error(f"Failed to process {filename}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Handle multi-page PDF consolidation
        if not df.empty:
            # Group and aggregate results
            df['invoice_numbers'] = df['invoice_numbers'].apply(lambda x: '|'.join(map(str, x)) if isinstance(x, list) else x)
            
            # Save to CSV
            df.to_csv(self.output_csv, index=False)
            logging.info(f"Results saved to {self.output_csv}")
        
        return df

def main():
    processor = PDFInvoiceProcessor(
        pdf_folder=r"C:\\Users\\gbsibot-3\\Desktop\\remmitslm\\smolVLM implementation\\images\\",
        output_csv="remmit_extraction.csv"
    )
    processor.process_folder()

if __name__ == "__main__":
    main()