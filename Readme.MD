# Project Readme

This repository contains several Python scripts focused on leveraging Vision Language Models (VLMs) for various tasks, including processing PDF invoices, interacting with large language models, and implementing different VLM architectures.

## File Descriptions and Usage

Below is a brief description of each file and instructions on how to use them:

### `newestsmol.py`

**Purpose:** Processes PDF invoices using a specified VLM model (`mjschock/SmolVLM-Instruct-SFT-LaTeX_OCR` by default). It extracts information from the PDF files by converting them into images and then applying the VLM for text and data extraction.

**Usage:**

1.  **Initialization:** Create an instance of the `PDFInvoiceProcessor` class. You can specify a different model name if needed.
    ```python
    processor = PDFInvoiceProcessor(model_name="your_model_name")
    ```
2.  **Process a single PDF:**
    ```python
    processor.process_pdf("path/to/your/invoice.pdf")
    ```
3.  **Process a folder of PDFs:** Place all PDF invoices in a folder named `invoices` in the same directory as the script.
    ```python
    processor.process_folder()
    ```
4.  **Run from the command line:**
    ```bash
    python newestsmol.py
    ```
    This will process all PDFs in the `invoices` folder.

### `paligemmaUsecase.py`

**Purpose:** This script likely demonstrates the use case of the PaliGemma model, potentially for image captioning, visual question answering, or other vision-language tasks.

**Usage:**

-   The exact usage would depend on the implementation details, but generally, you would initialize the model, load an image, and use the model to generate text based on the image content.
-   Refer to the internal documentation or comments within the script for detailed instructions.

### `qwenimplementation.py`

**Purpose:** This script likely contains an implementation or a usage example of the Qwen model, which is a large language model.

**Usage:**

-   Similar to `paligemmaUsecase.py`, the specifics of usage will be defined within the script.
-   Likely involves initializing the Qwen model, providing text input, and obtaining generated text output.
-   Consult the script's internal documentation for detailed usage instructions.

### `smolmutlipdf.py`

**Purpose:** This script probably handles multiple PDF files for processing with a VLM model, similar to `newestsmol.py` but possibly with additional features or a different approach to handling multiple files.

**Usage:**

-   Likely involves placing PDF files in a specific directory and then running the script to process them all.
-   May offer options to configure the VLM model being used.
-   Detailed usage instructions should be available within the script's documentation or comments.

### `smolvlm2.py`

**Purpose:** Likely an iteration or a different version of a VLM implementation, possibly with enhancements or changes compared to the model used in `newestsmol.py`.

**Usage:**

-   The usage pattern will probably be similar to other VLM-related scripts, focusing on image/PDF processing and text generation.
-   Check the script's internal documentation for specific instructions.

### `smolvlmquant.py`

**Purpose:** This script likely deals with a quantized version of a SmolVLM model. Quantization reduces the model's size and computational requirements, potentially at the cost of some accuracy.

**Usage:**

-   Probably involves loading a quantized model, processing images or PDFs, and generating text output.
-   May offer options to adjust quantization parameters or compare performance with the non-quantized version.
-   Refer to the internal documentation for specific usage instructions.

### `vllmsmol.py`

**Purpose:** This script likely integrates a SmolVLM model with VLLM (a fast and easy-to-use library for LLM inference and serving). It enables efficient deployment and serving of the SmolVLM model.

**Usage:**

1.  Ensure VLLM is installed and set up according to its documentation.
2.  The script likely provides functions or a command-line interface to load the SmolVLM model using VLLM.
3.  You can then send requests to the VLLM server for inference on images or PDFs.
4.  Refer to the script's internal documentation for detailed instructions on setup and usage.

## General Notes

-   Make sure to install the required dependencies for each script. A `requirements.txt` file is likely provided in the repository.
-   Most scripts probably accept command-line arguments for configuration. Use `python script_name.py --help` to see available options.
-   For scripts involving model loading, ensure you have sufficient memory (RAM and VRAM) available.
-   Some scripts that process entire folders may have issues with sub-folders, or special character encoding.

This `Readme.MD` provides a starting point for understanding and using your code. Remember to update it if any of the scripts are modified. Let me know when you have a specific request!
# SmolVLM2: Vision Language Model for Document Processing

`smolvlm2.py` is a Python script that utilizes a Vision Language Model (VLM) to extract structured information from images, particularly designed for processing remittance advice documents. It leverages the Hugging Face `transformers` library to load a pre-trained model and processor for performing vision-to-sequence tasks.

## Features

-   Loads a specified VLM model and processor (default: `mjschock/SmolVLM-Instruct-SFT-LaTeX_OCR`).
-   Supports local caching of the model and processor for offline use.
-   Processes images (e.g., scanned remittance advices) to extract key information.
-   Outputs the extracted information in JSON format.
-   Utilizes GPU acceleration if available, falling back to CPU otherwise.
-   Logs operations and errors to `invoice_processing.log`.

## Prerequisites

Before running `smolvlm2.py`, ensure you have the following installed:

-   Python 3.x
-   PyTorch (with CUDA support if you have a compatible GPU)
-   Hugging Face `transformers` library
-   `Pillow` (PIL) for image loading

You can install the required Python packages using `pip`:

```bash
pip install torch torchvision torchaudio transformers Pillow
```

## Installation

1.  Clone the repository or download the `smolvlm2.py` script.
2.  Navigate to the directory containing `smolvlm2.py` in your terminal.

## Usage

### 1. Configuration (Optional)

-   **Model and Processor:** By default, the script loads the `mjschock/SmolVLM-Instruct-SFT-LaTeX_OCR` model and its corresponding processor. You can change this by modifying the `MODEL_DIR` and `PROCESSOR_DIR` variables in the script to point to a different local model or HuggingFace model id.
-   **Image Path:** The script is configured to process a specific image defined by the `image_path` variable. Update this variable within `smolvlm2.py` to point to the image you want to process.

### 2. Running the Script

Execute the script from the command line:

```bash
python smolvlm2.py
```

### 3. Input Image

The script is designed to process a single image provided by the path in the `image_path` variable.
It defaults to a specific path `r"C:\\Users\\gbsibot-3\\Desktop\\remmitslm\\smolVLM implementation\\images\\Screenshot 2024-12-13 182051.png"`.
Make sure to replace the hardcoded path in the file `smolvlm2.py` with the actual path to your image file.

### 4. Prompt Customization
The core of the extraction process is driven by the prompt defined in the `messages` variable.
This prompt instructs the model on what information to extract and how to format it.
Modify the prompt to tailor the extraction to your specific needs.
Ensure to properly format the prompt using the format expected by the model.
Default Prompt:
```python
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
```
### 5. Output

The script will output the extracted information to the console. The output will be formatted as a JSON string, starting with "Assistant: {". Here's an example:

```
Assistant: {
    "company_name": "Example Company",
    "invoice_numbers": [
        "123456789",
        "987654321"
    ],
    "total_amount": "1234.56",
    "currency": "USD",
    "invoice_details": [
        {
            "invoice_number": "123456789",
            "amount": "617.28"
        },
        {
            "invoice_number": "987654321",
            "amount": "617.28"
        }
    ],
    "tax_amount": "100.00"
}
```

The script also logs information and errors to `invoice_processing.log`.

### Troubleshooting

-   **Model/Processor Loading Issues:** If the script cannot load the model or processor, ensure you have a stable internet connection or that the specified local directory is correct.
-   **Image Loading Issues:** Verify that the `image_path` variable points to a valid image file and that the script has the necessary permissions to access it.
-   **CUDA Errors:** If you encounter CUDA errors, ensure your GPU drivers are up-to-date and compatible with the installed PyTorch version. You can also force CPU usage by changing `DEVICE` to `"cpu"`.
-   **Incorrect Output:** If the output does not match your expectations, double-check the prompt in the `messages` variable and make sure it accurately describes the information you want to extract.

### Notes

-   The model's performance depends on the quality and clarity of the input image.
-   The provided prompt is a template; you may need to modify it for optimal results with different document types.
```
