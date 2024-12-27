
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
