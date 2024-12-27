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
