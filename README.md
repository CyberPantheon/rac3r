# rac3r 🏎️

A fast, interactive, and robust GUI-based race condition tester for web applications. Built with Streamlit and Python, rac3r allows you to easily test for race condition vulnerabilities by sending bursts of concurrent requests to a target endpoint.

## ⚠️ Disclaimer

This tool is intended for authorized security testing and educational purposes only. Sending a high volume of requests to a target without explicit, written permission from the system owner is illegal and unethical. The user is solely responsible for their actions and must comply with all applicable laws. The author assumes no liability for any misuse or damage caused by this tool.

## Features

  * **Interactive GUI**: Easy-to-use web interface powered by Streamlit.
  * **Raw Request Pasting**: Simply paste a raw HTTP request copied from Burp Suite or other proxy tools.
  * **Robust Parsing**: Handles various HTTP versions (HTTP/1.0, HTTP/1.1, HTTP/2) and intelligently infers the target scheme (http/https).
  * **Configurable Attack**: Control the number of concurrent requests, bursts (repetitions), and per-request timeouts.
  * **Dynamic Payloads**: Use variables like `_name_` or `{{name}}` in your request and define their values in the UI. Use the `unique` keyword to generate a unique token for each request.
  * **Detailed Analysis**:
      * View response code distribution and performance metrics (min/max/avg response time).
      * Automatically diff response bodies to quickly spot anomalies.
      * Inspect every raw request and response pair.
      * Visualize response times with a simple plot.
  * **History & Logging**: Save successful requests for later use and log all results to a local file.

## Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/CyberPantheon/rac3r.git
    cd rac3r
    ```
2.  Create and activate a virtual environment (recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1.  Once the dependencies are installed, run the Streamlit application:
    ```sh
    streamlit run race.py
    ```
2.  Your web browser will automatically open a new tab with the rac3r interface.
3.  Paste your raw HTTP request into the text area.
4.  Configure the concurrency, repetitions, and other settings in the sidebar.
5.  Click the "Run Test" button to launch the attack.
6.  Analyze the results that appear below the run controls.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
