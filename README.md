
# Text Search Engine

This project implements a simple text search engine using Python. It processes text files to create a searchable index of sentences, allowing users to perform similarity-based searches against this indexed data.

## Installation

To set up the project environment, follow these steps:

1. Clone the project repository or download the project files to your local machine.
2. Navigate to the project directory.
3. Create a Python virtual environment in the project directory:
   ```bash
   python -m venv env
   ```
4. Activate the virtual environment (mac/linux):
   ```bash
   source env/bin/activate
   ```
1. Install the required packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the necessary spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

To use the text search engine, follow these steps:

1. Place your `.txt` data files in the `data` directory.
2. Run the script using the following command, replacing `your_search_query` with your desired search term:
   ```bash
   python search_engine.py "your_search_query"
   ```
3. The script will process the text files (if not already processed) and perform the search, returning the top matching sentences along with their filenames and line numbers.

## Contributing

Contributions to this project are welcome.

## License

This project is licensed under [MIT License](LICENSE).
