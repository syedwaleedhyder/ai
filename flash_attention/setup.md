# Create a new environment
uv venv

### Activate the new virtual environment
#### On macOS/Linux:
source .venv/bin/activate
#### On Windows (PowerShell):
.venv\Scripts\activate

### Install the dependencies from the requirements.txt file
uv pip install -r requirements.txt