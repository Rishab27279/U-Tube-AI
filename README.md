  # --- Clone the repository and enter the directory ---
  git clone https://github.com/Rishab27279/U-Tube-AI.git
  cd U-Tube-AI
  
  # --- Create a virtual environment ---
  python -m venv venv
  
  # --- Activate the virtual environment ---
  # On Windows, use this command:
  .\venv\Scripts\activate
  # On macOS & Linux, use this command instead:
  # source venv/bin/activate
  
  # --- Install all required packages from the requirements file ---
  pip install -r requirements.txt
  
  # --- Create the .env file for your API key from the template ---
  # On Windows, use this command:
  copy example.env .env
  # On macOS & Linux, use this command instead:
  # cp example.env .env
