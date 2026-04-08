import os
from dotenv import load_dotenv

def get_config(key_name, default=None):
    """
    Utility to load configuration values from environment variables or a local token file.
    Priority:
    1. local 'token.txt' file (if exists, treats each line as a potential match)
    2. Environment variables
    3. Default value
    """
    # Load .env if present
    load_dotenv()
    
    # Check for local file 'token.txt'
    # Format expected: KEY_NAME=VALUE
    token_file = 'token.txt'
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    if k == key_name:
                        return v
    
    # Fallback to environment variables
    return os.getenv(key_name, default)
