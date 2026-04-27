import kagglehub
import os
from dotenv import load_dotenv

# This automatically finds the .env file and loads the secrets into os.environ!
load_dotenv()

print("Authenticating and downloading CIFAR-10...")

# Download latest version
path = kagglehub.competition_download('cifar-10')

print("Path to competition files:", path)
