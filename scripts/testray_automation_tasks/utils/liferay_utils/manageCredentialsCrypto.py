#!/usr/bin/env python
from os import remove
from pathlib import Path
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from os.path import expanduser
from shutil import rmtree
import os
import sys

FOLDER_NAME = ".jira_user"

def delete_credentials():
    home = Path.home()
    folder = home / FOLDER_NAME

    files_to_delete = [
        folder / "token.enc",
        folder / "public.pem",
        folder / "private.pem"
    ]

    for file in files_to_delete:
        try:
            file.unlink(missing_ok=True)  # Python 3.8+: safe even if file doesn't exist
        except Exception as e:
            print(f"⚠️ Could not delete {file}: {e}")

def encrypt_and_store_token():
    home = Path(expanduser("~"))
    folder = home / FOLDER_NAME
    path_to_token_plain = folder / "token"
    path_to_token_enc = folder / "token.enc"
    pub_key_path = folder / "public.pem"

    # Make sure RSA keys exist
    generate_rsa_keypair(folder)

    if not path_to_token_plain.is_file():
        print("No plaintext token to encrypt.")
        return

    with open(pub_key_path, "rb") as f:
        public_key = RSA.import_key(f.read())

    with open(path_to_token_plain, "r") as f:
        token = f.read().strip()

    cipher = PKCS1_OAEP.new(public_key)
    encrypted_token = cipher.encrypt(token.encode())

    with open(path_to_token_enc, "wb") as f:
        f.write(encrypted_token)

    remove(path_to_token_plain)
    print(f"Encrypted token saved to {path_to_token_enc} and plaintext token removed.")

def generate_rsa_keypair(folder: Path):
    private_key_path = folder / "private.pem"
    public_key_path = folder / "public.pem"

    if private_key_path.exists() and public_key_path.exists():
        return  # Already exists

    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()

    folder.mkdir(parents=True, exist_ok=True)

    with open(private_key_path, "wb") as f:
        f.write(private_key)
    with open(public_key_path, "wb") as f:
        f.write(public_key)

    print(f"RSA key pair generated in {folder}")

def get_credentials():
    """
    Hybrid mode:
      - First try GitHub Actions secrets (env variables JIRA_API_USER, JIRA_API_TOKEN)
      - If not present, fallback to ~/.jira_user encrypted storage
    """
    env_user = os.getenv("JIRA_API_USER")
    env_token = os.getenv("JIRA_API_TOKEN")

    if env_user and env_token:
        # Running in GitHub Actions (or with exported envs locally)
        return env_user, env_token

    # Otherwise, fallback to ~/.jira_user
    home = Path(expanduser("~"))
    folder = home / FOLDER_NAME
    path_to_user = folder / "user"
    path_to_token_plain = folder / "token"
    path_to_token_enc = folder / "token.enc"
    priv_key_path = folder / "private.pem"

    # Only encrypt if plaintext token is found
    if path_to_token_plain.is_file():
        encrypt_and_store_token()

    # Now validate required credential files
    if not (path_to_user.is_file() and priv_key_path.is_file() and path_to_token_enc.is_file()):
        print(f"Missing required credential files in {folder}.")
        sys.exit(1)

    with open(path_to_user, 'r') as f:
        username = f.read().strip()

    with open(priv_key_path, "rb") as f:
        private_key = RSA.import_key(f.read())

    cipher = PKCS1_OAEP.new(private_key)

    with open(path_to_token_enc, 'rb') as f:
        encrypted_token = f.read()

    token = cipher.decrypt(encrypted_token).decode()

    return username, token

if __name__ == "__main__":
    print("Set the credentials to be used when accessing Jira.")
    delete_credentials()
    encrypt_and_store_token()
    user, token = get_credentials()
    print(f"Credentials set. User: {user}")