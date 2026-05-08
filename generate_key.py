from cryptography.hazmat.primitives.ciphers.aead import AESCCM
import base64

key = AESCCM.generate_key(bit_length=256)
key = base64.b64encode(key).decode()

with open("key.txt", "w") as f:
    f.write(F"KEY={key}")