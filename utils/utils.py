import hashlib


def get_hash(content):
    return hashlib.md5(content.encode()).hexdigest()