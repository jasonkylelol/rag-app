import hashlib

def md5sum_str(input_string):
    byte_string = input_string.encode()
    md5_hash = hashlib.md5()
    md5_hash.update(byte_string)
    return md5_hash.hexdigest()
