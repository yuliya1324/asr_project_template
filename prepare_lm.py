import gzip
import os, shutil

lm_gzip_path = 'lm/3-gram.pruned.1e-7.arpa.gz'
uppercase_lm_path = 'lm/3-gram.pruned.1e-7.arpa'
if not os.path.exists(uppercase_lm_path):
    with gzip.open(lm_gzip_path, 'rb') as f_zipped:
        with open(uppercase_lm_path, 'wb') as f_unzipped:
            shutil.copyfileobj(f_zipped, f_unzipped)
    print('Unzipped the 3-gram language model.')
else:
    print('Unzipped .arpa already exists.')

lm_path = 'lm/lowercase_3-gram.pruned.1e-7.arpa'
if not os.path.exists(lm_path):
    with open(uppercase_lm_path, 'r') as f_upper:
        with open(lm_path, 'w') as f_lower:
            for line in f_upper:
                f_lower.write(line.lower())
print('Converted language model file to lowercase.')