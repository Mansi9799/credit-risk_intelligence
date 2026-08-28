import os
import glob
import re

files_to_fix = glob.glob('src/*.py')

for filepath in files_to_fix:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace r"C:\...credit-risk-intelligence\..." with "..."
    # or r'C:\...credit-risk-intelligence\...' with '...'
    new_content = re.sub(r'r"C:\\Users\\[^"]*credit-risk-intelligence\\', '"', content)
    new_content = re.sub(r"r'C:\\Users\\[^']*credit-risk-intelligence\\", "'", new_content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed {filepath}')
