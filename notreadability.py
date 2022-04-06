import re

text = 'hello@hello,hello. helo ,hello'

text = re.sub(r'(?<!(\s))([^a-zA-Z0-9\s])', r' \2', text)
text = re.sub(r'([^a-zA-Z0-9\s])(?!(\s))', r'\1 ', text)

print(text)