import re
with open('paper/main.typ', 'r', encoding='utf-8') as f:
    content = f.read()

# Simple replacement: \mathbf{X} -> bold(X)
content = re.sub(r'\\mathbf\{([^}]*)\}', lambda m: 'bold(' + m.group(1) + ')', content)
# \text{X} -> "X"
content = re.sub(r'\\text\{([^}]*)\}', lambda m: '"' + m.group(1) + '"', content)

with open('paper/main.typ', 'w', encoding='utf-8') as f:
    f.write(content)
print('done')