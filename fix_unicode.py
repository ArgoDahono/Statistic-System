#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

# Read file
with open('Trial.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace unicode characters with ASCII equivalents
replacements = {
    '✓': '[OK]',
    '✗': '[FAIL]',
    '⚠': '[WARN]',
    '⚡': '[ALERT]',
    '→': '->',
    '█': '[#]',
    '━': '=',
}

for unicode_char, ascii_char in replacements.items():
    content = content.replace(unicode_char, ascii_char)

# Write file back
with open('Trial.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully converted unicode characters to ASCII-safe versions")