import re

def standardize_category(category):
    save_category = category.replace(' ', '_').replace(',', '_').replace('/', '_')
    while ' ' in save_category or ',' in save_category:
        save_category = save_category.replace(' ', '_').replace(',', '_')
    save_category = save_category.replace('__', '_')
    return save_category

def standardize(string: str) -> str:
    s1 = re.sub('([a-z0-9])([A-Z])', '\\1_\\2', string)
    s2 = re.sub('([A-Z]+)([A-Z][a-z])', '\\1_\\2', s1)
    string = s2
    res = re.compile('[^一-龥^a-z^A-Z^0-9^_]')
    string = res.sub('_', string)
    string = re.sub('(_)\\1+', '_', string).lower()
    while string and string[0] == '_':
        string = string[1:]
    while string and string[-1] == '_':
        string = string[:-1]
    if not string:
        return string
    if string[0].isdigit():
        string = 'get_' + string
    return string

def change_name(name):
    change_list = ['from', 'class', 'return', 'false', 'true', 'id', 'and']
    if name in change_list:
        name = 'is_' + name
    return name
