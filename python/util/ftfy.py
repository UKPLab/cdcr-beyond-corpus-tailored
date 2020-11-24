import unicodedata

import ftfy


def clean_string(s: str):
    """
    Remove all sorts of unicode gremlins from a string
    :param s: dirty string
    :return: clean string
    """

    # all-in-one repair, covers several issues
    fixed = ftfy.fix_text(s)

    # more manual fixes, see https://www.compart.com/de/unicode/category for the categories
    chars_fixed = []
    for c in fixed:
        category = unicodedata.category(c)
        if category in ["Cc", "Cf"]:
            # specifically remove control characters, these have caused problems
            # TODO we probably lose support for Arabic or Hebrew here
            continue
        elif category in ["Zl", "Zp", "Zs"]:
            # Replace different kinds of whitespace chars with plain spaces. Replace repeated whitespace with a
            # single space.
            if chars_fixed and chars_fixed[-1] == " ":
                continue
            chars_fixed.append(" ")
        else:
            chars_fixed.append(c)

    fixed = "".join(chars_fixed)
    return fixed
