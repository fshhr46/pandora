
def mask_data(origin: str, positions, mask_char="*"):
    chars = list(origin)
    for position in positions:
        if position >= len(chars):
            chars.append(mask_char)
        else:
            chars[position] = mask_char
    return "".join(chars)
