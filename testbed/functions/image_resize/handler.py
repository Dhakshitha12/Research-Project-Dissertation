import json


_COLD = True


def handle(event, context):
    global _COLD
    if isinstance(event, str):
        event = json.loads(event)

    width, height = event["width"], event["height"]
    out_width, out_height = event["out_width"], event["out_height"]
    pixels = event["pixels"]
    checksum = 0
    for y in range(out_height):
        source_y = y * height // out_height
        for x in range(out_width):
            source_x = x * width // out_width
            offset = (source_y * width + source_x) * 3
            checksum += sum(pixels[offset:offset + 3])

    cold = _COLD
    _COLD = False
    return json.dumps({
        "width": out_width,
        "height": out_height,
        "checksum": checksum,
        "cold_start": cold,
    })
