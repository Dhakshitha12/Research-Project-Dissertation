import json


_COLD = True


def handle(event, context):
    global _COLD
    if isinstance(event, str):
        event = json.loads(event)

    transformed = [
        {
            "id": row["id"],
            "value": round(row["value"] * 1.15, 2),
            "group": row["group"].strip().lower(),
        }
        for row in event["records"]
        if row["active"]
    ]
    result = {
        "count": len(transformed),
        "total": round(sum(row["value"] for row in transformed), 2),
        "cold_start": _COLD,
    }
    _COLD = False
    return json.dumps(result)
