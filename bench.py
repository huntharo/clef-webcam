"""Benchmark clef-flash and check the optimized linear attention against the reference implementation."""

import io
import statistics
import urllib.request

from PIL import Image

from clef import Clef
from mps_kernels import patch_qwen3_5

SAMPLE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
RUNS = 8

image = Image.open(io.BytesIO(urllib.request.urlopen(SAMPLE).read())).convert("RGB")
image.thumbnail((336, 336))

REQUESTS = {
    "text": {
        "model": "clef-flash",
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "team": {"type": "choice", "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    },
    "image": {
        "model": "clef-flash",
        "state": "A photo.",
        "images": [image],
        "questions": {
            "subject": {"type": "choice", "criteria": {"candy": None, "car": None, "person": None, "landscape": None}},
            "animal_printed": {"type": "noul", "instructions": "Is an animal printed on the object?"},
        },
    },
}


def probabilities(response: dict) -> dict:
    flat = {}
    for question, answer in response["answers"].items():
        options = answer.get("probabilities") or {"true": answer["noul"]}
        flat.update({f"{question}.{option}": p for option, p in options.items()})
    return flat


def measure(clef: Clef, request: dict) -> tuple[float, dict]:
    times, response = [], None
    for i in range(RUNS + 2):
        response, ms = clef.decide(request)
        if i >= 2:
            times.append(ms)
    return statistics.median(times), response


clef = Clef(optimized=False)
print(f"clef-flash on {clef.hardware} ({clef.device})\n")

results = {name: measure(clef, request) for name, request in REQUESTS.items()}
if clef.device != "cuda":
    patch_qwen3_5()
    for name, request in REQUESTS.items():
        reference_ms, reference = results[name]
        optimized_ms, optimized = measure(clef, request)
        a, b = probabilities(reference), probabilities(optimized)
        drift = max(abs(a[key] - b[key]) for key in a)
        print(f"{name:6s} reference {reference_ms:5.0f}ms  optimized {optimized_ms:5.0f}ms  "
              f"{reference_ms / optimized_ms:.1f}x faster  max probability drift {drift:.4f}")
        for question, answer in optimized["answers"].items():
            print(f"         {question}: {answer.get('choice', answer.get('score', answer.get('noul')))}")
else:
    for name, (ms, response) in results.items():
        print(f"{name:6s} {ms:5.0f}ms")
