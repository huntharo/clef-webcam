# clef-flash on your webcam

Run [clef-flash](https://huggingface.co/Cloudflare/clef-flash), Cloudflare's open-weight decision model, locally on a Mac and turn your webcam into typed decisions with probabilities.

A decision model doesn't generate text. You give it a state (here, a webcam frame) and a schema of questions, and it returns a probability for every allowed answer in one forward pass. [Read the announcement](https://blog.cloudflare.com/clef-decision-models/).

## Quickstart

You need an Apple Silicon Mac with 32 GB+ of memory (the model is ~19 GB) and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run hf download Cloudflare/clef-flash --local-dir models/clef-flash
uv run uvicorn server:app --port 8787
```

Open http://localhost:8787 and allow camera access.

## Controls

| Key | Action |
| --- | --- |
| `E` | Edit the questions. They apply live, no retraining |
| `space` | Pause |
| `shift` + `R` | Reset to the default questions |

## Asking your own questions

Questions use the same schema as `@cf/cloudflare/clef` on Workers AI (and Jev / SystemOne):

```json
{
  "wearing_glasses": { "type": "noul", "instructions": "Wearing glasses?" },
  "mood": { "type": "choice", "criteria": { "happy": null, "neutral": null, "tired": null } },
  "energy": { "type": "score", "criteria": ["low", "medium", "high"] }
}
```

- `noul`: yes or no
- `choice`: pick one option. Descriptions are optional (`null`)
- `score`: an ordered scale

Latency grows with prompt length, so short instructions and few options keep it snappy.

## Performance

Measured on an M5 Max with `uv run python bench.py`, which also checks that the optimized model gives the same answers as the reference:

| Request | transformers on MPS | This repo |
| --- | --- | --- |
| Text, 3 questions | 566 ms | 155 ms |
| Webcam frame, 3 questions | 840 ms | 250 ms |

On a Mac, Qwen3.5's linear-attention layers fall back to a slow reference implementation, because the fast kernels are CUDA-only. Most of that time goes to `torch.linalg.solve_triangular`. [`mps_kernels.py`](mps_kernels.py) computes the same thing with batched block inversion, which Apple GPUs handle well. It is patched in automatically on MPS and CPU, and left alone on CUDA.

## Files

| File | |
| --- | --- |
| `server.py` | API: webcam frame + questions → decisions |
| `index.html` | The live demo page |
| `clef.py` | Model loading and timing |
| `mps_kernels.py` | Faster linear attention for Apple Silicon |
| `bench.py` | Speed and accuracy check |

`joint_schema_model.py` (Clef's reference inference code) is downloaded with the model.

## Settings

| Variable | Default | |
| --- | --- | --- |
| `CLEF_MODEL_DIR` | `models/clef-flash` | Where the model lives |
| `CLEF_MAX_SIDE` | `336` | Longest side of the frame sent to the model, in pixels |

## License

This code is [Apache 2.0](LICENSE) licensed, and so is the [clef-flash model](https://huggingface.co/Cloudflare/clef-flash).
