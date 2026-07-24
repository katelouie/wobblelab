# Running WobbleLab against a remote GPU

The harness already speaks the OpenAI `/v1` protocol (`OpenAICompatibleProvider`), so a remote
GPU is a config change, not a rewrite. Point the `WOBBLE_*` env vars at a server running vLLM
and the exact same experiment runs there. This doc is the runbook for doing that on RunPod.

## Which RunPod offering: use **Pods**

RunPod sells four things. For *developing and iterating on wobble experiments*, you want Pods.

| Offering | What it is | For us |
|---|---|---|
| **Pods** | A GPU container you rent and control (SSH in, run vLLM). Billed per-second while running; no scale-to-zero, no cold starts. | **Yes.** It's a remote `llama-server`: full control of model/quant/sampling, logprobs guaranteed (we need them for `ll` scoring), and a persistent endpoint our `concurrency=N` client can batch against with no mid-run cold starts. |
| **Serverless** | Scale-to-zero workers, per-second billing of active compute, OpenAI-compatible vLLM path. | Later. Its win is *no idle cost for intermittent traffic* — great for a public demo, overkill for a batch run where you'd just stop a Pod. RunPod lets you move the same container Pod→Serverless later, so this isn't a dead end. |
| **Public Endpoints** | Pre-built managed model APIs, zero setup. | No. You don't control the exact model/quant/sampling, and may not get logprobs — which defeats the provenance point of a reliability tool. |
| **Clusters** | Multi-node GPU clusters for distributed training / very large models. | No. Not needed until 70B+ distributed. |

The deciding factor is provenance: WobbleLab's whole claim is *reproducible numbers with exact
settings*. A Pod gives you the model, quantization, sampling config, and logprobs under your
control. The trade is that a Pod bills while it idles, so **stop it when you're done** (or use a
Spot/interruptible Pod, and set an idle timeout).

## Cost

A Community Cloud RTX 4090 runs roughly $0.34–0.69/hr and fits Qwen2.5-7B comfortably. A
2-category MMLU-Pro CoT run is a few minutes of that. Spin up, iterate for an hour, stop:
~$0.50–1 a session, ~$20 covers weeks. Set a spend alert.

## Setup, once

1. Create a RunPod account, add ~$10 of credit, set a spend limit.
2. Deploy a **Pod**: Community Cloud, a single **RTX 4090 (24GB)**, a vLLM-capable template
   (RunPod's official vLLM template, or any PyTorch/CUDA image where you `pip install vllm`).
3. Inside the Pod, serve the model with an API key:
   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --api-key "$MY_KEY"
   ```
   vLLM exposes an OpenAI-compatible server at `:8000/v1` with logprobs support.

## Connect, each session

Reach the Pod's `:8000` over an **SSH tunnel** (safest — nothing public):

```bash
# forward the Pod's 8000 to your local 8000 (RunPod gives you the ssh host/port)
ssh -N -L 8000:localhost:8000 root@<pod-ssh-host> -p <pod-ssh-port>
```

Then, in the shell where you run experiments:

```bash
export WOBBLE_BASE_URL=http://localhost:8000/v1
export WOBBLE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export WOBBLE_MODEL_LABEL="Qwen2.5-7B-Instruct"
export WOBBLE_BACKEND="vLLM (RunPod 4090)"
export WOBBLE_API_KEY="$MY_KEY"

python experiments/run_gpqa.py            # same experiment, now on the 7B
python experiments/gpqa_harness_sweep.py
```

`experiments/backend.py` reads those vars; with none set it falls back to the local llama.cpp
server, so nothing changes for local runs. Note `WOBBLE_MODEL` containing `qwen3` auto-enables
thinking suppression; Qwen2.5 (and other non-thinking models) correctly get no such flag.

## Discipline

- **Stop the Pod when done.** It bills while running, idle or not.
- **API key + SSH tunnel**, never a bare public port.
- **Public benchmark data only** on rented GPUs. No secrets, no sensitive inputs.
- Keys and Pod hostnames live in env vars / your shell, **never committed** to the repo.
