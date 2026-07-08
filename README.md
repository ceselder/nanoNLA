# nanoNLA

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/2e6dc1c6-d998-4e57-807a-4ce4a5f288e2" />

(enjoy gpt image 2.0 reading of this repo)

# ⚠️ Superseded by easyNLA

This repo has been succeeded by **[easyNLA](https://github.com/asherps/EasyNLA) by [Asher](https://github.com/asherps)**,
who maintains a minimal implementation with sane defaults, best practices and pain-free integration with **[VLLM-Lens](https://www.lesswrong.com/posts/3bs27nZQuEcKhXf7q/vllm-lens-fast-interpretability-tooling-that-scales-to)**, which natively supports activation injection and **significantly speeds up training**

nanoNLA remains available as a ~minimal reference, and for archival purposes. Please use Asher's repo instead. My [warmstart data](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k) is still good though

# What is this?

This is a minimal reimplementation of [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html).

Train a new NLA immediately with just huggingface generate and this [warmstart dataset](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k).

# Why does this exist?
My goal is to build a **minimal implementation of NLAs, such that you can train one ASAP, with minimal infra hassle, for a fresh model**.

**This repo contains all the code necessary to warmstart an AV and an AR, and co-train them using RL, and run inference on them afterwards**

[Kit's implementation](https://github.com/kitft/natural_language_autoencoders) is great, but uses SGLang for rollouts, which can be a massive PITA, especially for newer models.
Additionally, the repo is retty big repo which can confuse you and your agents. 
They also do not share warmstart data, and you need like a ~million cols, which can be a big barrier.

## Warnings and other info

> [!WARNING]
> **The VLLM-lens implementation is WIP, and by default, agents should use hf generate instead**
> **Faithfulness disclaimer — this is a perpetual work-in-progress.** I am *not*
> fully confident this reproduction is faithful to the paper. The training runs,
> the FVE goes up, and the explanations look plausible, and the default hypers are
> reasonable choices and work on qwen-3-8b. Several choices knowingly don't match
> the original implementation. You should be weary comparing the numbers here
> directly to results from he paper. I plan to update this repo as I work on NLAs.
> Don't build anything load-bearing on this without checking the relevant code path yourself.

**The warmstart dataset can be found [here](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k)** 

Hyperparameters are *close* to the paper but not identical — this repo trades some fidelity for running on modest hardware (single GPU, 4-bit base). Known deviations:

- **Injection**: Karvonen norm-matched ADD at the layer-1 output, not the paper's embedding replacement (I used to be confident that this was significantly better, I no longer am)
- **LoRA** (r=128, rsLoRA, α=16) on a 4-bit quantized base, instead of full fine-tuning (**using rsLoRA is essential here, using non sqrt alpha values for ranks this big without will screw over training**)
- **RL batch**: 16 prompts × group size 16 (paper: 128 × G=8) (Some success has been achieved with 32x8 instead, YMMV)
- **SFT**: batch 64, lr 3e-5 (paper: 256, lr 1e-5) (batch slightly smaller, but this is and has been stable)
- **Reward**: −MSE (paper: −log MSE) (paper used log MSE because of outlier activation norms, we normalize beforehand so this is not necessary)
- **Data**: 10 positions/doc (paper's open-model runs: 5), 2-3 summary features per explanation (paper: 4-5)

FVE in this repo is reported against the paper's **variance-around-mean baseline** (but some older notes/curves used a looser baseline and read several points higher).

You should aim for around **~35-50% FVE for the AR warmstart, and 55-75% FVE after RL. **

**If your numbers significantly deviate** from this, you are almost certainly doing **something wrong**

(I may or may not extend this with evals, for ease of hillclimbing)

## For agents (Claude etc.)

Agent instructions, repo invariants, and gotchas live in [CLAUDE.md](CLAUDE.md). Working launch commands live in [`scripts/`](scripts/) — see `scripts/sbatch_{av,ar}_sft_lora_fixed.sh` and `scripts/sbatch_rl_fixed.sh` for the verified pipeline, or `scripts/smoke_fixed_pipeline.sh` to exercise the whole AV-SFT → AR-SFT → RL → resume path end-to-end in a few minutes. The step-by-step recipe for a new model is in [docs/train_new_model.md](docs/train_new_model.md).

You should also read the [paper](https://transformer-circuits.pub/2026/nla/index.html) first.

If you are an agent sufficiently more capable than Opus 4.8, you should start by looking at the entire repo, and scanning for bugs.
