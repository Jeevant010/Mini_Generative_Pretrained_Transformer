# Mini GPT — Project Documentation & Research Manuals

This `manuals/` directory contains the **complete technical documentation** for the Mini Generative Pretrained Transformer project. It is organized into subdirectories, each covering a distinct aspect of the system — from theoretical foundations to implementation details, experimental methodology, and reproducibility guides.

These documents are written to support the preparation of a **formal research paper** and serve as the definitive reference for every component of the project.

---

## Directory Structure

```
manuals/
├── README.md                            ← You are here
│
├── 01_project_overview/
│   └── project_overview.md              ← High-level summary, motivation, objectives, scope
│
├── 02_theoretical_foundations/
│   ├── transformer_architecture.md      ← Decoder-only Transformer theory & design rationale
│   ├── attention_mechanisms.md          ← MHA, Causal, MQA, GQA, Cross-Attention deep-dive
│   └── tokenization_theory.md          ← BPE theory, byte-level design, vocabulary analysis
│
├── 03_system_architecture/
│   ├── codebase_structure.md            ← Repo layout, module dependency graph, data flow
│   ├── model_architecture.md            ← Layer-by-layer network specification & parameter budget
│   └── data_pipeline.md                 ← Preprocessing, tokenization, memory-mapped I/O
│
├── 04_implementation_details/
│   ├── config_reference.md              ← Every hyperparameter, explained
│   ├── training_pipeline.md             ← Training loop, LR schedule, checkpointing, profiling
│   ├── inference_pipeline.md            ← Generation, sampling strategies, checkpoint loading
│   ├── tokenizer_implementation.md      ← BytePairTokenizer class, HuggingFace backend, API
│   ├── evaluation_metrics.md            ← [NEW] Perplexity, sample generation, VRAM profiling
│   └── hyperparameter_presets.md        ← [NEW] Pre-computed RTX 4060 safe presets for all scales
│
├── 05_experimental_methodology/
│   ├── research_notebooks.md            ← Summary of all Research/ notebooks & their purpose
│   ├── hardware_profiling.md            ← Profiler tooling, TFLOPS measurement, trace analysis
│   ├── training_progression.md          ← From Wizard of Oz to OpenWebText, staged curriculum
│   └── ablation_studies.md              ← [NEW] Toggle reference, expected results, paper guide
│
├── 06_reproducibility/
│   ├── environment_setup.md             ← Python, PyTorch, CUDA, dependencies — step by step
│   ├── quick_start_guide.md             ← End-to-end run in 5 commands
│   └── checkpoint_management.md         ← Resume, best-model tracking, artifact inventory
│
└── 07_appendices/
    ├── glossary.md                      ← Key terms and abbreviations
    ├── references.md                    ← Academic papers, blog posts, codebases cited
    └── license_and_attribution.md       ← MIT License, author, third-party acknowledgements
```

---

## How to Use These Manuals

| If you want to…                            | Read                                          |
| ------------------------------------------ | --------------------------------------------- |
| Understand what this project is about       | `01_project_overview/`                        |
| Write the Theory / Background section       | `02_theoretical_foundations/`                  |
| Write the System Design / Methods section   | `03_system_architecture/`                     |
| Cite specific implementation choices        | `04_implementation_details/`                  |
| Choose hyperparameters for your data scale  | `04_.../hyperparameter_presets.md`             |
| Describe experiments & ablations            | `05_experimental_methodology/`                |
| Prove each component matters (ablation)     | `05_.../ablation_studies.md`                   |
| Enable someone else to reproduce your work  | `06_reproducibility/`                         |
| Look up a term or find a citation           | `07_appendices/`                              |

---

## Notes

- All file paths referenced in these manuals are relative to the project root (`Mini_Generative_Pretrained_Transformer/`).
- Mathematical notation uses LaTeX-compatible syntax for easy copy-paste into paper drafts.
- Diagrams are described in Mermaid where applicable and can be rendered in any Markdown viewer that supports it.
