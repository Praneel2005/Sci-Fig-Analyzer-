"""
expand_captions.py
Creates paragraph-length training targets by combining:
  - Original short caption
  - OCR tokens (axis labels, legend text, numbers from figure)
  - Figure type information
  - Structured 5-sentence description template

Produces ~/figcaps_data_paragraph — a new dataset where
every caption-prepend is a full paragraph (100-200 words).
No API needed. Fully local.
"""

import os
import re
import json
from tqdm import tqdm

ORIG_PATH   = os.path.expanduser("~/figcaps_data/Caption-All")
OUTPUT_PATH = os.path.expanduser("~/figcaps_data_paragraph/Caption-All")
SPLITS      = ["train", "val", "test"] # Adjusted to lowercase to match server filesystem

FIGURE_TYPE_PHRASES = {
    "Graph Plot":     "a graph plot showing quantitative data trends over variables",
    "Bar Chart":      "a bar chart comparing values across distinct categories",
    "Table":          "a data table presenting structured numerical or textual information",
    "Neural Network": "a neural network architecture diagram showing layers and connections",
    "Scatter Plot":   "a scatter plot illustrating the relationship between two variables",
    "Block Diagram":  "a block diagram showing system components and their interactions",
    "Pie Chart":      "a pie chart showing the proportional distribution of categories",
    "Flow Chart":     "a flowchart describing a sequential process or decision algorithm",
    "Box Plot":       "a box plot summarizing the statistical distribution of data",
    "Heat Map":       "a heatmap visualizing intensity or correlation across two dimensions",
}

CLOSING_BY_TYPE = {
    "Graph Plot":     ("The visualization supports the quantitative analysis "
                       "presented in the corresponding section of the paper, "
                       "enabling the reader to assess trends and compare conditions directly."),
    "Bar Chart":      ("The bar chart allows direct visual comparison across groups, "
                       "supporting the claims made in the paper regarding relative performance "
                       "or magnitude differences between the presented categories."),
    "Neural Network": ("The architecture diagram clarifies the structural design choices "
                       "of the proposed model, illustrating how information flows through "
                       "the network from input to output."),
    "Block Diagram":  ("The diagram illustrates the structural or procedural relationships "
                       "central to the methodology, helping the reader understand how the "
                       "system components interact to produce the described behaviour."),
    "Flow Chart":     ("The flowchart outlines the step-by-step logic of the described "
                       "algorithm or process, making the procedural dependencies and "
                       "decision points explicit for the reader."),
    "Scatter Plot":   ("The scatter plot reveals the distributional relationship between "
                       "the two measured variables, allowing the reader to assess correlation "
                       "strength and identify potential outliers in the data."),
    "Table":          ("The table organises the reported values in a structured format, "
                       "allowing direct comparison across rows and columns as discussed "
                       "in the paper."),
}

DEFAULT_CLOSING = ("This figure provides direct visual evidence supporting the "
                   "experimental results and analytical claims presented in the paper, "
                   "and should be interpreted in conjunction with the surrounding text.")


def build_paragraph(data):
    """
    Builds a 5-sentence paragraph description from figure metadata.
    Target length: 100-200 words.
    """
    orig_caption = data.get("0-originally-extracted", "").strip()
    figure_type  = data.get("figure-type", "figure")
    ocr_tokens   = data.get("Img-text", [])

    # Clean OCR tokens
    clean_ocr = [
        str(t).strip()
        for t in ocr_tokens
        if isinstance(t, str)
        and len(str(t).strip()) > 1
        and not str(t).strip().isspace()
    ]

    # Separate numbers from text labels
    numbers = [t for t in clean_ocr
               if any(c.isdigit() for c in t)][:8]
    labels  = [t for t in clean_ocr
               if not any(c.isdigit() for c in t)
               and len(t) > 2][:8]

    # Figure type phrase
    type_phrase = FIGURE_TYPE_PHRASES.get(
        figure_type,
        f"a scientific {figure_type.lower()} figure"
    )

    # Clean original caption — remove "Figure X:" prefix
    clean_cap = re.sub(
        r'^Figure\s*\d+\s*[:\.\-]\s*', '', orig_caption,
        flags=re.IGNORECASE
    ).strip()
    if not clean_cap:
        clean_cap = "This figure illustrates results relevant to the paper."

    # Build paragraph sentences
    sentences = []

    # S1 — Figure type identification
    sentences.append(
        f"This figure presents {type_phrase}."
    )

    # S2 — Core content from original caption
    sentences.append(clean_cap if clean_cap.endswith('.') else clean_cap + '.')

    # S3 — OCR-derived visual element description
    if labels and numbers:
        label_str = ", ".join(f'"{l}"' for l in labels[:5])
        num_str   = ", ".join(numbers[:5])
        sentences.append(
            f"The figure contains labeled elements including {label_str}, "
            f"with key numerical values such as {num_str} visible in the visualization."
        )
    elif labels:
        label_str = ", ".join(f'"{l}"' for l in labels[:5])
        sentences.append(
            f"The figure contains labeled elements including {label_str}, "
            f"which identify the variables, axes, or components depicted."
        )
    elif numbers:
        num_str = ", ".join(numbers[:6])
        sentences.append(
            f"Key numerical values visible in the figure include {num_str}, "
            f"representing the quantitative data presented."
        )
    else:
        sentences.append(
            "The visual elements of the figure encode the relevant "
            "experimental or theoretical information described in the paper."
        )

    # S4 — Interpretation sentence based on figure type
    if figure_type in ["Graph Plot", "Bar Chart", "Scatter Plot"]:
        sentences.append(
            "To interpret this figure, the reader should examine the axis "
            "scales, legend entries, and the relative positions or heights "
            "of the plotted elements to extract the quantitative findings."
        )
    elif figure_type in ["Neural Network", "Block Diagram", "Flow Chart"]:
        sentences.append(
            "To understand this figure, the reader should trace the "
            "directional connections between components, noting how "
            "information or control flow is represented by the arrows "
            "and structural groupings shown."
        )
    else:
        sentences.append(
            "The reader should examine all labeled components and their "
            "spatial relationships to fully understand the information "
            "conveyed by this figure."
        )

    # S5 — Closing contextual sentence
    sentences.append(
        CLOSING_BY_TYPE.get(figure_type, DEFAULT_CLOSING)
    )

    paragraph = " ".join(sentences)
    return paragraph


# ── Main processing loop ──────────────────────────────────────────
print("Building paragraph-length caption dataset...")
print(f"Source : {ORIG_PATH}")
print(f"Output : {OUTPUT_PATH}")
print()

all_lengths = []

for split in SPLITS:
    src_dir = os.path.join(ORIG_PATH,   split)
    dst_dir = os.path.join(OUTPUT_PATH, split)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.endswith(".json")])
    print(f"Processing {split}: {len(files)} files...")

    split_lengths = []

    for fname in tqdm(files, desc=split):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        with open(src_path) as f:
            data = json.load(f)

        paragraph = build_paragraph(data)
        word_count = len(paragraph.split())
        split_lengths.append(word_count)
        all_lengths.append(word_count)

        # Write paragraph into all four human-feedback caption-prepend fields
        for hf_type in ["helpfulness", "ocr", "visual", "takeaway"]:
            try:
                label = data["human-feedback"][hf_type]["label"]
                data["human-feedback"][hf_type]["caption-prepend"] = \
                    f"{label} {paragraph}"
            except (KeyError, TypeError):
                pass

        # Store for reference
        data["expanded-caption"] = paragraph

        with open(dst_path, "w") as f:
            json.dump(data, f)

    avg_len = sum(split_lengths) / len(split_lengths)
    print(f"  {split} done. Avg paragraph length: {avg_len:.1f} words\n")

# ── Symlink image and metadata folders ───────────────────────────
paragraph_root = os.path.expanduser("~/figcaps_data_paragraph")
for item in ["No-Subfig-Img",
             "List-of-Files-for-Each-Experiments",
             "human-feedback.csv",
             "arxiv-metadata-oai-snapshot.json"]:
    src = os.path.expanduser(f"~/figcaps_data/{item}")
    dst = os.path.join(paragraph_root, item)
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"Symlinked: {dst} → {src}")

# ── Final summary ─────────────────────────────────────────────────
print()
print("=" * 60)
print("DATASET CREATION COMPLETE")
print("=" * 60)
print(f"Total files processed : {len(all_lengths)}")
print(f"Average paragraph len : {sum(all_lengths)/len(all_lengths):.1f} words")
print(f"Min length            : {min(all_lengths)} words")
print(f"Max length            : {max(all_lengths)} words")
under_50  = sum(1 for l in all_lengths if l < 50)
over_100  = sum(1 for l in all_lengths if l >= 100)
over_150  = sum(1 for l in all_lengths if l >= 150)
print(f"Paragraphs < 50 words : {under_50}  (should be near 0)")
print(f"Paragraphs >= 100 words: {over_100}")
print(f"Paragraphs >= 150 words: {over_150}")
print()
print("Next step: run this to verify one sample:")
print("  python -c \"import json; d=json.load(open(")
print("  '~/figcaps_data_paragraph/Caption-All/train/") # Fixed case
print("  1001.0025v1-Figure2-1.json')); ")
print("  print(d['expanded-caption'])\"")
print()
print("Then use --benchmark_path ~/figcaps_data_paragraph")
print("in train_blip_paragraph.py for paragraph training.")
