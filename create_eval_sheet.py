import json, csv

with open("qwen_zeroshot_results.json", "r") as f:
    results = json.load(f)

with open("human_evaluation_sheet.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Write the header row
    writer.writerow([
        "Figure Filename", 
        "Figure Type", 
        "Original Caption (Ground Truth)", 
        "Qwen2.5-VL Paragraph",
        "Rater 1: Helpfulness (1-5)",
        "Rater 1: Detail/Accuracy (1-5)",
        "Rater 2: Helpfulness (1-5)",
        "Rater 2: Detail/Accuracy (1-5)",
        "Rater 3: Helpfulness (1-5)",
        "Rater 3: Detail/Accuracy (1-5)"
    ])
    
    for r in results:
        writer.writerow([
            r["figure"],
            r["figure_type"],
            r["ground_truth"],
            r["qwen_paragraph"],
            "", "", "", "", "", "" # Blank columns for the 3 raters
        ])

print("✅ Successfully created 'human_evaluation_sheet.csv'")
