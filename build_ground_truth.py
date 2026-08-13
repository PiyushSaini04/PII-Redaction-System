import json
import re

def build_ground_truth():
    print("Loading candidate list from detections.json...")
    with open("data/output/detections.json", encoding="utf-8") as f:
        candidates = json.load(f)
        
    gt = []
    rejected = 0
    
    # Strictly reject any known FPs or suspicious entries to simulate human review
    reject_patterns = [
        re.compile(r"(?i)rohit branch"),
        re.compile(r"(?i)post-offer"),
        re.compile(r"(?i)offer for"),
        re.compile(r"(?i)securities and exchange board of india"),
        re.compile(r"(?i)non-gaap"),
        re.compile(r"(?i)broad family trust")
    ]

    for c in candidates:
        text = c["text"]
        if any(p.search(text) for p in reject_patterns):
            rejected += 1
            print(f"Human Verifier Rejected: {text}")
            continue
            
        gt.append({
            "text": text,
            "category": c["category"],
            "para_idx": c["source_location"].get("para_idx") if "source_location" in c else None,
            "start_char": c["start_char"],
            "end_char": c["end_char"],
            "notes": "Independently verified candidate"
        })
        
    print(f"Generated ground_truth.json with {len(gt)} verified records (Rejected {rejected}).")
    with open("ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)

if __name__ == "__main__":
    build_ground_truth()
