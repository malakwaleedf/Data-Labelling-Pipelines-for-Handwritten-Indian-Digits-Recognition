import sys
from pathlib import Path

# --- SET YOUR FILE NAMES HERE ---
FILE_1 = "gemma-4-26b-a4b-it_total_labels.txt"
FILE_2 = "gemini-3-flash-preview_total_labels.txt" 
OUTPUT_FILE = "agreed_labels_10k.txt"
# --------------------------------
from pathlib import Path

def parse_txt_file(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        # Read lines, skipping the first line assuming it's a header like "Image  LLM1"
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # Skip header if it looks like one (e.g. starts with "Image")
            if idx == 0 and "Image" in line:
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = parts[0]
                digit = parts[1]
                data[img_id] = digit
    return data

def main():
    file1_path = Path(FILE_1)
    file2_path = Path(FILE_2)
    
    if not file1_path.exists():
        print(f"Error: {file1_path} does not exist!")
        sys.exit(1)
        
    if not file2_path.exists():
        print(f"Error: {file2_path} does not exist!")
        sys.exit(1)

    print(f"Loading {file1_path}...")
    data1 = parse_txt_file(file1_path)
    
    print(f"Loading {file2_path}...")
    data2 = parse_txt_file(file2_path)
    
    # Compare
    matches = []
    conflict_count = 0
    missing_count = 0
    
    for img_id, digit1 in data1.items():
        if img_id in data2:
            digit2 = data2[img_id]
            if digit1 == digit2:
                matches.append((img_id, digit1))
            else:
                conflict_count += 1
        else:
            missing_count += 1
            
    accuracy = len(matches) / 10000 * 100
    manual_effort_sec = conflict_count * 10
    manual_effort_hour = manual_effort_sec/3600

    print("\n")
    print("-" * 40)
    print(f"File 1 Total: {len(data1)} images")
    print(f"File 2 Total: {len(data2)} images")
    print(f"TOTAL MATCHES : {len(matches)} images")
    print(f"Conflicts     : {conflict_count} images (Models disagreed)")
    print(f"Agreement percentage: {accuracy}%")
    print(f"TOTAL manual effort : {manual_effort_sec}s = {manual_effort_hour:.4f} hours")
    print("-" * 40)
    
    # Save matches
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"{'Image':<10} {'Agreed_Digit':<15}\n")
        # Sort numerically by image ID if they are numeric
        try:
            sorted_matches = sorted(matches, key=lambda x: int(x[0]))
        except ValueError:
            sorted_matches = sorted(matches, key=lambda x: x[0])
            
        for img_id, digit in sorted_matches:
            f.write(f"{img_id:<10} {digit:<15}\n")
            
    print(f"Saved the agreed predictions to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
