"""
Eastern Arabic / Indian Digit Recognition Pipeline
Uses OpenRouter API (google/gemma-4-26b-a4b-it) with detailed reasoning output
"""
from __future__ import annotations
import io, time, json, cv2, random
import numpy as np, csv, base64, logging
from pathlib import Path
from PIL import Image
from openai import OpenAI
import argparse

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY    = "API_KEY"
#google/gemma-4-26b-a4b-it
#google/gemini-3-flash-preview
#openai/gpt-5.4
#anthropic/claude-sonnet-4.6
#openai/gpt-4o
MODEL      = "google/gemma-4-26b-a4b-it"
IMAGE_DIR  = Path("500_labels_for_testing")
GT_JSON    = Path("500_labels_for_testing.json")
#pipeline_google/gemini-3-flash-preview
#pipeline_google/gemma-4-26b-a4b-it
#pipeline_google/gpt-5.4
#pipeline_google/claude-sonnet-4.6
#pipeline_google/gpt-4o
OUTPUT_CSV = Path("pipeline_google/gemma-4-26b-a4b-it.csv")
OUTPUT_TXT = Path("pipeline_google/gemma-4-26b-a4b-it.txt")

model_name = "Gemma 4 26B A4B"
#Gemini 3 Flash
#openai/gpt-5.4
#Claude Sonnet 4.6
#openai/gpt-4o

UPSCALE_TO  = 448
DELAY_S     = 3
MAX_RETRIES = 5
CLAHE_CLIP  = 1.5

# ── CHANGE THESE ──────────────────────────────────────────────────────────────
SEED       = 40   # change to get a different random selection of images
MAX_IMAGES = 500   # change to process more/fewer images (0 = ALL images)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# ── PROMPT ────────────────────────────────────────────────────────────────────
PROMPT = """You are an expert in Eastern Arabic (Hindi-Indic) numerals as handwritten in Egypt.

These digits look NOTHING like Western 0-9. You must perfectly match the stroke structure, corners, and loops to these examples.

════════════════════════════════════════════════════
DIGIT-BY-DIGIT VISUAL GUIDE
════════════════════════════════════════════════════
ZERO (٠, Reference Digit 0):
  - A tiny, centered blob, dot, or diamond.
  - CRITICAL - THE SIZE RULE ‘so you can decide if it five or zero’: Check the overall footprint of the shape compared to the empty background. If the entire shape is just a small dense clump of pixels in the center of the image, it is ONE HUNDRED PERCENT a ZERO (0).
  - Do NOT over-analyze the micro-shape of a tiny speck. Even if a tiny dot has jagged edges that look like "teeth", if it is tiny, it is a Zero.
ONE (١, Reference Digit 1):
  - A single straight vertical or slightly tilted stroke.
  - NO curves, NO horizontal bars, NO hooks, its elongated line.
TWO (٢, Reference Digit 2):
  - A single continuous stroke. It has a curved hook/arch at the top that starts left, arches UP and RIGHT, and then sweeps downwards into a diagonal tail.
At top-left that curves upward and to the right, then the stroke descends diagonally to the bottom-right,ending in an open tail (no flat base). The overall shape resembles a fishhook or a backwards-leaning comma.
  - CRITICAL EDGE CASE 1: Sometimes a Two is drawn with a very flattened top, making it look like a horizontal bar extending RIGHT from a left stem. If a bar extends RIGHT from a left stem, it is a 2 (or 3), NEVER a 6.
  - CRITICAL EDGE CASE 2 : Sometimes the open hook at the top accidentally squishes closed, forming a loop. However, a Two will ALWAYS have a sweeping tail going down-right. Do not confuse heavily looped Twos with Five (which has no tail).

THREE (٣, Reference Digit 3):
  - A vertical/diagonal downward stem on the LEFT.
  - Attached to the right side of this stem's top is a horizontal-ish cluster featuring TWO or THREE distinct, sharp UPWARD-pointing teeth/peaks.
  - CRITICAL: In messy handwriting, these teeth may smudge together into a thick top block. If you see a thick block extending RIGHT from a left vertical stem, it is almost certainly a 2 or 3.

FOUR (٤, Reference Digit 4):
  - A zigzag or curvy stroke that looks like a Western "3" or Greek epsilon (ε).
  - It opens completely to the LEFT, with its solid bulk on the RIGHT.
  - It has no straight vertical stems.
  - CRITICAL DISTINCTION FROM TWO: A Four has roughly EQUAL distribution of ink across the top half and the bottom half of the image. The shape zigzags left-right-left-right symmetrically, with no dominant diagonal flow. The entire shape opens uniformly to the LEFT.
  - A Four typically has 3 horizontal peaks/bumps stacking vertically, and the left edge of the shape is roughly aligned vertically (consistent left boundary).
  - If you see a shape where the top part is LEFT-heavy and the bottom part sweeps RIGHT (diagonal flow), it is a TWO, not a Four.

FIVE (٥, Reference Digit 5):
  - A completely closed, empty loop (like a large 'O' or teardrop).
  - Much larger than a Zero dot. No long tails attached.
  - CRITICAL EDGE CASE: Sometimes messy ink fills or partially streaks the inside of the loop. If the outer boundary is clearly one large continuous circular/oval loop, it is a Five, regardless of noise inside the loop.

SIX (٦, Reference Digit 6):
  - A single structure with a sharp corner. 
  - It has a heavy vertical stem descending on the RIGHT side. From the top of this stem, a straight horizontal bar extends strictly to the LEFT.
  - CRITICAL: The horizontal top bar is on the LEFT of the main vertical stem. This forms a corner at the TOP-RIGHT.
  - If a digit has a top bar that extends to the RIGHT from a left stem, it is absolutely NOT a 6, it is a 2 or 3.

SEVEN (٧, Reference Digit 7):
  - A clear "V" shape. The vertex (point) is at the BOTTOM, opening widely upwards, the v is looking upward.

EIGHT (٨, Reference Digit 8):
  - An inverted "V" shape or caret ( ^ ). The vertex (point) is at the TOP, opening widely downwards.
  - CRITICAL EDGE CASE: In some handwriting, the left arm of the 8 is drawn completely straight vertically down, making it look like a left vertical stem with a diagonal stroke branching down to the right. As long as it forms an inverted V/rooftop shape opening downward, it is an Eight. Do not confuse this with a Three (which must have upward pointing teeth).

NINE (٩, Reference Digit 9):
  - A fully closed loop at the top left, with a distinct sweeping tail extending downwards from the bottom-right of the loop.
 - A closed OR partially open loop (or deep hook) at the TOP of the digit, with a distinct, thick tail/stroke extending DOWNWARD from it.
  - The loop/hook sits on top, the tail hangs below. The overall shape is like a lollipop, lowercase 'q', or a hook.
- 9 and 2 faces opposite sides
  
═══════════════════════════════════════════════════
CRITICAL CONFUSION PAIRS TO CHECK:
════════════════════════════════════════════════════
• 6 vs 2/3: Look at Reference 6. The top horizontal bar extends LEFT from a RIGHT stem. If a top bar extends RIGHT from a LEFT stem, it is a 2 or 3.
• 8 vs 3: If you see a left vertical line and a single branch going down-right, that is a messy 8 (inverted V). A 3 requires teeth pointing UPWARDS on the right.
• 5 vs 2: If a shape is a closed loop BUT has a distinct swooping tail going down, it is a messy 2 (or 9), NOT a 5. A 5 is a pure loop.
• 0 vs Anything: Check the overall size. Tiny speck = 0. Do not hallucinate structure in a tiny dot.

════════════════════════════════════════════════════
STEP-BY-STEP DECISION PROCESS:
════════════════════════════════════════════════════
1. Compare target image to Reference 0: Is it a tiny dot/speck overall? -> ZERO.
2. Compare to Ref 1: Is it an isolated straight line? -> ONE.
3. Compare to Ref 5 & 9: Is it a large enclosed loop? -> FIVE (even if inside is noisy). Does it also have a downward tail? -> NINE or messy TWO.
4. Compare to Ref 7 & 8: Is it a V shape? Vertex at bottom -> SEVEN. Vertex at top -> EIGHT (even if left leg is vertical).
5. Compare to Ref 4: Is it a curly/zigzag shape opening left (like a Western 3)? -> FOUR.
6. Compare to Ref 6: Is there a vertical stem on the RIGHT, with a sharp horizontal bar extending to the LEFT? -> SIX.
7. Compare to Ref 2 & 3: Is there a vertical stem on the LEFT, with features extending to the RIGHT?
   - Just a single sweeping curved arch or flat bar on top? -> TWO.
   - Multiple upward-pointing teeth (or a smudged block of teeth) extending right? -> THREE.

Respond ONLY with a JSON object in this exact format, nothing else:
{
  "digit": <integer 0-9>,
  "confidence": "<HIGH|MEDIUM|LOW>",
  "visual_observations": "<describe exactly how the stroke matches the Reference Image you selected, referencing stem positions, left/right direction, teeth, or corners>",
  "reasoning": "<walk through the decision steps comparing to the references, explicitly stating why confusion pairs were ruled out>",
  "final_answer": "<restate the digit as a single integer>"
}"""
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(path: Path) -> str:
    """Preprocess and base64-encode a BMP digit image."""
    img = Image.open(path).convert("L")
    arr = np.array(img)

    # Denoise
    denoised = cv2.GaussianBlur(arr, (3, 3), 0)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(4, 4))
    enhanced = clahe.apply(denoised)

    # Ensure dark digit on light background
    if np.mean(enhanced) < 127:
        enhanced = cv2.bitwise_not(enhanced)

    # Binarize
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=4
    )

    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Crop to digit bounding box with padding
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 14
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2 = min(x + w + pad, binary.shape[1])
        y2 = min(y + h + pad, binary.shape[0])
        cropped = binary[y1:y2, x1:x2]
    else:
        cropped = binary

    # Invert back: black digit on white
    cropped = cv2.bitwise_not(cropped)

    # Pad to square
    h, w = cropped.shape
    diff = abs(h - w)
    if h > w:
        pl, pr = diff // 2, diff - diff // 2
        cropped = cv2.copyMakeBorder(cropped, 0, 0, pl, pr, cv2.BORDER_CONSTANT, value=255)
    elif w > h:
        pt, pb = diff // 2, diff - diff // 2
        cropped = cv2.copyMakeBorder(cropped, pt, pb, 0, 0, cv2.BORDER_CONSTANT, value=255)

    # Upscale
    final = cv2.resize(cropped, (UPSCALE_TO, UPSCALE_TO), interpolation=cv2.INTER_LANCZOS4)
    final_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)

    buf = io.BytesIO()
    Image.fromarray(final_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_response(response: str, img_id: str) -> tuple[str, str, str, str]:
    """Parse the model JSON response. Returns (digit, confidence, visual_obs, reasoning)."""
    clean = response.strip()
    if clean.startswith("```json"):
        clean = clean[len("```json"):]
    elif clean.startswith("```"):
        clean = clean[len("```"):]
    if clean.endswith("```"):
        clean = clean[:-len("```")]
    clean = clean.strip()
    try:
        data = json.loads(clean)
        digit      = str(data.get("digit", -1))
        confidence = data.get("confidence", "MEDIUM").upper()
        visual_obs = data.get("visual_observations", "")
        reasoning  = data.get("reasoning", "")
        if digit.isdigit() and 0 <= int(digit) <= 9:
            return digit, confidence, visual_obs, reasoning
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: scan for any digit character
    for ch in response:
        if ch.isdigit() and ch != "-":
            return ch, "LOW", "", f"Fallback parse — raw: {response[:120]}"

    logger.warning(f"Could not parse response for {img_id}: {response[:120]}")
    return "-1", "LOW", "", "Parse failed"


class RateLimitExceeded(Exception):
    pass


def query_model(image_b64: str, img_id: str = "", ref_content: list = None) -> tuple[str, str, str, str]:
    """Query the vision model. Returns (digit, confidence, visual_obs, reasoning)."""
    if ref_content is None:
        ref_content = []

    user_content = [
        {"type": "text", "text": PROMPT},
    ] + ref_content + [
        {"type": "text", "text": "\n════════════════════════════════════════════════════\nNow carefully examine this TARGET IMAGE and output the precise JSON response:"},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{image_b64}"
        }},
    ]

    wait = 10
    for attempt in range(MAX_RETRIES):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    # Single user turn: full prompt + image
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                stream=True,
                temperature=0.1,
                max_tokens=400,
            )
            response = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices[0].delta
            )
            return parse_response(response, img_id)

        except Exception as e:
            msg = str(e)
            if "free-models-per-day" in msg:
                raise RateLimitExceeded("Daily limit hit")
            if "No endpoints found" in msg:
                raise RateLimitExceeded("Model has no vision support")
            logger.warning(f"[{attempt+1}/{MAX_RETRIES}] {img_id}: {msg[:80]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
                wait = min(wait * 2, 60)
    return "-1", "LOW", "", "All retries exhausted"


def load_existing(path: Path) -> dict:
    done = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("image_id"):
                    done[row["image_id"]] = row
        logger.info(f"Resuming — {len(done)} already processed.")
    return done


def print_thinking(img_path: Path, gt: str, pred: str, correct: bool, total: int, correct_count: int):
    """Simplified print for the terminal and log."""
    status = "CORRECT" if correct else "WRONG"
    acc = (correct_count / total * 100) if total > 0 else 0.0
    
    line = f"{img_path.name}  | Ground Truth : {gt}   | Prediction: {pred: <4} | {status}"
    print(f"\n{line}")
    print(f"Running accuracy: {correct_count}/{total} = {acc:.2f}%\n")
    return line, acc


def main():
    parser = argparse.ArgumentParser(description="Eastern Arabic Digit Recognizer")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES,
                        help=f"Number of images to process (default: {MAX_IMAGES}, use 0 for ALL)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed for image selection (default: {SEED})")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR),
                        help="Directory containing BMP images")
    parser.add_argument("--gt-json", type=str, default=str(GT_JSON),
                        help="Path to ground_truth.json")
    parser.add_argument("--output-csv", type=str, default=str(OUTPUT_CSV),
                        help="Output CSV file path")
    parser.add_argument("--output-txt", type=str, default=str(OUTPUT_TXT),
                        help="Output TXT file path")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing CSV/TXT results")
    args = parser.parse_args()

    max_images = None if args.max_images == 0 else args.max_images
    image_dir  = Path(args.image_dir)
    gt_json    = Path(args.gt_json)
    output_csv = Path(args.output_csv)
    output_txt = Path(args.output_txt)

    logger.info(f"Seed={args.seed}  Max images={'ALL' if max_images is None else max_images}")
    logger.info(f"Image dir: {image_dir}  |  GT: {gt_json}  |  Out: {output_csv}")
    logger.info(f"Output TXT: {output_txt}")

    # Load ground truth
    gt_data = json.loads(gt_json.read_text(encoding="utf-8"))

    # Collect and shuffle images
    images = sorted(image_dir.glob("*.bmp"))
    images = [p for p in images if p.stem.replace("img_", "") in gt_data]
    if not images:
        logger.error(f"No matching .bmp files found in {image_dir}. Exiting.")
        return

    random.seed(args.seed)
    random.shuffle(images)
    if max_images is not None:
        images = images[:max_images]

    # Pre-encode reference images from example_digits
    ref_content = []
    ref_dir = Path("example_digits")
    if ref_dir.exists():
        # Dynamically load all available digit_*.bmp reference images from example_digits
        all_ref_files = sorted(ref_dir.glob("digit_*.bmp"))
        
        ref_content.append({"type": "text", "text": "\n════════════════════════════════════════════════════\nREFERENCE EXAMPLES\nThese are exactly preprocessed handwritten examples of the digits. Compare the target against these. When variants exist (like 2b, 3c, 9b), they show common alternative handwriting styles.\n════════════════════════════════════════════════════\n"})
        for epath in all_ref_files:
            # Extract the actual digit and variant from the filename 
            # e.g., 'digit_2_b.bmp' -> digit=2, variant='b'
            import re
            match = re.search(r'digit_(\d)(?:_([a-z\d]))?', epath.name)
            if match:
                digit_val = int(match.group(1))
                variant_char = match.group(2) or ""
                eb64 = encode_image(epath)
                
                ref_content.append({"type": "text", "text": f"\nReference Digit {digit_val}{variant_char} ({chr(0x0660 + digit_val)}):"})
                ref_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{eb64}"}})
        logger.info(f"Loaded {len([x for x in ref_content if x['type'] == 'image_url'])} reference images from {ref_dir}.")
    else:
        logger.warning(f"Could not find reference directory {ref_dir}.")

    # Resume support
    done = {} if args.no_resume else load_existing(output_csv)
    need_header = not output_csv.exists() or args.no_resume or not done

    fieldnames = ["image_id", "gt_label", "predicted", "confidence",
                  "correct", "visual_observations", "reasoning"]

    write_mode = "w" if args.no_resume else "a"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, write_mode, newline="", encoding="utf-8") as f_csv, \
         open(output_txt, write_mode, encoding="utf-8") as f_txt:
        
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
            f_txt.write(f"MODEL: {MODEL}\n\n")

        correct_count = sum(1 for r in done.values() if r.get("correct") == "True")
        total = len(done)

        for idx, img_path in enumerate(images, 1):
            img_id   = img_path.stem.replace("img_", "")
            gt_label = str(gt_data[img_id])

            if img_id in done:
                print(f"[{idx:>4}] {img_path.name:<20} — already done, skipping.")
                continue

            try:
                b64 = encode_image(img_path)
                pred, conf, visual_obs, reasoning = query_model(b64, img_id, ref_content=ref_content)
            except RateLimitExceeded as e:
                logger.error(f"Rate limit — stopping. ({e})")
                break
            except Exception as e:
                logger.error(f"{img_id}: {e}")
                pred, conf, visual_obs, reasoning = "-1", "LOW", "", str(e)

            is_correct = gt_label == pred
            total         += 1
            correct_count += int(is_correct)

            writer.writerow({
                "image_id":           img_id,
                "gt_label":           gt_label,
                "predicted":          pred,
                "confidence":         conf,
                "correct":            str(is_correct),
                "visual_observations": visual_obs,
                "reasoning":          reasoning,
            })
            f_csv.flush()

            # Text and Terminal reporting
            log_line, accuracy = print_thinking(img_path, gt_label, pred, is_correct, total, correct_count)
            f_txt.write(f"{log_line}\n")
            f_txt.write(f"Running accuracy: {correct_count}/{total} = {accuracy:.2f}%\n\n")
            f_txt.flush()

            time.sleep(DELAY_S)

    if total:
        print(f"\n{'═'*70}")
        print(f"Model: {model_name}")
        print(f"FINAL RESULTS: {correct_count}/{total} = {correct_count/total*100:.2f}%")
        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()