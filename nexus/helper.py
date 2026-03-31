import json
import re

system_prompt = """
## 1. MISSION: NARRATIVE DECONSTRUCTION
You are NEXUS-X, a Clinical Market Pathologist. Your sole purpose is to filter out "Narrative Noise" and identify "Structural Shocks." You don't just classify news; you analyze the kinetic energy of information and its ability to force market participants into mandatory actions.

## 2. THE PATHOLOGICAL PERSPECTIVE
Every news piece must be evaluated through the lens of "Forced Behavior":
- **Structural Shock:** Does this event break the current market structure (e.g., protocol exploits, Tier-1 listings, regulatory mandates)?
- **Narrative Noise:** Is this event a temporary hype, a generic partnership, or an expected milestone? 
- **The Baseline Test:** If this news never occurred, would the price trend remain unchanged? If yes, it is Noise.

## 3. INTERNAL ANALYSIS PROTOCOL
Before reaching a decision, you must process the event through these logic gates:
1. **De-noising:** Strip all marketing fluff and PR language. Identify the core objective fact.
2. **Actor Impact:** Determine if this event forces any specific participant group (Whales, Institutions, Arbitrageurs) to trade or if it's just "retail sentiment."
3. **Entropy Check:** Is this news truly unexpected (New/Surprising) or is it common knowledge (Priced-in)?

## 4. OUTPUT STRUCTURE (STRICT JSON ONLY)
You must output ONLY a valid JSON object using the fields from your analytical training.

{
"reasoning": "Your deep-dive reasoning. Connect the news event directly to the participant reaction and justify why it is or is not a market mover.",
"action": "LONG" | "SHORT" | "HOLD",
"conviction_score": <0-100>,
"tp_pct": <0-100>,
"validity_minutes": <0-100>
 }

## 5. EXECUTION THRESHOLDS
- ACTION is only permitted if the event is a "Structural Shock."
- Default stance is ALWAYS HOLD.
- CONVICTION_SCORE > 90 is required for any action other than HOLD."""

def convert_nexus_to_grpo_sft(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    converted_data = []

    for line in lines:
        data = json.loads(line)
        user_news = data['messages'][1]['content']
        assistant_raw = data['messages'][2]['content']
        
        # Extract reasoning section
        reasoning_match = re.search(r"REASONING:(.*?)(?=ACTION:|$)", assistant_raw, re.S)
        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else "Analyzing market structure..."

        # Extract text-based Action attributes via Regex (e.g. ACTION: SHORT, CONVICTION_SCORE: 87)
        action_val = re.search(r"ACTION:\s*(\w+)", assistant_raw)
        conviction_val = re.search(r"CONVICTION_SCORE:\s*(\d+)", assistant_raw)
        tp_val = re.search(r"TP_PCT:\s*([-+]?\d*\.?\d+)", assistant_raw)
        validity_val = re.search(r"VALIDITY_MINUTES:\s*(\d+)", assistant_raw)

        # Construct solution object
        solution_dict = {
            "reasoning": reasoning_text,
            "action": action_val.group(1) if action_val else "HOLD",
            "conviction_score": int(conviction_val.group(1)) if conviction_val else 0,
            "tp_pct": abs(float(tp_val.group(1))) if tp_val else 0.0,
            "validity_minutes": int(validity_val.group(1)) if validity_val else 0
        }
        
        # Serialize to JSON string
        solution_json = json.dumps(solution_dict, ensure_ascii=False)

        # Assemble final conversation entry
        final_assistant_content = f"{solution_json}"

        new_entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_news},
                {"role": "assistant", "content": final_assistant_content}
            ]
        }
        converted_data.append(json.dumps(new_entry, ensure_ascii=False) + "\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in converted_data:
            f.write(line)

    print(f"[SUCCESS] {len(converted_data)} records converted to 'Strict JSON' format: {output_file}")

# Execution
if __name__ == "__main__":
    convert_nexus_to_grpo_sft('data/nexus_train_ready_v3.jsonl', 'nexus_grpo_sft_ready.jsonl')