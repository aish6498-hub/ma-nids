# Preprocessing Notes - MA-NIDS

---

## What Preprocessing Does
Takes raw messy CSV files → produces one clean, balanced, model-ready CSV that all three agents load from.
**Rule: preprocess once, everyone loads the same file. Never preprocess again.**

---

## The Raw Dataset - CIC-IDS2018
- 9 CSV files, one per day of captured network traffic
- Each row = one network flow (a connection between two machines)
- Each column = one feature of that flow (packet size, duration, port, etc.)
- One special column: `Label` - tells you what type of traffic it was

---

## File Excluded - 02-20-2018.csv
- Has **84 columns**, all others have **80 columns**
- Extra columns: `Src IP`, `Dst IP`, `Src Port`, `Flow ID`
- These are network identifiers, not behaviour features
- Including this file would corrupt the feature space with a column mismatch
- **Decision: exclude it. Infiltration traffic it contains is already in other files.**

---

## Step 1 - Load Each File Separately (Memory Safety)
- Loading all files at once would need ~20GB RAM → crash
- Instead: load one file → sample it → delete from RAM → load next
- `del df` after each file explicitly frees memory

---

## Step 2 - Column Name Cleaning
```
df.columns = df.columns.str.strip()
```
- CIC-IDS2018 has hidden spaces in column names e.g. `" Label"` not `"Label"`
- `.str.strip()` removes leading/trailing spaces from all column names silently
- Without this, `df["Label"]` would throw a KeyError

---

## Step 3 - Drop Identifier Columns
Dropped: `Timestamp`, `Dst IP`, `Src IP`, `Flow ID`
- These are addresses and IDs, not traffic behaviour
- If kept, model might learn "IP 10.0.0.5 = attack" instead of actual patterns
- This is called **data leakage** - model learns something it can't know in deployment

---

## Step 4 - Clean Bad Values
**Missing labels** → drop rows with no label (can't learn from unlabelled data)

**Non-numeric values** → `errors="coerce"` converts bad values to NaN silently
- e.g. stray "N/A" text in a numeric column becomes NaN instead of crashing

**Infinity values** → replace with NaN then drop
- Some features are ratios (bytes/second) → division by zero → infinity
- Models crash or produce wrong results on infinity

---

## Step 5 - Drop Rare Classes
These classes had too few samples to train reliably:
```
Brute Force -Web       → dropped
Brute Force -XSS       → dropped
DDOS attack-LOIC-UDP   → dropped
SQL Injection          → dropped
DoS attacks-Slowloris  → dropped (only ~10k samples)
"Label"                → dropped (data artifact from bad concatenation)
```
**Rule used: drop any class with fewer than 10,000 samples**

Also dropped automatically by `errors="coerce"` + `dropna()`:
- The `"Label"` artifact rows (rogue header rows from concatenation)

---

## Step 6 - Balance Classes (Two Passes)

**Why balance?**
Without balancing: Benign had 320,000 rows, attacks had 40,000.
A model trained on this learns "always say Benign" → 50% accuracy, catches zero attacks.

**Pass 1 - Per file sampling**
Sample up to 40,000 rows per class within each file.
Problem: Benign appears in every file → still ends up with 320,000 rows combined.

**Pass 2 - Global re-sampling after combining**
Sample up to 40,000 rows per class across the full combined dataset.
This is the pass that actually enforces balance.

**Final result: 8 classes × 40,000 rows = 320,000 rows total**

---

## Step 7 - Global Shuffle
```python
combined.sample(frac=1, random_state=42)
```
- Without shuffling: all Benign rows come first, then all DDOS, etc.
- Models trained on ordered data learn order patterns, not traffic patterns
- `frac=1` = sample 100% of rows = shuffle everything

---

## Step 8 - Label Encoding
```
Benign                   → 0
DDOS attack-HOIC         → 1
DoS attacks-GoldenEye    → 2
DoS attacks-Hulk         → 3
DoS attacks-SlowHTTPTest → 4
FTP-BruteForce           → 5
Infilteration            → 6
SSH-Bruteforce           → 7
```
- Models need numbers, not strings
- LabelEncoder assigns integers alphabetically
- **Encoder saved separately as `label_encoder.pkl`**
- Without the encoder, you can never map "3" back to "DoS attacks-Hulk"

---

## Final Output
```
data/processed/
├── cleaned_data.csv        ← 320,000 rows, 79 columns (78 features + Label)
└── label_encoder.pkl       ← maps integers back to class names
```

---

## Key Decisions Summary

| Decision | What | Why |
|---|---|---|
| Exclude 02-20-2018.csv | Different column count | Would corrupt feature space |
| Process files one at a time | Memory safety | 20GB+ needed otherwise |
| Drop identifier columns | Src IP, Dst IP etc. | Data leakage risk |
| Drop rare classes | < 10,000 samples | Too few to train reliably |
| Two-pass balancing | Per file + global | Per file alone didn't fix Benign imbalance |
| Global shuffle | Randomise row order | Prevents model learning order patterns |
| Save encoder separately | label_encoder.pkl | Needed by Agent 2 and Agent 3 to decode predictions |
