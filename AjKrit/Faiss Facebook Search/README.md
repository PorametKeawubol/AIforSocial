# FAISS Facebook Search — Thai Model Comparison

This submission implements the supplied lab sheet: compare three Thai-capable
sentence encoders with exact FAISS top-1 search and verify the relationship
between cosine similarity and Euclidean distance.

## What the program does

1. Encodes the five Thai corpus texts with each target model.
2. L2-normalizes both corpus and query embeddings.
3. Searches the same vectors with `faiss.IndexFlatIP` (cosine similarity) and
   `faiss.IndexFlatL2` (squared Euclidean distance).
4. Verifies the normalized-vector identities:

   ```text
   cosine = 1 - squared_L2 / 2
   L2 = sqrt(2 * (1 - cosine))
   ```

5. Saves a display-ready CSV and full-precision JSON result file.

## Run

Use any Python environment containing the packages in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
python3 benchmark_faiss.py
```

The first run downloads the three Hugging Face models.  To try another query:

```bash
python3 benchmark_faiss.py --query "สำนักงานใหญ่ตั้งอยู่ที่ไหน?"
```

## Output

- `faiss_model_comparison_results.csv` — result table for the lab report.
- `faiss_model_comparison_results.json` — unrounded values, indexes, and
  verification errors.

`IndexFlatIP` returns the inner product.  Because both vector sets are
L2-normalized, it is the cosine similarity.  `IndexFlatL2` returns *squared*
L2 distance, so the program takes its square root before reporting the ordinary
Euclidean distance.
