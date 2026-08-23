# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# DATASET
# ============================================================

documents = [
    # Doc 0
    "การรักษาโรคฟันผุและรากฟันอักเสบในเด็กเล็ก "
    "ต้องพบทันตแพทย์เฉพาะทางทันที",

    # Doc 1
    "กุมารแพทย์แนะนำว่า เมื่อลูกน้อยมีอาการปวดซี่ฟันรุนแรง"
    "เวลาดื่มน้ำเย็น อาจเกิดจากสารเคลือบฟันสึกกร่อน",

    # Doc 2
    "ข้อควรระวังในการใช้ยา Ibuprofen 400mg "
    "เพื่อบรรเทาอาการปวดฟันในผู้ป่วยโรคกระเพาะ",

    # Doc 3
    "สิทธิการเบิกจ่ายประกันสุขภาพหมวดทันตกรรม "
    "รหัส D-9902 สำหรับเคสอุดฟันด้วยวัสดุสีเหมือนฟัน",

    # Doc 4
    "แนวทางการดูแลสุขภาพช่องปาก ป้องกันกลิ่นปากและคราบพลัค "
    "ด้วยการแปรงฟันอย่างถูกวิธีวันละ 2 ครั้ง",

    # Doc 5
    "การดูแลและรักษาโรคไข้หวัดใหญ่ในเด็กช่วงฤดูฝน "
    "ต้องระวังภาวะช็อกเมื่อไข้สูง",
]


queries = [
    {
        "name": "Q1",
        "type": "Synonym/Context",
        "query": "ลูกน้อยเสียวฟันเวลาทานของเย็นทำยังไงดี",
        "ground_truth": 1,
    },

    {
        "name": "Q2",
        "type": "Exact Code Match",
        "query": "รหัสเบิกประกันอุดฟัน D-9902",
        "ground_truth": 3,
    },

    {
        "name": "Q3",
        "type": "Hybrid Keyword & Intent",
        "query": "ยาลดปวด Ibuprofen กินแก้ปวดซี่ฟันได้ไหม",
        "ground_truth": 2,
    },
]


# ============================================================
# MODELS
# ============================================================

models = {
    "E5-Small": {
        "name": "intfloat/multilingual-e5-small",
        "prefix": True,
    },

    "E5-Large": {
        "name": "intfloat/multilingual-e5-large",
        "prefix": True,
    },

    "BGE-M3": {
        "name": "BAAI/bge-m3",
        "prefix": False,
    },

    "MPNet-Multi": {
        "name":
            "sentence-transformers/"
            "paraphrase-multilingual-mpnet-base-v2",
        "prefix": False,
    },
}


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_scores(query_embedding, doc_embeddings):
    """
    Embeddings ถูก normalize แล้ว
    ดังนั้น dot product = cosine similarity
    """

    return np.dot(
        doc_embeddings,
        query_embedding
    )


# ============================================================
# BENCHMARK
# ============================================================

results = []
model_summary = []


for model_label, config in models.items():

    print()
    print("=" * 100)
    print(f"MODEL: {model_label}")
    print(f"HuggingFace: {config['name']}")
    print("=" * 100)


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    load_start = time.perf_counter()

    model = SentenceTransformer(
        config["name"]
    )

    load_time = (
        time.perf_counter()
        - load_start
    )


    print(
        f"Model load time: "
        f"{load_time:.3f} sec"
    )


    # --------------------------------------------------------
    # PREFIX
    # --------------------------------------------------------

    if config["prefix"]:

        prepared_documents = [
            "passage: " + doc
            for doc in documents
        ]

    else:

        prepared_documents = documents


    # --------------------------------------------------------
    # DOCUMENT EMBEDDING
    # --------------------------------------------------------

    doc_start = time.perf_counter()

    doc_embeddings = model.encode(
        prepared_documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    doc_encode_time = (
        time.perf_counter()
        - doc_start
    )


    print(
        f"Document encoding: "
        f"{doc_encode_time:.4f} sec"
    )

    print(
        f"Embedding dimension: "
        f"{doc_embeddings.shape[1]}"
    )


    # --------------------------------------------------------
    # QUERY SEARCH
    # --------------------------------------------------------

    correct_count = 0
    query_times = []


    for item in queries:

        query = item["query"]
        ground_truth = item["ground_truth"]


        if config["prefix"]:

            prepared_query = (
                "query: " + query
            )

        else:

            prepared_query = query


        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_start = time.perf_counter()

        query_embedding = model.encode(
            prepared_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_time = (
            time.perf_counter()
            - query_start
        )


        query_times.append(
            query_time
        )


        # ----------------------------------------------------
        # Similarity
        # ----------------------------------------------------

        scores = cosine_scores(
            query_embedding,
            doc_embeddings
        )


        ranking = np.argsort(
            scores
        )[::-1]


        top1_doc = int(
            ranking[0]
        )


        top1_score = float(
            scores[top1_doc]
        )


        gt_score = float(
            scores[ground_truth]
        )


        # Ground Truth rank
        gt_rank = int(
            np.where(
                ranking == ground_truth
            )[0][0]
            + 1
        )


        correct = (
            top1_doc
            ==
            ground_truth
        )


        if correct:
            correct_count += 1


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({
            "Model": model_label,
            "Model_Name": config["name"],

            "Query": item["name"],
            "Query_Type": item["type"],
            "Query_Text": query,

            "Ground_Truth": f"Doc {ground_truth}",

            "Top1_Doc": f"Doc {top1_doc}",
            "Top1_Score": top1_score,

            "GT_Rank": gt_rank,
            "GT_Score": gt_score,

            "Correct_Top1": correct,

            "Query_Time_ms":
                query_time * 1000,

            "Embedding_Dimension":
                doc_embeddings.shape[1],
        })


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        status = (
            "✓"
            if correct
            else "✗"
        )


        print()

        print(
            f"{item['name']} "
            f"({item['type']})"
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Ground Truth: "
            f"Doc {ground_truth}"
        )

        print(
            f"Top-1: "
            f"Doc {top1_doc} "
            f"| Score={top1_score:.6f} "
            f"| {status}"
        )

        print(
            f"GT Rank: {gt_rank} "
            f"| GT Score={gt_score:.6f}"
        )

        print(
            f"Query inference: "
            f"{query_time * 1000:.2f} ms"
        )


        print("Ranking:")

        for rank, doc_idx in enumerate(
            ranking,
            start=1
        ):

            print(
                f"  Rank {rank} "
                f"| Doc {doc_idx} "
                f"| Score "
                f"{scores[doc_idx]:.6f}"
            )


    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    avg_query_ms = (
        np.mean(query_times)
        * 1000
    )


    accuracy = (
        correct_count
        /
        len(queries)
    )


    model_summary.append({
        "Model": model_label,
        "Model_Name": config["name"],

        "Top1_Correct":
            correct_count,

        "Total_Queries":
            len(queries),

        "Top1_Accuracy":
            accuracy,

        "Load_Time_sec":
            load_time,

        "Document_Encode_sec":
            doc_encode_time,

        "Avg_Query_Time_ms":
            avg_query_ms,

        "Embedding_Dimension":
            doc_embeddings.shape[1],
    })


    print()
    print("-" * 100)

    print(
        f"{model_label} SUMMARY"
    )

    print(
        f"Top-1 Accuracy: "
        f"{correct_count}/{len(queries)} "
        f"= {accuracy:.2%}"
    )

    print(
        f"Average Query Time: "
        f"{avg_query_ms:.2f} ms"
    )

    print("-" * 100)


# ============================================================
# DATAFRAMES
# ============================================================

result_df = pd.DataFrame(
    results
)

summary_df = pd.DataFrame(
    model_summary
)


# ============================================================
# PIVOT TABLE FOR ASSIGNMENT
# ============================================================

rank_table = result_df.pivot(
    index=[
        "Query",
        "Ground_Truth",
    ],
    columns="Model",
    values="GT_Rank",
)


# จัดลำดับ column
model_order = [
    "E5-Small",
    "E5-Large",
    "BGE-M3",
    "MPNet-Multi",
]

rank_table = rank_table.reindex(
    columns=model_order
)


# ============================================================
# SAVE CSV
# ============================================================

result_df.to_csv(
    "dense_model_comparison_detail.csv",
    index=False,
    encoding="utf-8-sig",
)


summary_df.to_csv(
    "dense_model_comparison_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


rank_table.to_csv(
    "dense_model_rank_table.csv",
    encoding="utf-8-sig",
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print()
print("=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

print(
    summary_df[
        [
            "Model",
            "Top1_Correct",
            "Total_Queries",
            "Top1_Accuracy",
            "Avg_Query_Time_ms",
            "Embedding_Dimension",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("GROUND TRUTH RANK TABLE")
print("=" * 100)

print(
    rank_table.to_string()
)


print()
print("CSV CREATED:")
print(
    "  dense_model_comparison_detail.csv"
)

print(
    "  dense_model_comparison_summary.csv"
)

print(
    "  dense_model_rank_table.csv"
)