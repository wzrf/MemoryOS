#!/usr/bin/env bash
# Run in the repository's existing activated Python environment.
DRAFT_MODEL_PATH=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct/ PYTHONPATH=FusionRAG/ FUSIONRAG=false python3 search_longmemeval_BGE_rerank.py \
  --dataset /mnt/qjhs-sh-lab-01/eli/lme_complete_bge/longmemeval_s_complete_common.json \
  --memory-dir ./mem_tmp_longmemeval_GLM-4.5-Air \
  --output-dir ./results_longmemeval_GLM_BGE_rerank_complete_common --index-dir ./indexes/longmemeval_GLM_BGE \
  --embedding-model-path /mnt/qjhs-sh-lab-01/models/bge-m3 \
  --reranker-model /mnt/qjhs-sh-lab-04/models/bge-reranker-v2-m3 \
  --devices cuda:4,cuda:5,cuda:6,cuda:7 \
  --candidate-limit 96 --segment-limit 5 --total-limit 10 --knowledge-limit 10 --answer-max-tokens 512 \
  --batch-size 32 --reranker-max-length 192 --workers 128 --judge-workers 16 \
  --llm-base-url http://127.0.0.1:30013/v1 --llm-model GLM-4.5-Air \
  --judge-base-url http://192.168.0.239:31002/v1 --judge-model GLM-5.3
