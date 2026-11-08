"""Read-only BGE/reranker evaluation of existing MemoryOS memory snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

from bge_retrieval import BGERetriever
from token_consumption import llm_judge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('memory-dir', 'output-dir', 'index-dir', 'embedding-model-path',
                 'reranker-model', 'llm-base-url', 'judge-base-url'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--dataset', default='locomo10.json')
    p.add_argument('--llm-model', default='GLM-4.5-Air')
    p.add_argument('--judge-model', default='GLM-5.3')
    p.add_argument('--llm-api-key', default=os.getenv('OPENAI_API_KEY', 'sk-dummy'))
    p.add_argument('--judge-api-key', default=os.getenv('JUDGE_API_KEY', 'sk-dummy'))
    p.add_argument('--devices', default='cuda:0')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=128)
    p.add_argument('--judge-workers', type=int, default=8)
    p.add_argument('--candidate-limit', type=int, default=96)
    p.add_argument('--segment-limit', type=int, default=5)
    p.add_argument('--total-limit', type=int, default=10)
    p.add_argument('--knowledge-limit', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--reranker-max-length', type=int, default=192)
    p.add_argument('--retrieval-backend', choices=['dense', 'hybrid'], default='dense')
    p.add_argument('--rrf-constant', type=int, default=60)
    args = p.parse_args()
    if min(args.workers, args.judge_workers, args.candidate_limit, args.segment_limit,
           args.total_limit, args.knowledge_limit, args.batch_size) < 1:
        p.error('Concurrency, candidate and selection limits must be positive')
    if args.candidate_limit < max(args.segment_limit, args.total_limit, args.knowledge_limit):
        p.error('candidate-limit must cover selection limits')
    os.environ['FUSIONRAG'] = 'false'
    os.environ['DUMP_QUESTIONS'] = 'false'
    # Import the original reader only in the parent. Reuse its evidence formatting,
    # native API invocation and token accounting without running its build/main.
    import main_loco_parse as original
    from utils import OpenAIClient
    from short_term_memory import ShortTermMemory
    from long_term_memory import LongTermMemory
    original.LLM_MODEL = args.llm_model
    original.API_BASE_URL = args.llm_base_url
    client = OpenAIClient(api_key=args.llm_api_key, base_url=args.llm_base_url,
                          recomputation_rate=0.0)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if 'api_key' not in k}
    config_path = root / 'retrieval_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        p.error('Output configuration differs; use a new output directory')
    config_path.write_text(json.dumps(config, indent=2))
    engine = BGERetriever(args)
    judge_slots = threading.BoundedSemaphore(args.judge_workers)
    all_results = []
    try:
        for sample in json.loads(Path(args.dataset).read_text()):
            sid = sample['sample_id']
            base = Path(args.memory_dir)
            paths = {kind: base / f'{sid}_{kind}_term.json' for kind in ('short', 'mid', 'long')}
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(f'Missing built memory: {path}')
            sessions = json.loads(paths['mid'].read_text()).get('sessions', {})
            short = ShortTermMemory(max_capacity=5, file_path=str(paths['short']))
            long = LongTermMemory(embedding_model=None, file_path=str(paths['long']))
            # Session summaries select sessions; only their QA pages enter prompt.
            segments, pages = [], {}
            for segment_id, session in sessions.items():
                segments.append(dict(id=segment_id, payload=dict(memory=session.get('summary', ''))))
                pages[segment_id] = []
                for i, page in enumerate(session.get('details', [])):
                    text = (f"{page.get('user_input', '')}\n{page.get('timestamp', '')}\n"
                            f"{page.get('agent_response', '')}")
                    pages[segment_id].append(dict(id=f'{segment_id}:{i}',
                        payload=dict(memory=text), page=page))
            knowledge = [dict(id=f'knowledge:{i}', payload=dict(memory=k['knowledge']), knowledge=k)
                         for i, k in enumerate(long.get_knowledge())]
            # Index every channel once, before concurrent questions. No old vectors
            # are used and no original JSON snapshots are modified.
            for entries in [segments, knowledge] + list(pages.values()):
                if entries:
                    engine.retrieve(entries, 'index initialization', limit=1)
            output = root / f'sample_{sid}.json'
            results = []
            if output.exists():
                results = json.loads(output.read_text())['results']
            done = {r['qa_idx'] for r in results}
            def worker(index, qa):
                question = qa['question']
                selected_segments = engine.retrieve(segments, question, args.segment_limit)
                candidates = [page for seg in selected_segments for page in pages[seg['id']]]
                selected_pages = engine.retrieve(candidates, question, args.total_limit)
                selected_knowledge = engine.retrieve(knowledge, question, args.knowledge_limit)
                conversation = sample['conversation']
                answer, system, prompt, pt, ct, elapsed, _ = original.generate_system_response_with_meta(
                    question, short, long, [e['page'] for e in selected_pages],
                    [e['knowledge'] for e in selected_knowledge], client, sid,
                    conversation['speaker_a'], conversation['speaker_b'], {})
                reference = qa.get('answer') or qa.get('adversarial_answer', '')
                with judge_slots:
                    score = llm_judge(question, str(reference), answer, model=args.judge_model,
                        api_key=args.judge_api_key, endpoint=args.judge_base_url)
                return dict(sample_id=sid, qa_idx=index, question=question,
                    prediction=answer, reference=reference, system_answer=answer,
                    original_answer=reference, category=qa['category'], correct=bool(score),
                    metrics=dict(judge_correct=int(score)),
                    prompt_tokens=pt, completion_tokens=ct,
                    token_usage=dict(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt+ct),
                    answer_time=elapsed, system_prompt=system, user_prompt=prompt,
                    selected_segments=[e['id'] for e in selected_segments],
                    selected_pages=[e['id'] for e in selected_pages],
                    selected_knowledge=[e['id'] for e in selected_knowledge])
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(worker, i, qa) for i, qa in enumerate(sample['qa'])
                           if i not in done and qa.get('category') in (1, 2, 3, 4)]
                for future in as_completed(futures):
                    results.append(future.result())
                    results.sort(key=lambda r: r['qa_idx'])
                    temporary = output.with_suffix('.tmp')
                    temporary.write_text(json.dumps(dict(sample_id=sid, results=results), indent=2))
                    temporary.replace(output)
            all_results.extend(results)
            # Compatible with the repository's existing flat-list statistics.
            (root / 'locomo_result.json').write_text(json.dumps(all_results, indent=2))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
