"""MemoryOS LME evaluation over existing snapshots; no memory construction."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','memory-dir','output-dir','index-dir','embedding-model-path','reranker-model','llm-base-url','judge-base-url'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--llm-model',default='GLM-4.5-Air')
    p.add_argument('--judge-model',default='GLM-5.3')
    p.add_argument('--llm-api-key',default=os.getenv('OPENAI_API_KEY','sk-dummy'))
    p.add_argument('--judge-api-key',default=os.getenv('JUDGE_API_KEY','sk-dummy'))
    p.add_argument('--devices',default='cuda:4,cuda:5,cuda:6,cuda:7')
    p.add_argument('--device',default='cuda:0')
    for name,value in [('workers',128),('judge-workers',16),('candidate-limit',96),('segment-limit',5),('total-limit',10),('knowledge-limit',10),('batch-size',32),('reranker-max-length',192),('rrf-constant',60),('answer-max-tokens',512)]:
        p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--retrieval-backend',choices=['dense','hybrid'],default='dense')
    args=p.parse_args()
    if min(args.workers,args.judge_workers,args.total_limit,args.segment_limit,args.knowledge_limit,args.batch_size,args.answer_max_tokens)<1 or args.candidate_limit<max(args.total_limit,args.segment_limit,args.knowledge_limit):p.error('Invalid limits')
    os.environ['FUSIONRAG']='false'
    from bge_retrieval import BGERetriever
    from short_term_memory import ShortTermMemory
    from long_term_memory import LongTermMemory
    from main_longevalmem_parse import generate_system_response_longmemeval
    from token_consumption import llm_judge
    from openai import OpenAI
    engine=BGERetriever(args)
    api=OpenAI(api_key=args.llm_api_key,base_url=args.llm_base_url,timeout=300)
    class Reader:
        def chat_completion_with_usage(self,model,messages,temperature=0.7,max_tokens=2000):
            # Original LME helper hardcodes qwen3-8b; enforce the requested model
            # here while reusing its exact prompt construction.
            response=api.chat.completions.create(model=args.llm_model,messages=messages,
                temperature=temperature,max_tokens=args.answer_max_tokens,
                extra_body={'chat_template_kwargs':{'enable_thinking':False,'thinking':False}})
            return response.choices[0].message.content or '',response.usage.prompt_tokens,response.usage.completion_tokens,None,None
    reader=Reader();judge_slots=threading.BoundedSemaphore(args.judge_workers)
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if 'api_key' not in k}
    cf=root/'retrieval_config.json'
    if cf.exists() and json.loads(cf.read_text())!=config:p.error('Use a new output directory')
    cf.write_text(json.dumps(config,indent=2))
    items=json.loads(Path(args.dataset).read_text())
    def worker(item):
        sid=str(item['question_id']);output=root/f'result_{sid}.json'
        record=json.loads(output.read_text()) if output.exists() else None
        if record and record.get('judge_status')=='ok':return record
        if not record:
            paths={kind:Path(args.memory_dir)/f'{sid}_{kind}_term.json' for kind in ('short','mid','long')}
            for path in paths.values():
                if not path.is_file():raise FileNotFoundError(path)
            short=ShortTermMemory(max_capacity=5,file_path=str(paths['short']))
            long=LongTermMemory(embedding_model=None,file_path=str(paths['long']))
            sessions=json.loads(paths['mid'].read_text()).get('sessions',{})
            segments=[dict(id=k,payload=dict(memory=v.get('summary',''))) for k,v in sessions.items()]
            selected_segments=engine.retrieve(segments,item['question'],args.segment_limit)
            pages=[]
            for selected in selected_segments:
                for i,page in enumerate(sessions[selected['id']].get('details',[])):
                    text=f"{page.get('user_input','')}\n{page.get('timestamp','')}\n{page.get('agent_response','')}"
                    pages.append(dict(id=f"{selected['id']}:{i}",payload=dict(memory=text),page=page))
            selected_pages=engine.retrieve(pages,item['question'],args.total_limit)
            knowledge=[dict(id=f'knowledge:{i}',payload=dict(memory=k['knowledge']),knowledge=k) for i,k in enumerate(long.get_knowledge())]
            selected_knowledge=engine.retrieve(knowledge,item['question'],args.knowledge_limit)
            answer,system,prompt,pt,ct=generate_system_response_longmemeval(item['question'],item['question_date'],short,long,
                [e['page'] for e in selected_pages],[e['knowledge'] for e in selected_knowledge],reader,sid)
            record=dict(question_id=sid,question_type=item['question_type'],question=item['question'],
                system_answer=answer,golden_answer=item['answer'],reference=item['answer'],
                prompt_tokens=pt,completion_tokens=ct,system_prompt=system,user_prompt=prompt,
                selected_segments=[e['id'] for e in selected_segments],selected_pages=[e['id'] for e in selected_pages],
                selected_knowledge=[e['id'] for e in selected_knowledge],judge_status='pending',correct=None)
            output.write_text(json.dumps(record,indent=2))
        for attempt in range(5):
            try:
                with judge_slots:
                    correct=llm_judge(item['question'],str(item['answer']),record['system_answer'],model=args.judge_model,endpoint=args.judge_base_url,api_key=args.judge_api_key)
                record.update(correct=bool(correct),judge_status='ok',judge_error=None);break
            except Exception as exc:
                record.update(correct=None,judge_status='failed',judge_error=str(exc))
                print(f'[Judge error] {sid} {attempt+1}/5: {exc}',flush=True)
                if attempt<4:time.sleep(min(2**attempt,8))
        output.write_text(json.dumps(record,indent=2));return record
    results=[];errors=[]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,item):item['question_id'] for item in items}
            for future in as_completed(futures):
                try:results.append(future.result())
                except Exception as exc:errors.append(dict(question_id=futures[future],error=str(exc)));print(errors[-1],flush=True)
                (root/'longmemeval_result.json').write_text(json.dumps(results,indent=2))
    finally:engine.close()
    if errors:
        (root/'errors.json').write_text(json.dumps(errors,indent=2));raise RuntimeError(f'{len(errors)} questions failed; see errors.json')


if __name__=='__main__':main()
