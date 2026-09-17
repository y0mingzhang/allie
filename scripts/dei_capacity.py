"""Conservative admission snapshot for opportunistic A6000 science on DEI nodes."""
import re,subprocess

def gpu_count(job_text):
 m=re.search(r'\bReqTRES=([^ ]+)',job_text)
 if not m:raise ValueError('Missing ReqTRES; cannot establish DEI demand')
 fields=dict(v.split('=',1) for v in m.group(1).split(',') if '=' in v)
 # The DEI partition consists of A6000 nodes; generic GPU requests count too.
 return int(fields.get('gres/gpu',fields.get('gres/gpu:a6000','0')))

def allowed_a6000(others):
 pending=[r for r in others if r['state']=='PENDING' and r['gpus']>0]
 running=sum(r['gpus'] for r in others if r['state'] in ('RUNNING','COMPLETING','CONFIGURING'))
 return dict(allowed_user_a6000=0 if pending else max(0,16-running),other_running_a6000=running,other_pending_gpu_jobs=pending,other_dei_jobs=others)

def snapshot():
 rows=[]
 raw=subprocess.check_output(['squeue','-q','dei_group_qos','-h','-o','%i|%u|%T|%P'],text=True,timeout=20)
 for line in raw.splitlines():
  jid,user,state,partition=line.split('|')
  if user=='yimingz3' or 'dei-group' not in partition:continue
  detail=subprocess.check_output(['scontrol','show','job','-o',jid],text=True,timeout=20)
  rows.append(dict(job_id=jid,user=user,state=state,gpus=gpu_count(detail),scheduler=detail.strip()))
 return allowed_a6000(rows)
