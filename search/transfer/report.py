"""Export the finite transfer comparison; no best-method selection or new runs."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from search.engine.service import ROOT,atomic
from search.engine.analyze_balanced import cellmean,bootstrap_deltas
from search.engine.balanced_eval import digest
from search.engine.budget_policy import policy
from .collect import OUT,inventory,BUDGETS,freeze
from .analyze import data,score
from .cost import compute as compute_cost
from search.training_cm import excess

SHOW=['port_raw','legal','calibrated_direct','allie_released','allie_repaired','two_ply','four_ply','frozen_64','frozen_128','frozen_256','frozen_512','adaptive_frozen','frozen_1000','frozen_projected','refit_1000']
LABEL={'port_raw':'Served raw','legal':'Legal policy','calibrated_direct':'Calibrated direct','allie_released':'Allie adaptive (released)','allie_repaired':'Allie repaired (50)','two_ply':'2-ply expectation','four_ply':'4-ply expectation','adaptive_frozen':'Adaptive Elo budget','frozen_projected':'+ critic consistency','refit_1000':'Search + simple refit'}
LABEL.update({f'frozen_{b}':f'Coverage search ({b})' for b in BUDGETS})
LABEL.update({f'refit_{b}':f'Coverage search ({b}) + simple refit' for b in BUDGETS})
LABEL['refit_1000']='Search + simple refit'


def takeaways(reports):
    large=reports['large']['methods'];a=large['adaptive_frozen'];f=large['frozen_1000']
    reduction=100*(1-a['mean_nodes']/f['mean_nodes'])
    out=['## Practical reading of the frontier','',
        '**Use the frozen adaptive budget as the quality/cost reference for further work.** On the 129M checkpoint it uses '+f"{a['mean_nodes']:.0f} rather than {f['mean_nodes']:.0f} NN evaluations ({reduction:.1f}% fewer), with no resolved CE difference from fixed 1,000. "+'That supports the cheaper operating point, not a claim that it is statistically better. For a tighter budget, 128 nominal simulations already gives most of the macro gain; additional search mainly improves expert CE.','',
        'Coverage search at 128 nominal simulations uses fewer nodes and beats both tested fixed-ply references on both metrics at both model sizes (exploratory paired 95% intervals). The released Allie reference is worse than the legal prior here; its repaired reference also has worse point estimates. These are frozen transfers of particular configurations, not a proof that all time-adaptive MCTS is ineffective.','',
        'The critic-consistency correction has unresolved small gains. The simple output refit improves large-model macro at the cost of a worse expert point estimate. Neither justifies replacing the frozen reference from these reused-sample results alone.','',
        '| 129M comparison to frozen 1,000 | Macro CE difference | Paired 95% CI | Expert CE difference | Paired 95% CI |',
        '|---|---:|---|---:|---|']
    for key in ['adaptive_frozen','frozen_projected','refit_1000']:
        r=large[key];values=[]
        for metric in ['macro','expert_macro']:
            d=r[metric+'_delta_vs_frozen_1000'];lo,hi=r[metric+'_delta_vs_frozen_1000_ci95'];values += [f'{d:+.5f}',f'{lo:+.5f}, {hi:+.5f}']
        out.append('| '+LABEL[key]+' | '+' | '.join(values)+' |')
    out += ['', 'Negative differences favor the named method. Intervals are evaluation-only, unadjusted for the reported family of methods.','']
    return out


def point_frontiers(reports):
    result={}
    for size,report in reports.items():
        ms=report['methods'];keys=list(dict.fromkeys(SHOW+[f'refit_{b}' for b in BUDGETS]))
        front={}
        for name,axes in [('macro',['mean_nodes','macro']),('expert',['mean_nodes','expert_macro']),('joint',['mean_nodes','macro','expert_macro'])]:
            selected=[]
            for key in keys:
                dominated=any(all(ms[other][a]<=ms[key][a] for a in axes) and any(ms[other][a]<ms[key][a] for a in axes) for other in keys if other!=key)
                if not dominated:selected.append(key)
            front[name]=sorted(selected,key=lambda k:(ms[k]['mean_nodes'],ms[k]['macro']))
        result[size]=front
    atomic(OUT/'frontiers.json',dict(point_estimates=result,note='Non-dominated among the declared representative methods only. These are not significance tests or selection decisions. Nodes are sample-average new NN evaluations; root prefill is separate.'))


def execution_notes(reports):
    audit=json.loads((OUT/'order-large-general/results.json').read_text())
    state=json.loads((ROOT/'status.json').read_text());job=next(j for j in state['jobs'] if j['id']=='10505672')
    accounting=dict(job='10505672',allocations=job['allocations'],elapsed_seconds=job['elapsed_seconds'],gpu_hours=job['gpu_hours'],prior_gpu_hours=job['prior_cumulative_gpu_hours'],cumulative_gpu_hours=state['gpu_hours'],state=job['state'],note='Allocated time, including staging, startup, idle and preempted work; no previous charge reset. Slurm timestamps are America/New_York.')
    atomic(OUT/'accounting.json',accounting)
    out=['## Runtime and numerical checks','',
        'All primary small-model search passes used one RTX PRO 6000; all primary large-model golden passes used one L40S after preemption. The incomplete large RTX golden pass is retained separately and excluded. Cross-model runtime ratios therefore do not measure model scaling on a common GPU.','',
        '| 8,192-position pass | 34M / RTX PRO 6000 | 129M / L40S |',
        '|---|---:|---:|']
    for key,label in [('shared_fixed_grid_seconds','Shared 64→1,000 grid'),('adaptive','Adaptive budget'),('shallow','Joint 2/4-ply'),('released','Released Allie'),('repaired','Repaired Allie')]:
        out.append(f"| {label} | {reports['small']['timing'][key]:.1f} s | {reports['large']['timing'][key]:.1f} s |")
    out += ['', 'These are measured warm collection times, including root prefills. They exclude model load, analysis and file writes outside the timed region. Fixed-budget curves reuse a continued tree and are not five independent latency runs. The large L40S service cold start took 64.8 s after node-local staging; total allocated time below includes staging and interruptions.','',
        f"On the authoritative L40S order audit (1,024 positions), changing batch order shifted macro CE by {audit['order2_minus_order1_macro']:+.6f} and expert CE by {audit['expert']:+.6f}; both paired intervals include zero. Root logits were identical. Descendant search branches still amplify BF16 rounding: maximum per-position policy difference was {audit['max_policy_gap']:.4f}, mean policy KL {audit['mean_policy_kl']:.2g}. This diagnostic is smaller than the main search gains but is not a proof of exact batch invariance or an extra variance estimate. See [order-large-general/results.json](order-large-general/results.json) and [parity-large-general/results.json](parity-large-general/results.json).",'',
        f"This finite transfer stage consumed **{job['gpu_hours']:.3f} allocated GPU-hours**, across three incarnations of one job, preserving the prior {job['prior_cumulative_gpu_hours']:.3f} GPU-hours. Cumulative search usage: {state['gpu_hours']:.3f} GPU-hours. The job is stopped and its pending automatic requeue was cancelled after every requested evaluation completed. [Accounting](accounting.json).",'']
    return out


def clock_check(size):
    rows,features,_=inventory('aug');n=len(rows);scores=[];base=[];cells=np.array([r['cell'] for r in rows]);check=np.array([r['fold']==1 for r in rows]);p=freeze()['old_parameters']['unchanged']
    for lo in range(0,n,128):
        losses=[]
        for clock in ('predicted','zero'):
            with np.load(OUT/f'{size}-aug-fixed-{clock}'/f'{lo:06d}.npz') as f:
                nn=len(f['game']);part=rows[lo:lo+nn];np.testing.assert_array_equal(f['game'],[r['game'] for r in part])
                pp=policy(part,f['root'],f['q'][-1],f['ids'],f['mask'],np.array([a[-1,0] for a in features[lo:lo+nn]]),p)
                target=np.array([r['legal'].index(r['target']) for r in part]);losses.append(-np.log(pp[np.arange(nn),target]))
        scores.append(losses[1]-losses[0])
    delta=np.concatenate(scores)[check];cc=cells[check];gg=np.array([r['game'] for r in rows])[check]
    b=bootstrap_deltas(delta[:,None],cc,gg)[:,:,0];m=cellmean(delta,cc)
    return dict(zero_minus_predicted_macro=float(m.mean()),expert=float(m[3::4].mean()),macro_ci95=np.quantile(b.mean(1),[.025,.975]).tolist(),expert_ci95=np.quantile(b[:,3::4].mean(1),[.025,.975]).tolist(),
        population='2048 disjoint August confirmation moves; no clock-rule selection; predicted remains primary',
        hardware_confound=None if size=='small' else 'Predicted pass used RTX_PRO6000; zero-elapsed pass used L40S after preemption. This is a combined clock+hardware sensitivity, not an isolated clock-effect estimate.')


def comparisons(reports):
    loaded={}
    for size in reports:
        with np.load(OUT/f'scores-{size}.npz') as f:loaded[size]={k:f[k] for k in f.files}
    a,b=loaded['small'],loaded['large'];np.testing.assert_array_equal(a['games'],b['games']);np.testing.assert_array_equal(a['cells'],b['cells'])
    ds=[];keys=['frozen_256','adaptive_frozen','frozen_1000','frozen_projected']
    for key in keys:
        loss=[]
        for f in (a,b):
            names=list(f['names']);loss.append(f['scores'][names.index(key),:,0]-f['scores'][names.index('legal'),:,0])
        ds.append(loss[1]-loss[0])
    delta=np.stack(ds,1);boot=bootstrap_deltas(delta,a['cells'],a['games']);results={}
    for i,k in enumerate(keys):
        m=cellmean(delta[:,i],a['cells']);r={}
        for metric,ix in [('macro',np.arange(16)),('expert_macro',np.arange(3,16,4))]:
            r[metric]=dict(large_minus_small_search_delta=float(m[ix].mean()),ci95=np.quantile(boot[:,ix,i].mean(1),[.025,.975]).tolist())
        results[k]=r
    # Same game-bootstrap weights for both checkpoints: CM-trend uncertainty is paired.
    cm_keys=['legal',*keys];columns=[]
    for f in (a,b):
        names=list(f['names'])
        columns.extend(f['scores'][names.index(k),:,0]-f['canonical_nll'] for k in cm_keys)
    db=bootstrap_deltas(np.stack(columns,1),a['cells'],a['games']);laws=json.loads((ROOT/'training-cm-laws.json').read_text())
    for i,k in enumerate(keys,1):
        for metric,ix in [('macro',np.arange(16)),('expert_macro',np.arange(3,16,4))]:
            law=laws['metrics'][metric]['law'];gamma=law['alpha']*law['beta']/(law['alpha']+law['beta']);cms=[]
            for si,size in enumerate(['small','large']):
                r=reports[size];residual=excess(law,r['law_coordinate_rung']);change=db[:,ix,si*len(cm_keys)+i].mean(1)
                assert np.all(residual+change>0)
                cms.append((residual/(residual+change))**(1/gamma))
            point=reports['large']['methods'][k][metric+'_training_eq_cm']/reports['small']['methods'][k][metric+'_training_eq_cm']
            results[k][metric]['large_over_small_cm']=point
            results[k][metric]['cm_ratio_ci95']=np.quantile(cms[1]/cms[0],[.025,.975]).tolist()
    atomic(OUT/'scale-differences.json',dict(interpretation='Positive means the search CE reduction is smaller on the larger checkpoint. Paired same golden positions, whole-game bootstrap.',methods=results))
    dominance={}
    for size,f in loaded.items():
        names=list(f['names']);selected=[k for k in SHOW if k in names];pairs=[];deltas=[]
        rr=reports[size]['methods']
        for x in selected:
            for y in selected:
                if x==y or rr[x]['mean_nodes']>rr[y]['mean_nodes']:continue
                if rr[x]['macro']>=rr[y]['macro'] or rr[x]['expert_macro']>=rr[y]['expert_macro']:continue
                pairs.append((x,y));deltas.append(f['scores'][names.index(x),:,0]-f['scores'][names.index(y),:,0])
        draws=bootstrap_deltas(np.stack(deltas,1),f['cells'],f['games']);out=[]
        for j,(x,y) in enumerate(pairs):
            macro=np.quantile(draws[:,:,j].mean(1),[.025,.975]);expert=np.quantile(draws[:,3::4,j].mean(1),[.025,.975])
            if macro[1]<0 and expert[1]<0:out.append(dict(better=x,worse=y,macro_ci95=macro.tolist(),expert_ci95=expert.tolist()))
        dominance[size]=dict(unadjusted_paired_95_both_metrics=out,
            caveat='Exploratory pairwise intervals, not familywise significance; sample-average node costs. Point frontiers alone do not establish dominance.')
    atomic(OUT/'dominance.json',dominance)
    return results,dominance


def plots(reports):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={'coverage':'#1971c2','adaptive':'#e67700','ply':'#6741d9','allie':'#868e96','cheap':'#212529','project':'#2b8a3e'}
    for row,(size,title) in enumerate([('small','34M · final recipe'),('large','129M · final recipe')]):
        ms=reports[size]['methods']
        for col,metric in enumerate(['macro','expert_macro']):
            ax=axes[row,col];keys=[f'frozen_{b}' for b in BUDGETS]
            ax.plot([ms[k]['mean_nodes'] for k in keys],[ms[k][metric] for k in keys],'-o',color=colors['coverage'],label='Frozen coverage search',lw=2,ms=5)
            refits=[f'refit_{b}' for b in BUDGETS]
            ax.plot([ms[k]['mean_nodes'] for k in refits],[ms[k][metric] for k in refits],'--',color='#66a80f',label='Simple calibration refit',lw=1.4)
            for k in keys:
                r=ms[k];lo,hi=r[metric+'_ci95'];ax.errorbar(r['mean_nodes'],r[metric],yerr=[[r[metric]-lo],[hi-r[metric]]],fmt='none',ecolor=colors['coverage'],alpha=.35,capsize=2)
            for k,marker,color,label in [('legal','s','cheap','Legal policy'),('calibrated_direct','x','cheap','Calibrated direct'),('adaptive_frozen','*','adaptive','Adaptive budget'),('frozen_projected','D','project','+ critic consistency'),('two_ply','^','ply','2-ply'),('four_ply','v','ply','4-ply'),('allie_released','P','allie','Allie adaptive'),('allie_repaired','X','allie','Allie repaired')]:
                if k not in ms:continue
                r=ms[k];ax.scatter(r['mean_nodes'],r[metric],marker=marker,c=colors[color],s=90 if marker=='*' else 43,label=label,zorder=4)
            chosen=[k for k in SHOW if k in ms]+refits;front=[];best=float('inf')
            for k in sorted(chosen,key=lambda k:(ms[k]['mean_nodes'],ms[k][metric])):
                if ms[k][metric]<best:front.append(k);best=ms[k][metric]
            ax.plot([ms[k]['mean_nodes'] for k in front],[ms[k][metric] for k in front],':',c='#adb5bd',lw=1,zorder=0)
            ax.set_xscale('symlog',linthresh=20);ax.set_xticks([0,50,128,256,512,1000]);ax.set_xticklabels(['0','50','128','256','512','1000'])
            ax.set_title(title+' · '+('all 16 cells' if col==0 else 'expert cells'));ax.set_ylabel('Move CE ↓');ax.grid(alpha=.15)
            if row==1:ax.set_xlabel('Average new NN evaluations per golden position')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    fig.suptitle('Search quality and cost on the final recipe',fontsize=15)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('pareto.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    old=json.loads((ROOT/'golden-bellman-projection-v1/results.json').read_text())['methods']['unchanged']
    for ax,metric,title in zip(axs,['macro','expert_macro'],['Macro','Expert macro']):
        items=[old,reports['small']['methods']['frozen_1000'],reports['large']['methods']['frozen_1000']]
        values=[a[metric+'_training_eq_cm'] for a in items];intervals=np.array([a[metric+'_cm_ci95'] for a in items])
        ax.bar(['Original 8×512','Final recipe 8×512','Final recipe 16×768'],values,color=['#adb5bd','#74c0fc','#1971c2'],width=.65)
        ax.errorbar(np.arange(3),values,yerr=np.array([np.array(values)-intervals[:,0],intervals[:,1]-np.array(values)]),fmt='none',ecolor='#495057',capsize=4)
        for i,x in enumerate(values):ax.text(i,intervals[i,1]+.08,f'{x:.2f}×',ha='center')
        ax.axhline(1,c='#495057',lw=1);ax.set_ylim(0,float(intervals[:,1].max())*1.18);ax.set_title(title);ax.set_ylabel('Training-equivalent CM vs raw model');ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Same frozen 1,000-budget search; CM is conditional on the transferred law',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('transfer-cm.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def flop_plot(reports):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    costs={s:compute_cost(s) for s in reports}
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for size,color,title in [('small','#6741d9','34M'),('large','#1971c2','129M')]:
        ms=reports[size]['methods'];cost=costs[size]['methods'];keys=['legal']+[f'frozen_{b}' for b in BUDGETS]
        def x(k):
            a=cost[k];return (a['total_flops'] if a['total_flops'] is not None else np.mean(a['total_flops_bounds']))/1e9
        for ax,metric in zip(axs,['macro','expert_macro']):
            ax.plot([x(k) for k in keys],[ms[k][metric] for k in keys],'-o',color=color,label=title+' legal → coverage')
            for key,marker,label in [('adaptive_frozen','*','adaptive'),('four_ply','v','4-ply'),('allie_released','P','Allie')]:
                ax.scatter(x(key),ms[key][metric],marker=marker,color=color,s=90 if marker=='*' else 45,label=title+' '+label)
            ax.set_xscale('log');ax.set_xlabel('Analytical inference GFLOPs / position, including prefill');ax.set_ylabel('Move CE ↓');ax.grid(alpha=.15)
    axs[0].set_title('All 16 cells');axs[1].set_title('Expert cells');fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Across models, a node is not a common unit of compute',fontsize=14)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('inference-flops.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    reports={s:json.loads((OUT/f'results-{s}.json').read_text()) for s in ('small','large')}
    for s,r in reports.items():assert all(k in r['methods'] for k in SHOW),(s,'incomplete comparisons')
    scale,dominance=comparisons(reports);clock={s:clock_check(s) for s in reports};atomic(OUT/'clock-sensitivity.json',clock)
    plots(reports);flop_plot(reports);point_frontiers(reports)
    sm=reports['small']['methods']['frozen_1000'];lg=reports['large']['methods']['frozen_1000']
    text=['# Search transfer and node–quality frontier','',
        '**The frozen search gain survives at 129M, with smaller CE reductions.** At 1,000 nominal simulations, the reduction versus each checkpoint’s legal policy falls from '+f"{-sm['macro_delta_vs_legal']:.4f} to {-lg['macro_delta_vs_legal']:.4f} macro and {-sm['expert_macro_delta_vs_legal']:.4f} to {-lg['expert_macro_delta_vs_legal']:.4f} expert. "+f"All-in conditional CM changes from {sm['macro_training_eq_cm']:.2f}× / {sm['expert_macro_training_eq_cm']:.2f}× to {lg['macro_training_eq_cm']:.2f}× / {lg['expert_macro_training_eq_cm']:.2f}× (macro / expert). Expert CM need not fall with the CE gain because its assumed training curve gets flatter.",'',
        'Frozen methods were tested on two completed checkpoints with the final data/model recipe: 34.2M parameters (8×512) and 128.8M (16×768). Both add a board CNN, SwiGLU, clock features and no key offset. The original exploration checkpoint lacked those changes. No model weights were trained or changed.','',
        '![Node–quality comparison](pareto.png)','']
    ix=text.index('![Node–quality comparison](pareto.png)');text[ix:ix]=takeaways(reports)
    for size,title in [('small','34M checkpoint'),('large','129M checkpoint')]:
        r=reports[size];text += ['## '+title,'',f"Full canonical raw CE: **{r['canonical']['macro']:.5f} / {r['canonical']['expert_macro']:.5f}** (macro / expert). Useful training FLOPs: {r['training_useful_flops']:.4g}.",'',
            '| Method | Avg. NN nodes | Macro CE | Expert CE | Macro CM* | Expert CM* |','|---|---:|---:|---:|---:|---:|']
        for k in SHOW:
            a=r['methods'][k];text.append(f"| {LABEL[k]} | {a['mean_nodes']:.1f} | {a['macro']:.5f} | {a['expert_macro']:.5f} | {a['macro_training_eq_cm']:.2f}× | {a['expert_macro_training_eq_cm']:.2f}× |")
        text += ['', '*CM here includes the served-policy and legality changes relative to canonical raw. Search-only CM relative to legal and calibrated-direct policies, per-cell scores, and paired 95% intervals are in '+f'[results-{size}.json](results-{size}.json).','']
    text += ['## Does the gain shrink with pretraining?','','![Scale comparison](transfer-cm.png)','',
        'The original → new small comparison changes the recipe. New small → new large compares pretraining scale within the same recipe: both model size and tokens increase. This is not a size-only intervention. The [training comparison](training-comparability.json) records the shared data/architecture and small schedule/pool-size differences. The same frozen search/calibration parameters are used; separately reported simple refits use eight output scalars and no search or neural-weight changes.','',
        '| Same frozen method | Large − small macro search delta | Paired 95% CI | Large − small expert search delta | Paired 95% CI |','|---|---:|---|---:|---|']
    for k,v in scale.items():
        a,b=v['macro'],v['expert_macro'];text.append(f"| {LABEL[k]} | {a['large_minus_small_search_delta']:+.5f} | {a['ci95'][0]:+.5f}, {a['ci95'][1]:+.5f} | {b['large_minus_small_search_delta']:+.5f} | {b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f} |")
    cm=scale['frozen_1000']
    text += ['', 'For frozen1,000 search, large/small CM ratios (paired evaluation-only uncertainty): '+ '; '.join(f"{m}: {cm[m]['large_over_small_cm']:.2f}, 95% CI {cm[m]['cm_ratio_ci95'][0]:.2f}–{cm[m]['cm_ratio_ci95'][1]:.2f}" for m in ['macro','expert_macro'])+'.','']
    text += ['', 'Positive values mean a smaller CE reduction at the larger model. This is a paired comparison against each model’s own legal prior, on identical golden positions. Two scales do not identify a scaling exponent. The scaled checkpoint is 129M, not a completed 4B production run.','',
        'The same CE improvement maps to a larger CM on a flatter curve. Local dL/dlnC at the primary anchors (macro / expert): '+ '; '.join(f"{s}: {r['methods']['legal']['macro_local_dL_dlnC']:.5f} / {r['methods']['legal']['expert_macro_local_dL_dlnC']:.5f}" for s,r in reports.items())+'.','',
        '## Cost across model sizes','','![Inference-compute comparison](inference-flops.png)','',
        'The node frontiers are separate for each checkpoint. Across sizes, use inference arithmetic or actual runtime: a larger-model node is more expensive. This second plot charges attention/MLP/board/clock/head matmuls and root prefill. Intermediate-budget attention counts have depth bounds (plot uses their midpoint). It excludes nonlinearities, memory traffic, padding and CPU search, so these are analytical useful FLOPs, not executed GPU instructions. See cost-small.json and cost-large.json for formulas and bounds.','',
        '**The larger direct model is the better use of inference arithmetic here:** its legal policy has CE 1.35437 / 1.24028 at 12.23 GFLOPs per sampled position, versus the small model’s frozen 1,000-budget search at 1.41895 / 1.28654 and 58.22 GFLOPs. This comparison charges a fresh root prefill per query; it does not measure CPU latency or amortized serving across a whole game. Training and model-memory costs are separate.','',
        '## What the algorithms do','',
        '- **Coverage search:** allocate root simulations in proportion to a diminishing-return score based on √(p(1−p)); use PUCT within each root-action branch. Back up values with a prior-weighted soft maximum whose temperature falls with subtree size. A frozen calibrated mixture turns action values and the human prior into a full-support move distribution.',
        '- **Adaptive budget:** the old Elo-conditioned router chooses 128/256/512/1,000 simulations, executed live with the same tree and cache. It does not look at the played move.',
        '- **Critic consistency:** a bottom-up consistency correction on the same tree before value backup. It adds CPU work but no NN nodes. Small point improvements are not automatically established dominance.',
        '- **2/4-ply:** evaluate every legal root move; average truncated human-policy continuations with critic fallback for unexpanded mass. Widths are 4,2,2 after the root.',
        '- **Allie:** the released adaptive allocation uses predicted thinking time and its reverse-KL output solve. The repaired 50-simulation reference fixes first-prior selection and depth truncation, with the previously frozen calibrated reverse-KL output. These specific references do not exhaust every possible adaptive MCTS.',
        '- **Calibration-only refit:** per-Elo α and β in softmax(α·logit+β·Q), fit on August fold0 only. This is simpler than the original conditional-mixture head, so its failure is not evidence that every recalibration would fail.','',
        '## Evaluation and limits','',
        'The original golden inventory and 16 equally weighted format×Elo cells are unchanged. Expert macro averages the four ≥2400 cells. All 8,192 sampled moves (512/cell) stay in every method; no failed/forced positions are removed. CE uses each model’s full canonical cell means plus the paired sample difference, not full-population search scoring. Whole games are resampled for confidence intervals. This previously reused golden sample is a transfer check, not a fresh independent final confirmation; no methods were selected on its transfer losses. August fit/check games are disjoint but potentially seen during model training. Test/test_expert remain unopened. One checkpoint per size is tested; intervals exclude training-seed variance.','',
        'CM is a **law-shape-conditional training-equivalent estimate**, not measured extra training and not an inference speedup. The old golden compute-optimal law is vertically shifted to each checkpoint at its study rung (3e16 / 3e17), matching source fits and training ledgers. The initial provisional update used useful training FLOPs / 6; those coordinates remain a sensitivity field in each results JSON, with the correction recorded in cm-convention-addendum.json. We have not established that the improved recipe shares the old law’s local slope. Bootstrap intervals exclude law-fit uncertainty. The CE reductions are the directly observed result.','',
        'The dotted lines are point-estimate frontiers. The separate macro, expert and joint non-dominated sets are in [frontiers.json](frontiers.json). Paired comparisons that favor a method on both metrics at no greater sample-average node count are in [dominance.json](dominance.json); those intervals are exploratory and not familywise adjusted. Do not infer significance from a line crossing alone.','',
        'A discrepancy remains: on the potentially training-seen August confirmation subset, large-model frozen 1,000 search was essentially null/slightly worse (1.3997 / 1.1199 versus legal 1.3966 / 1.1174), while July golden improves. We report both populations rather than selecting between them; their difference does not identify a cause. The original 10×/10× target remains unmet.','',
        '## Serving correctness, clocks and cost','',
        'The portable FP32 model matches each frozen source exactly in independent full-math checks. Full canonical GPU scoring reproduces the official reports within ordinary numerical variation. Dense versus cached BF16 comparisons and their value discrepancies are recorded in parity-small/results.json and parity-large/results.json. Each query owns its prefix, board state, clocks and KV cache; there is no cross-position tree/value memory.','',
        'Search sees actual pre-move clock features at the root. Hypothetical children use the model’s expected think time, rounded and capped by time remaining, then apply increment; the third feature is the next mover’s previous OWN think time. Opening and missing-clock conventions follow the sampler. Future observed clocks/outcomes are never used. The development-only zero-elapsed sensitivity is reported in [clock-sensitivity.json](clock-sensitivity.json); the predicted-time rule remains primary.','',
        '**Imagined clock handling remains a weakness.** On small-model August confirmation, assuming zero elapsed time improved macro CE by 0.00596 (paired 95% CI 0.00132–0.01051 better); the expert difference was unresolved. This shows that using the predicted mean time is not an established best rule. It is a development diagnostic, not a new golden-selected method. The large sensitivity crosses GPU types after preemption and is confounded, so it cannot isolate the clock effect.','',
        'A node means one actual new non-root NN evaluation. Rules-only terminal backups and cache hits do not count. All methods also pay root prefills. The five fixed-budget points share a continued tree, so their cumulative collection time is not a standalone latency benchmark for each point. See results JSON timing fields for real shared-grid, adaptive and baseline times; the same checkpoint/GPU is used for all its methods. GPU throughput does not establish CPU-hosting speed.','',
        'All new code and results are private to the search worktree. Claude’s checkpoints, sources, stores and jobs were read-only. See [plan.json](plan.json), [scale-differences.json](scale-differences.json), the per-model result JSONs, and source search/transfer for reproducibility. Original-stage history is preserved in [STAGE_REPORT.md](../STAGE_REPORT.md).','']
    text += execution_notes(reports)
    (OUT/'REPORT.md').write_text('\n'.join(text))
    print('WROTE',OUT/'REPORT.md',flush=True)


if __name__=='__main__':main()
