from pathlib import Path
import csv,statistics,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
p=Path(__file__).resolve().parent.parent
from data import load_rows
rows=load_rows()
for r in rows:
 for k in ['bits','rep','threads','overquery','qps','recall','visited','build_s','train_s','encode_s','provider_s','nvq_s']:
  if k in r:r[k]=float(r[k]) if r[k] else 0.
datasets=[('ada002-1M','Ada002-1M'),('cap-1M','CAP-1M'),('cohere-english-v3-1M','Cohere-1M')]
colors={'ASH':'#145ee8','PQ':'#d62929'}
preliminary = len(rows) < 96
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
def selected(ds,b,m,t,o):return [r for r in rows if r['dataset']==ds and r['bits']==b and r['mode']==m and r['threads']==t and r['overquery']==o]
def avg(rr,key):return statistics.mean(r[key] for r in rr)
fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
for ax,(ds,title) in zip(axs,datasets):
 for b in (2,4):
  for mode in ('ASH','PQ'):
   rr=[selected(ds,b,mode,1,o) for o in (1,2)]
   if not all(rr):continue
   xx=[avg(x,'recall')*100 for x in rr];yy=[avg(x,'qps') for x in rr]
   ax.plot(xx,yy,color=colors[mode],ls='--' if b==2 else '-',marker='x',ms=7,lw=2)
   for o,x,y in zip((1,2),xx,yy):ax.annotate(f'{o}×',(x,y),xytext=(5,5 if mode=='ASH' else -12),textcoords='offset points',fontsize=9,color=colors[mode])
 ax.margins(x=.08,y=.12);ax.set_title(title);ax.set_xlabel('Recall@10 (%)');ax.set_ylabel('Single-thread QPS');ax.grid(alpha=.2)
fig.suptitle('Production graph: ASH vs latest-main PQ · NVQ reranking' + (' (preliminary)' if preliminary else ''))
fig.legend(handles=[Line2D([],[],color=colors[m],ls=ls,lw=2,label=f'{m} · {label}') for b,ls,label in [(2,'--','32× nominal'),(4,'-','16× nominal')] for m in ['ASH','PQ']],loc='outside lower center',ncol=4,handlelength=3.5)
fig.savefig(p/'graph-qps-recall.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
for ax,(ds,title) in zip(axs,datasets):
 for x,b in enumerate((2,4)):
  for shift,m in [(-.18,'ASH'),(.18,'PQ')]:
   rr=selected(ds,b,m,1,1)
   if not rr:continue
   stages=[avg(rr,'train_s'),avg(rr,'encode_s')+avg(rr,'provider_s')+avg(rr,'nvq_s'),avg(rr,'build_s')]
   bottom=0
   for val,hatch in zip(stages,['///','xxx','']):
    ax.bar(x+shift,val,.32,bottom=bottom,color=colors[m],hatch=hatch,edgecolor='white',linewidth=.8)
    bottom+=val
   totals=[sum(z[k] for k in ['train_s','encode_s','provider_s','nvq_s','build_s']) for z in rr]
   if len(totals)>1: ax.errorbar(x+shift,bottom,yerr=[[bottom-min(totals)],[max(totals)-bottom]],fmt='none',ecolor='black',capsize=4,lw=1)
   ax.text(x+shift,max(totals)+2,f'{bottom:.0f}s',ha='center',fontsize=9)
 ax.set_xticks([0,1],['32× nominal\n2-bit ASH','16× nominal\n4-bit ASH']);ax.set_title(title);ax.set_ylabel('Indexing phase seconds');ax.grid(axis='y',alpha=.2)
fig.suptitle('Fresh production index construction · ' + ('preliminary observations' if preliminary else 'mean of two builds; whiskers show range'))
fig.legend(handles=[Patch(facecolor=colors[m],label=m) for m in ['ASH','PQ']]+[Patch(facecolor='gray',edgecolor='white',hatch=h,label=t) for h,t in [('///','Training'),('xxx','Encoding + setup'),('','Graph build/write')]],loc='outside lower center',ncol=5)
fig.savefig(p/'graph-construction.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
for ax,(ds,title) in zip(axs,datasets):
 for b in (2,4):
  for m in ['ASH','PQ']:
   for o,mark in [(1,'o'),(2,'x')]:
    rr=[selected(ds,b,m,t,o) for t in (1,48)]
    if not all(rr):continue
    yy=[avg(x,'qps') for x in rr]
    ax.plot([1,48],yy,color=colors[m],ls='--' if b==2 else '-',marker=mark,lw=1.5)
 ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks([1,48],['1','48']);ax.set_title(title);ax.set_xlabel('Query workers');ax.set_ylabel('QPS');ax.grid(alpha=.2)
fig.suptitle('Same cached graphs: single-thread and concurrent query throughput')
fig.legend(handles=[Line2D([],[],color=colors[m],label=m) for m in ['ASH','PQ']]+[Line2D([],[],color='black',ls=ls,label=label) for ls,label in [('--','32× nominal'),('-','16× nominal')]]+[Line2D([],[],color='black',ls='',marker=m,label=f'{o}× overquery') for m,o in [('o',1),('x',2)]],loc='outside lower center',ncol=6,handlelength=3)
fig.savefig(p/'graph-concurrency.png');plt.close(fig)
# Published raster panels are shown unchanged; they are not new measurements.
fig,axs=plt.subplots(1,3,figsize=(15,5.2),layout='constrained')
for i,ax in enumerate(axs):
 ax.imshow(plt.imread(p/f'paper-panel-{i}.png'));ax.axis('off')
fig.suptitle('Published IVF results — Figure 9 (unchanged; original paper colors)')
fig.savefig(p/'published-ivf-reference.png');plt.close(fig)
print('Plots written')
