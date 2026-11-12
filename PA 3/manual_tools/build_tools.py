import csv, json, html, sys
from pathlib import Path
ROOT = Path("/home/zain/Desktop/ATML/PA 3")
OUT = Path(sys.argv[1])

CSS = """
:root{--bg:#fff;--fg:#1b1b1b;--mut:#666;--card:#f6f6f7;--line:#d6d6da;--acc:#2a62d9;--warn:#b25b00}
@media (prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e8e8ea;--mut:#9a9aa0;--card:#202226;--line:#34363c;--acc:#7aa2ff;--warn:#f0a050}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:5;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
main{padding:12px 16px;max-width:1500px;margin:auto}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:7px 12px;cursor:pointer;font-size:14px}
.group{margin:18px 0 30px}.prompt{background:var(--card);border:1px solid var(--line);padding:10px 12px;border-radius:8px;white-space:pre-wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;margin-top:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.card h4{margin:0 0 6px}.resp{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;max-height:340px;overflow:auto;font-size:13.5px}
.meta{color:var(--mut);font-size:12.5px;margin:4px 0}.trunc{color:var(--warn);font-weight:600}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:6px;font-size:13.5px}
.row b{min-width:120px}label{white-space:nowrap}input[type=text]{width:100%;box-sizing:border-box;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:5px}
.done{outline:2px solid var(--acc)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pair img{width:100%;border-radius:6px;border:1px solid var(--line)}
.q{font-size:17px;font-weight:600;margin:6px 0}
@media (max-width:800px){.pair{grid-template-columns:1fr}}
"""

# ---------- Task 1 qualitative annotation ----------
rows = list(csv.DictReader(open(ROOT / "results/task1/qualitative/annotation_sheet.csv", encoding="utf-8")))
cols = list(rows[0].keys())
order = ["baseline", "full_ft_mid", "full_ft_final", "full_ft_l2sp", "lora4", "lora16", "oplora16"]
fields = [("format_compliance", ["yes", "partial", "no", "n/a"]),
          ("coherence_relevance", ["good", "partial", "poor"]),
          ("repetition_manual", ["no", "yes"]),
          ("math_framing", ["no", "yes"]),
          ("inappropriate", ["no", "yes"])]
data = {"cols": cols, "rows": rows, "order": order, "fields": fields}
js = r"""
const D=__DATA__;const KEY='pa3_t1_annot_v2';
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
const id=(r)=>r.condition+'|'+r.qindex+'|'+r.fmt;
const groups={};D.rows.forEach(r=>{const g=r.qindex+'|'+r.fmt;(groups[g]=groups[g]||[]).push(r)});
const main=document.getElementById('main');
Object.keys(groups).sort().forEach(g=>{
  const rs=groups[g].sort((a,b)=>D.order.indexOf(a.condition)-D.order.indexOf(b.condition));
  const div=document.createElement('div');div.className='group';
  div.innerHTML='<h3>Question '+rs[0].qindex+' ('+rs[0].fmt+')</h3><div class="prompt"></div><div class="grid"></div>';
  div.querySelector('.prompt').textContent=rs[0].prompt;
  const grid=div.querySelector('.grid');
  rs.forEach(r=>{
    const c=document.createElement('div');c.className='card';c.dataset.id=id(r);
    let h='<h4>'+r.condition+'</h4><div class="meta">'+r.n_tokens+' tokens'+(r.truncated==='True'?' <span class="trunc">TRUNCATED at cap</span>':'')+' &middot; auto-repetition '+r.repetition+'</div><div class="resp"></div>';
    D.fields.forEach(([f,opts])=>{h+='<div class="row"><b>'+f+'</b>'+opts.map(o=>'<label><input type="radio" name="'+id(r)+'|'+f+'" data-f="'+f+'" value="'+o+'"> '+o+'</label>').join('')+'</div>'});
    h+='<div class="row"><input type="text" placeholder="notes: short excerpt, what changed vs baseline" data-note="1"></div>';
    c.innerHTML=h;c.querySelector('.resp').textContent=r.response;
    const st=S[id(r)]||{};
    c.querySelectorAll('input[type=radio]').forEach(inp=>{
      const f=inp.dataset.f;if(st[f]===inp.value)inp.checked=true;
      inp.addEventListener('change',()=>{S[id(r)]=S[id(r)]||{};S[id(r)][f]=inp.value;save();upd()});});
    const n=c.querySelector('[data-note]');n.value=st.notes||'';n.addEventListener('input',()=>{S[id(r)]=S[id(r)]||{};S[id(r)].notes=n.value;save()});
    grid.appendChild(c)});
  main.appendChild(div)});
function upd(){let d=0;D.rows.forEach(r=>{const st=S[id(r)]||{};const ok=D.fields.every(([f])=>st[f]);if(ok)d++;
  const el=document.querySelector('[data-id="'+id(r)+'"]');if(el)el.classList.toggle('done',ok)});
  document.getElementById('prog').textContent=d+' / '+D.rows.length+' fully annotated'}
function exp(){const q=v=>'"'+String(v).replace(/"/g,'""')+'"';
  const lines=[D.cols.join(',')];
  D.rows.forEach(r=>{const st=S[id(r)]||{};const o=Object.assign({},r);
    D.fields.forEach(([f])=>o[f]=st[f]||'');o.notes=st.notes||'';
    lines.push(D.cols.map(c=>q(o[c]??'')).join(','))});
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));
  a.download='annotation_sheet_filled.csv';a.click()}
document.getElementById('exp').onclick=exp;upd();
"""
page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task 1 annotation</title><style>{CSS}</style></head><body>
<header><b>Task 1 qualitative annotation</b><span id="prog"></span><button id="exp">Export filled CSV</button><span class="meta">Progress is saved in this browser. Export at any time.</span></header>
<main id="main"></main><script>{js.replace('__DATA__', json.dumps(data))}</script></body></html>"""
(OUT / "annotate_task1.html").write_text(page, encoding="utf-8")

# ---------- Task 2 grounding label audit ----------
au = list(csv.DictReader(open(ROOT / "splits/task2/grounding_audit.csv", encoding="utf-8")))
def prio(r): return {"gtest": 0, "gdev": 1, "gtrain": 2}[r["pair_id"].split("_")[0]]
au.sort(key=lambda r: (prio(r), r["pair_id"]))
for r in au:
    for k in ("a", "b"):
        r["url_" + k] = f"file://{ROOT}/data/coco/{r['split']}/COCO_{r['split']}_{int(r['image_' + k]):012d}.jpg"
adata = {"rows": au, "cols": list(csv.DictReader(open(ROOT / "splits/task2/grounding_audit.csv")).__next__().keys())}
ajs = r"""
const D=__DATA__;const KEY='pa3_t2_audit_v1';let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
const main=document.getElementById('main');const OPTS=[['ok','both labels correct'],['bad_a','image A label wrong'],['bad_b','image B label wrong'],['both_bad','both wrong'],['ambiguous','ambiguous / cannot tell']];
D.rows.forEach((r,i)=>{
  const c=document.createElement('div');c.className='card';c.dataset.id=r.pair_id;c.style.marginBottom='14px';
  const pres=r.pair_id.includes('presence');
  c.innerHTML='<div class="meta">#'+(i+1)+' &middot; '+r.pair_id+' &middot; '+r.split+'</div><div class="q"></div>'+
   '<div class="pair"><div><div class="meta"><b>Image A</b> &mdash; label: <b class="la"></b></div><img loading="lazy"></div><div><div class="meta"><b>Image B</b> &mdash; label: <b class="lb"></b></div><img loading="lazy"></div></div>'+
   '<div class="row">'+OPTS.map(o=>'<label><input type="radio" name="'+r.pair_id+'" value="'+o[0]+'"> '+o[1]+'</label>').join('')+'</div>'+
   '<div class="row"><input type="text" placeholder="notes (optional)" data-note="1"></div>';
  c.querySelector('.q').textContent=r.question;c.querySelector('.la').textContent=r.answer_a;c.querySelector('.lb').textContent=r.answer_b;
  const im=c.querySelectorAll('img');im[0].src=r.url_a;im[1].src=r.url_b;
  const st=S[r.pair_id]||{};
  c.querySelectorAll('input[type=radio]').forEach(inp=>{if(st.v===inp.value)inp.checked=true;
    inp.addEventListener('change',()=>{S[r.pair_id]=S[r.pair_id]||{};S[r.pair_id].v=inp.value;save();upd()})});
  const n=c.querySelector('[data-note]');n.value=st.notes||'';n.addEventListener('input',()=>{S[r.pair_id]=S[r.pair_id]||{};S[r.pair_id].notes=n.value;save()});
  main.appendChild(c)});
function upd(){let d=0;D.rows.forEach(r=>{const ok=!!(S[r.pair_id]||{}).v;if(ok)d++;document.querySelector('[data-id="'+r.pair_id+'"]').classList.toggle('done',ok)});
  document.getElementById('prog').textContent=d+' / '+D.rows.length+' audited';}
function exp(){const q=v=>'"'+String(v).replace(/"/g,'""')+'"';const cols=D.cols.concat(['notes']);const lines=[cols.join(',')];
  D.rows.forEach(r=>{const st=S[r.pair_id]||{};const o=Object.assign({},r);o['label_ok(manual)']=st.v||'';o.notes=st.notes||'';lines.push(cols.map(c=>q(o[c]??'')).join(','))});
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));a.download='grounding_audit_filled.csv';a.click()}
document.getElementById('exp').onclick=exp;upd();
"""
apage = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task 2 label audit</title><style>{CSS}</style></head><body>
<header><b>Task 2 grounding label audit</b><span id="prog"></span><button id="exp">Export filled CSV</button><span class="meta">Order: final-test pairs, then dev, then a train sample. Saved in this browser.</span></header>
<main id="main"></main><script>{ajs.replace('__DATA__', json.dumps(adata))}</script></body></html>"""
(OUT / "audit_task2.html").write_text(apage, encoding="utf-8")
print("built", len(rows), "responses;", len(au), "pairs")
