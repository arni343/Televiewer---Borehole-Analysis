"""Build a local, interactive review dashboard from the saved candidate CSVs.
Run after televiewer.py. No network service or extra dependencies required.
Reviewer decisions are downloaded as CSV; they do not establish missed-feature truth.
"""
import argparse
import csv
import html
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",default=str(Path(__file__).with_name("results")))
    args=parser.parse_args()
    root=Path(args.results)
    with (root/"all_candidates.csv").open(encoding="utf-8-sig",newline="") as f:
        data=list(csv.DictReader(f))
    rows=[]
    for i,d in enumerate(data):
        score=float(d["quality_score"])
        values=[d["borehole"],d["channel"],d["candidate_id"],
                f'{float(d["center_row_px"]):.1f}',
                f'{float(d["depth_m"]):.3f}' if d.get("depth_m") else "",
                f'{float(d["amplitude_px"]):.1f}',
                d.get("borehole_relative_dip_deg") or "Unavailable",
                "Annotated" if d.get("annotated_input")=="True" else "Unpicked report"]
        cells="".join("<td>"+html.escape(v)+"</td>" for v in values)
        rows.append(f'<tr data-score="{score}" data-key="{i}">{cells}'
                    f'<td><meter min="0" max="1" value="{score}"></meter> {score:.2f}</td>'
                    f'<td>{html.escape(d.get("review_flag",""))}</td>'
                    f'<td><select aria-label="Review decision for candidate {i}">'
                    '<option value="">Unreviewed</option><option>accept</option>'
                    '<option>reject</option><option>uncertain</option></select></td></tr>')
    galleries=[]
    for folder in sorted(root.glob("BH*")):
        if not folder.is_dir():
            continue
        pages=sorted(folder.glob("page_*.png"))
        items="".join(f'<p>{p.name}</p><img loading="lazy" alt="{html.escape(folder.name)} {p.name}" '
                      f'src="{p.relative_to(root).as_posix()}">' for p in pages)
        galleries.append(f'<details><summary>{html.escape(folder.name)} ({len(pages)} pages)</summary>'
                         +items+"</details>")
    template = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>RGL candidate review</title><style>
body{font:15px system-ui,sans-serif;margin:24px;color:#152334;background:#f8fafc}
h1{font-size:26px}p{max-width:1000px;line-height:1.5}.controls{display:flex;gap:24px;align-items:center;flex-wrap:wrap}
input,select,button{padding:7px}table{border-collapse:collapse;background:white;width:100%;font-size:13px}
td,th{border:1px solid #d8e0e8;padding:7px;text-align:left}thead{position:sticky;top:0;background:#e7eef7}
.table-wrap{max-height:550px;overflow:auto;margin-top:15px}meter{width:85px}
details{margin:15px 0;padding:15px;background:white;border:1px solid #d8e0e8}img{max-width:100%;height:auto}
summary,button{cursor:pointer}.notice{padding:14px;border-left:4px solid #c48c21;background:#fff6df}
</style><h1>RGL televiewer candidate review</h1>
<p class="notice">These are unclassified planar candidates. Fit quality ranks image support and is
<strong>not a probability</strong> of a correct geological interpretation. Accepting candidates here does not
label missed features. Independent accuracy requires fully reviewed intervals and reference traces.
Annotated acoustic images are exploratory only; empty dip values mean missing calibration.</p>
<p>Use the page images below to review candidates by ID. Choose accept, reject or uncertain, then download
the decisions CSV. Decisions remain in this tab until downloaded; nothing is sent to a server.</p>
<div class="controls"><label>Find borehole/channel <input id="search" type="search" placeholder="e.g. BH03 optical"></label>
<label>Minimum fit quality <input id="threshold" type="range" min="0" max="1" value="0" step=".05">
<output id="value">0.00</output></label><button id="download">Download reviewed decisions</button>
<span id="count"></span></div>
<div class="table-wrap"><table><thead><tr><th>Borehole</th><th>Track</th><th>ID</th>
<th>Report row</th><th>Depth m*</th><th>Amplitude px</th><th>Relative dip deg</th><th>Input</th>
<th>Fit quality</th><th>Review flag</th><th>Decision</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p>* Depth uses provisional report-tick calibration. Amplitude is half the peak-to-trough height.
<a href="README_LINK">Analysis overview</a> | <a href="all_candidates.csv">All candidate parameters</a></p>
<h2>Review images</h2>__GALLERIES__
<script>
const records=__DATA__;
const rows=[...document.querySelectorAll('tbody tr')];
const search=document.querySelector('#search'),threshold=document.querySelector('#threshold');
function filter(){
 const terms=search.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);
 let shown=0;
 for(const row of rows){row.hidden=Number(row.dataset.score)<Number(threshold.value)||
 !terms.every(t=>row.textContent.toLowerCase().includes(t));if(!row.hidden)shown++;}
 document.querySelector('#value').textContent=Number(threshold.value).toFixed(2);
 document.querySelector('#count').textContent=shown+' / '+rows.length+' candidates shown';
}
search.addEventListener('input',filter);threshold.addEventListener('input',filter);filter();
document.querySelector('#download').addEventListener('click',()=>{
 const reviewed=rows.filter(r=>r.querySelector('select').value).map(r=>({...records[Number(r.dataset.key)],
 review_decision:r.querySelector('select').value}));
 if(!reviewed.length){alert('Choose at least one review decision first.');return;}
 const keys=[...new Set(reviewed.flatMap(Object.keys))];
 const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
 const lines=[keys.map(quote).join(','),...reviewed.map(r=>keys.map(k=>quote(r[k])).join(','))];
 const url=URL.createObjectURL(new Blob([lines.join('\\r\\n')],{type:'text/csv;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download='review_decisions.csv';a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
});
</script></html>"""
    document=template.replace("__ROWS__","".join(rows)).replace("__GALLERIES__","".join(galleries))
    document=document.replace("__DATA__",json.dumps(data).replace("</","<\\/")).replace("README_LINK","index.html")
    # JS uses actual regex/escape syntax, not doubled Python-literal escapes.
    document=document.replace("/\\\\s+/","/\\s+/").replace("'\\\\r\\\\n'","'\\r\\n'")
    (root/"review.html").write_text(document,encoding="utf-8")
    print(f"Built {root/'review.html'} with {len(data)} candidates.")

if __name__=="__main__":
    main()

