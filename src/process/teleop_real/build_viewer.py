"""
Stage 3: build a single HTML dataset viewer from the _index.json files.

Shows, for both folders (test + test_object):
  - summary: total sessions, valid count, prompt count, total data size
  - per-session cards: FIRST|LAST composite (26053260), valid badge, prompt text,
    moved object, focus-pair frame counts (260 vs 465), size
  - filters: all / valid / invalid / has-prompt / no-prompt, and a text search

Output: <local_shared_dir>/capture/hri_vive/dataset_viewer.html
Open it in a browser (image paths are relative to that folder).
"""
import os
import json
import html

from paradex.utils.path import local_shared_dir

ROOT = local_shared_dir
HRI = os.path.join(ROOT, "capture", "hri_vive")
FOLDERS = ["capture/hri_vive/test", "capture/hri_vive/test_object"]


def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def load_index(folder):
    p = os.path.join(ROOT, folder, "_index.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    indices = [(f, load_index(f)) for f in FOLDERS]
    indices = [(f, idx) for f, idx in indices if idx]

    # image paths in the HTML are relative to HRI (where the html is written):
    #   test/<session>/_frames/26053260_firstlast.jpg
    data = {}
    for folder, idx in indices:
        leaf = folder.split("/")[-1]     # "test" | "test_object"
        rows = []
        for e in idx["entries"]:
            comp = e.get("composite")
            img = f"{leaf}/{comp}" if comp else None   # comp starts with "<session>/_frames/..."
            fc = e.get("frame_counts", {})
            rows.append({
                "s": e["session"], "valid": e["valid"], "prompt": e.get("prompt"),
                "obj": e.get("moved_object"), "conf": e.get("confidence"),
                "reason": e.get("reason", []), "med": e.get("med_frames"),
                "c260": fc.get("26053260"), "c465": fc.get("25305465"),
                "arm": e.get("arm_len"), "bytes": e.get("bytes"), "img": img,
            })
        data[leaf] = {
            "folder": folder, "rows": rows,
            "total": idx["total_sessions"], "valid": idx["valid"],
            "prompts": idx["with_prompt"], "bytes": idx["total_bytes"],
        }

    grand_bytes = sum(d["bytes"] for d in data.values())
    grand_total = sum(d["total"] for d in data.values())
    grand_valid = sum(d["valid"] for d in data.values())
    grand_prompt = sum(d["prompts"] for d in data.values())

    payload = json.dumps(data)
    summary_cards = ""
    for leaf, d in data.items():
        summary_cards += (
            f'<div class="sum"><h3>{html.escape(d["folder"])}</h3>'
            f'<div class="big">{d["total"]}</div><div class="lbl">sessions</div>'
            f'<div class="row"><span>valid</span><b>{d["valid"]}</b></div>'
            f'<div class="row"><span>with prompt</span><b>{d["prompts"]}</b></div>'
            f'<div class="row"><span>size</span><b>{human(d["bytes"])}</b></div></div>')

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Teleop dataset viewer</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
 header{{padding:16px 20px;background:#161a22;position:sticky;top:0;z-index:5;border-bottom:1px solid #262b36}}
 h1{{margin:0 0 10px;font-size:18px}}
 .sums{{display:flex;gap:12px;flex-wrap:wrap}}
 .sum{{background:#1b2130;border:1px solid #2a3140;border-radius:10px;padding:10px 14px;min-width:150px}}
 .sum h3{{margin:0 0 6px;font-size:12px;color:#9fb0c8;font-weight:600}}
 .big{{font-size:26px;font-weight:700}} .lbl{{font-size:11px;color:#8894a8;margin-bottom:6px}}
 .sum .row{{display:flex;justify-content:space-between;font-size:12px;color:#c3ccda}}
 .grand{{background:#123; border-color:#2b6}}
 .controls{{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
 .controls button{{background:#232a38;color:#dbe4f0;border:1px solid #333c4d;border-radius:6px;padding:6px 10px;cursor:pointer;font-size:13px}}
 .controls button.on{{background:#2b6;color:#04210f;border-color:#2b6;font-weight:700}}
 input#q{{background:#0d1017;border:1px solid #333c4d;color:#e6e6e6;border-radius:6px;padding:6px 10px;font-size:13px;min-width:200px}}
 #grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px;padding:16px}}
 .card{{background:#161a22;border:1px solid #262b36;border-radius:10px;overflow:hidden}}
 .card img{{width:100%;display:block;background:#000;cursor:zoom-in}}
 .meta{{padding:8px 10px;font-size:12px}}
 .sname{{font-weight:600;font-size:12px;color:#cdd7e5;word-break:break-all}}
 .badge{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:700;margin-right:6px}}
 .ok{{background:#123d22;color:#57e08a;border:1px solid #2b6}}
 .bad{{background:#3d1414;color:#ff8a8a;border:1px solid #b33}}
 .prompt{{margin-top:6px;color:#ffd479;font-size:12px}}
 .noprompt{{margin-top:6px;color:#7a869b;font-size:12px;font-style:italic}}
 .nums{{margin-top:4px;color:#8894a8;font-size:11px}}
 .reason{{margin-top:4px;color:#ff9d9d;font-size:11px}}
 #modal{{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;align-items:center;justify-content:center;z-index:20}}
 #modal img{{max-width:96vw;max-height:92vh}}
</style></head><body>
<header>
 <h1>Teleop dataset viewer &nbsp;<span style="font-size:12px;color:#8894a8">
   {grand_total} sessions &middot; {grand_valid} valid &middot; {grand_prompt} prompts &middot; {human(grand_bytes)} total</span></h1>
 <div class="sums">{summary_cards}
   <div class="sum grand"><h3>TOTAL</h3><div class="big">{human(grand_bytes)}</div>
   <div class="lbl">all data</div>
   <div class="row"><span>sessions</span><b>{grand_total}</b></div>
   <div class="row"><span>valid</span><b>{grand_valid}</b></div>
   <div class="row"><span>prompts</span><b>{grand_prompt}</b></div></div>
 </div>
 <div class="controls">
   <button data-f="all" class="on">All</button>
   <button data-f="valid">Valid</button>
   <button data-f="invalid">Invalid</button>
   <button data-f="prompt">Has prompt</button>
   <button data-f="noprompt">No prompt</button>
   <span style="width:14px"></span>
   <button data-folder="all" class="on">Both folders</button>
   <button data-folder="test">test</button>
   <button data-folder="test_object">test_object</button>
   <input id="q" placeholder="search session / object..."/>
   <span id="count" style="color:#8894a8;font-size:12px"></span>
 </div>
</header>
<div id="grid"></div>
<div id="modal"><img/></div>
<script>
const DATA = {payload};
let fFilter="all", folderFilter="all", q="";
function human(n){{for(const u of["B","KB","MB","GB","TB"]){{if(n<1024)return n.toFixed(1)+u;n/=1024;}}return n+"P";}}
function rows(){{
  let out=[];
  for(const leaf of Object.keys(DATA)){{
    if(folderFilter!=="all" && folderFilter!==leaf) continue;
    for(const r of DATA[leaf].rows) out.push(Object.assign({{leaf}},r));
  }}
  return out;
}}
function pass(r){{
  if(fFilter==="valid"&&!r.valid) return false;
  if(fFilter==="invalid"&&r.valid) return false;
  if(fFilter==="prompt"&&!r.prompt) return false;
  if(fFilter==="noprompt"&&r.prompt) return false;
  if(q){{const s=(r.s+" "+(r.obj||"")+" "+(r.prompt||"")).toLowerCase();if(!s.includes(q))return false;}}
  return true;
}}
function render(){{
  const g=document.getElementById("grid"); g.innerHTML="";
  const rs=rows().filter(pass);
  document.getElementById("count").textContent=rs.length+" shown";
  for(const r of rs){{
    const d=document.createElement("div"); d.className="card";
    const vb=r.valid?'<span class="badge ok">valid</span>':'<span class="badge bad">invalid</span>';
    const pr=r.prompt?('<div class="prompt">\\u201c'+r.prompt+'\\u201d</div>')
                      :('<div class="noprompt">no prompt'+(r.obj&&r.obj!=="none"?" ("+r.obj+")":"")+'</div>');
    const reason=(!r.valid&&r.reason&&r.reason.length)?('<div class="reason">'+r.reason.join("; ")+'</div>'):"";
    const img=r.img?('<img loading="lazy" src="'+r.img+'" onclick="zoom(this.src)">'):'<div style="height:120px;background:#000"></div>';
    d.innerHTML=img+'<div class="meta"><div class="sname">'+vb+r.leaf+" / "+r.s+'</div>'
      +pr
      +'<div class="nums">260='+r.c260+' 465='+r.c465+' &middot; arm='+r.arm+' &middot; '+human(r.bytes)+' &middot; '+(r.conf||"")+'</div>'
      +reason+'</div>';
    g.appendChild(d);
  }}
}}
function zoom(src){{const m=document.getElementById("modal");m.querySelector("img").src=src;m.style.display="flex";}}
document.getElementById("modal").onclick=function(){{this.style.display="none";}};
document.querySelectorAll("[data-f]").forEach(b=>b.onclick=()=>{{
  fFilter=b.dataset.f; document.querySelectorAll("[data-f]").forEach(x=>x.classList.toggle("on",x===b)); render();}});
document.querySelectorAll("[data-folder]").forEach(b=>b.onclick=()=>{{
  folderFilter=b.dataset.folder; document.querySelectorAll("[data-folder]").forEach(x=>x.classList.toggle("on",x===b)); render();}});
document.getElementById("q").oninput=e=>{{q=e.target.value.toLowerCase();render();}};
render();
</script></body></html>"""

    out = os.path.join(HRI, "dataset_viewer.html")
    with open(out, "w") as f:
        f.write(doc)
    print(f"[viewer] wrote {out}")
    print(f"[viewer] TOTAL {grand_total} sessions, {grand_valid} valid, "
          f"{grand_prompt} prompts, {human(grand_bytes)}")


if __name__ == "__main__":
    main()
