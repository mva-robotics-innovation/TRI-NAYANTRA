/* TRI-NAYANTRA shared frontend library: camera, websocket streaming, dotted overlays, speech, audio. */
const TN = {
  api: window.TN_API || "http://localhost:8000",
  POSE: [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32],[0,11],[0,12]],
  HAND: [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]],
  LANG_TTS: {en:"en-IN",hi:"hi-IN",bho:"hi-IN",mai:"hi-IN",bn:"bn-IN",ta:"ta-IN",te:"te-IN",mr:"mr-IN",gu:"gu-IN",kn:"kn-IN",ml:"ml-IN",pa:"pa-IN",ur:"ur-IN",or:"hi-IN"},

  async get(p){ const r = await fetch(this.api+p); if(!r.ok) throw new Error(await r.text()); return r.json(); },
  async post(p, body){ const r = await fetch(this.api+p,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); if(!r.ok) throw new Error(await r.text()); return r.json(); },
  async form(p, fd){ const r = await fetch(this.api+p,{method:"POST",body:fd}); if(!r.ok) throw new Error((await r.json().catch(()=>({detail:r.statusText}))).detail||r.statusText); return r; },

  async startCam(video){
    const s = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
    video.srcObject = s; await video.play(); return s;
  },
  stopCam(video){ (video.srcObject?.getTracks()||[]).forEach(t=>t.stop()); video.srcObject = null; },

  /* Send JPEG frames to /ws/<mode>, wait for the reply (back-pressure), call onResult(json). */
  stream({mode, video, canvas, onResult, onState, fps=5, cfg}){
    const ws = new WebSocket(this.api.replace(/^http/,"ws")+"/ws/"+mode);
    const grab = document.createElement("canvas"); let alive = true, busy = false;
    const loop = async () => {
      if(!alive || ws.readyState !== 1 || busy || !video.videoWidth) return;
      grab.width = 480; grab.height = Math.round(480*video.videoHeight/video.videoWidth);
      grab.getContext("2d").drawImage(video,0,0,grab.width,grab.height);
      busy = true; grab.toBlob(b => ws.send(b), "image/jpeg", 0.7);
    };
    ws.onopen = () => { onState?.("connected"); if(cfg) ws.send(JSON.stringify(cfg)); this.timer = setInterval(loop, 1000/fps); };
    ws.onclose = () => { onState?.("disconnected"); };
    ws.onerror = () => onState?.("error");
    ws.onmessage = e => {
      busy = false; const r = JSON.parse(e.data);
      if(canvas){ canvas.width = video.videoWidth; canvas.height = video.videoHeight; canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height); }
      onResult(r, canvas?.getContext("2d"), canvas);
    };
    return { send:o=>ws.send(JSON.stringify(o)), stop:()=>{ alive=false; clearInterval(this.timer); ws.close(); } };
  },

  /* ---------- dotted drawing (all coords normalised 0..1) ---------- */
  dots(ctx, pts, color="#22d3ee", r=2){ ctx.fillStyle = color; const W=ctx.canvas.width,H=ctx.canvas.height;
    pts.forEach(p=>{ ctx.beginPath(); ctx.arc(p[0]*W,p[1]*H,r,0,7); ctx.fill(); }); },
  dotLines(ctx, pts, conn, color="#facc15", minVis=0.4){ const W=ctx.canvas.width,H=ctx.canvas.height;
    ctx.strokeStyle=color; ctx.lineWidth=3.5; ctx.lineCap="round"; ctx.setLineDash([0.1,9]);
    conn.forEach(([a,b])=>{ const p=pts[a],q=pts[b]; if(!p||!q||(p[2]!==undefined&&(p[2]<minVis||q[2]<minVis)))return;
      ctx.beginPath(); ctx.moveTo(p[0]*W,p[1]*H); ctx.lineTo(q[0]*W,q[1]*H); ctx.stroke(); }); ctx.setLineDash([]); },
  box(ctx, b, label, color="#f97316"){ const W=ctx.canvas.width,H=ctx.canvas.height; ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash([3,6]);
    ctx.strokeRect(b[0]*W,b[1]*H,(b[2]-b[0])*W,(b[3]-b[1])*H); ctx.setLineDash([]);
    if(label){ ctx.font="bold 14px system-ui"; ctx.fillStyle=color; ctx.fillText(label,b[0]*W+4,Math.max(14,b[1]*H-5)); } },
  ellipse(ctx, o, color="#f472b6", label=true){ const W=ctx.canvas.width,H=ctx.canvas.height; ctx.save(); ctx.translate(o.cx*W,o.cy*H); ctx.rotate(o.angle*Math.PI/180);
    ctx.strokeStyle=color; ctx.lineWidth=2.5; ctx.setLineDash([0.1,7]); ctx.lineCap="round"; ctx.beginPath(); ctx.ellipse(0,0,o.rx*W,o.ry*H,0,0,7); ctx.stroke(); ctx.restore();
    if(label){ ctx.setLineDash([]); ctx.font="bold 12px system-ui"; ctx.fillStyle=color; ctx.fillText(o.name+(o.warmth_idx!=null?" · "+o.warmth_idx:""),o.cx*W+o.rx*W+4,o.cy*H+4); } },
  drawPerson(ctx, r, {face=true,body=true,hands=true,labels=true}={}){
    if(face && r.face_pts){ this.dots(ctx,r.face_pts,"#22d3ee",1.6); if(r.face_box) this.box(ctx,r.face_box,labels?(r.face?.expression||""):"","#22d3ee"); }
    if(body && r.pose){ this.dotLines(ctx,r.pose,this.POSE,"#facc15"); this.dots(ctx,r.pose.filter(p=>p[2]>.4),"#fde047",3); }
    if(hands) (r.hands||[]).forEach(h=>{ this.dotLines(ctx,h.pts,this.HAND,"#4ade80"); this.dots(ctx,h.pts,"#86efac",2.5);
      if(labels&&h.gesture){ ctx.font="bold 16px system-ui"; ctx.fillStyle="#4ade80"; ctx.fillText(h.gesture,h.pts[0][0]*ctx.canvas.width,h.pts[0][1]*ctx.canvas.height+22);} });
  },

  /* ---------- UI helpers ---------- */
  flash(el, on, text){ el.classList.toggle("flash", !!on); const b=el.querySelector(".flash-msg"); if(b) b.textContent = on ? text : ""; },
  beep(){ try{ const a=new AudioContext(),o=a.createOscillator(); o.frequency.value=880; o.connect(a.destination); o.start(); setTimeout(()=>{o.stop();a.close()},250);}catch{} },
  speak(text, lang="en"){ const u=new SpeechSynthesisUtterance(text); u.lang=this.LANG_TTS[lang]||lang; speechSynthesis.cancel(); speechSynthesis.speak(u); },
  set(id, v){ const e=document.getElementById(id); if(e) e.textContent = v ?? "—"; },
  async record(ms, stream){ stream = stream || await navigator.mediaDevices.getUserMedia({audio:true});
    return new Promise(res=>{ const mr=new MediaRecorder(stream), ch=[]; mr.ondataavailable=e=>ch.push(e.data);
      mr.onstop=()=>{ stream.getTracks().forEach(t=>t.stop()); res(new Blob(ch,{type:mr.mimeType})); }; mr.start(); setTimeout(()=>mr.stop(),ms); }); },
  async langs(sel){ try{ const s=await this.get("/api/status"); sel.innerHTML=Object.entries(s.languages).map(([k,v])=>`<option value="${k}">${v}</option>`).join(""); }catch{ sel.innerHTML='<option value="en">English</option><option value="hi">Hindi</option>'; } },
  nav(active){ const L=[["index.html","Home"],["health.html","Health Scan"],["workplace.html","Factory Monitor"],["students.html","Students"],["gesture.html","Sign → Speech"],["advanced.html","Advanced Health"],["voice.html","Voice Assist"]];
    document.getElementById("nav").innerHTML = `<a class="brand" href="index.html"><svg width="26" height="26" viewBox="0 0 32 32"><circle cx="16" cy="16" r="14" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="1 4" stroke-linecap="round"/><circle cx="16" cy="16" r="5" fill="#22d3ee"/><circle cx="16" cy="6" r="2" fill="#f59e0b"/></svg> TRI-NAYANTRA</a>`+L.map(([h,t])=>`<a href="${h}" class="${h===active?"on":""}">${t}</a>`).join(""); },
  disclaimer: "Camera-based estimates are approximate and for wellness/safety awareness only - not a medical diagnosis. Obtain consent from everyone being monitored."
};
