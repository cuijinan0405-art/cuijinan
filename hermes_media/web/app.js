/* Hermes Media 节点画布：ComfyUI 同款的 litegraph，节点定义来自后端 /api/nodes */
(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const STORE_KEY = "hermes-media-graph-v2";
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* 隐私模式等 */ } },
  };

  let DEFS = {};            // type -> 节点定义
  let INFO = {};            // /api/nodes 的完整返回
  let running = null;       // {jobId, nodeIds}
  const execState = { current: null, error: null, done: new Set() };

  // ---------------------------------------------------------------- 画布
  LiteGraph.NODE_TITLE_COLOR = "#ddd";
  LiteGraph.NODE_TEXT_COLOR = "#bbb";
  LiteGraph.NODE_DEFAULT_BGCOLOR = "#353535";
  LiteGraph.NODE_DEFAULT_COLOR = "#2b2b2b";
  LiteGraph.release_link_on_empty_shows_menu = true;
  LiteGraph.alt_drag_do_clone_nodes = true;

  const graph = new LGraph();
  const canvasEl = $("#graph");
  const canvas = new LGraphCanvas(canvasEl, graph);
  window.hm = { graph, canvas };  // 方便在浏览器控制台调试
  canvas.render_canvas_border = false;
  canvas.show_info = false;
  canvas.allow_searchbox = true;
  canvas.links_render_mode = LiteGraph.SPLINE_LINK;
  canvas.default_link_color = "#8a8";
  LGraphCanvas.link_type_colors = Object.assign(LGraphCanvas.link_type_colors || {}, {
    STRING: "#8ab", IMAGE: "#6a6", RATIO: "#b9a", VOICE: "#ca7", EMOTION: "#d88", AUDIO: "#7ad",
  });

  function resize() {
    const wrap = $("#canvas-wrap");
    canvas.resize(wrap.clientWidth, wrap.clientHeight);
  }
  window.addEventListener("resize", resize);

  // 运行中的节点绿框，出错的节点红框（同 ComfyUI）
  canvas.onDrawForeground = function (ctx) {
    for (const node of graph._nodes) {
      let color = null;
      if (String(node.id) === execState.current) color = "#5ec26a";
      else if (String(node.id) === execState.error) color = "#e5534b";
      if (!color) continue;
      ctx.save();
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      const th = LiteGraph.NODE_TITLE_HEIGHT;
      ctx.beginPath();
      ctx.roundRect(node.pos[0] - 4, node.pos[1] - th - 4, node.size[0] + 8, node.size[1] + th + 8, 10);
      ctx.stroke();
      ctx.restore();
    }
  };

  // ---------------------------------------------------------------- 自定义控件：多行文本
  function wrapLines(ctx, text, maxWidth, maxLines) {
    const out = [];
    for (const para of String(text || "").split("\n")) {
      let line = "";
      for (const ch of para) {
        if (ctx.measureText(line + ch).width > maxWidth) {
          out.push(line);
          line = ch;
          if (out.length >= maxLines) return out;
        } else line += ch;
      }
      out.push(line);
      if (out.length >= maxLines) return out;
    }
    return out;
  }

  function textWidget(node, key, label, value, multiline) {
    const H = multiline ? 96 : LiteGraph.NODE_WIDGET_HEIGHT;
    const w = {
      name: key, label, type: "hm_text", value: value ?? "", options: { key },
      computeSize: (width) => [width, H],
      draw(ctx, n, width, y) {
        ctx.save();
        ctx.fillStyle = "#1e1e1e";
        ctx.strokeStyle = "#555";
        ctx.beginPath();
        ctx.roundRect(15, y, width - 30, H, 6);
        ctx.fill();
        ctx.stroke();
        ctx.font = "13px sans-serif";
        ctx.textAlign = "left";
        if (!multiline) {
          ctx.fillStyle = "#999";
          ctx.fillText(label, 24, y + H * 0.7);
          ctx.fillStyle = "#ddd";
          ctx.textAlign = "right";
          const v = String(this.value || "（点击填写）");
          ctx.fillText(v.length > 16 ? v.slice(0, 16) + "…" : v, width - 24, y + H * 0.7);
        } else {
          const v = this.value || "点击这里输入文字…";
          ctx.fillStyle = this.value ? "#ddd" : "#777";
          const lines = wrapLines(ctx, v, width - 50, 5);
          lines.forEach((l, i) => ctx.fillText(l, 24, y + 20 + i * 17));
        }
        ctx.restore();
      },
      mouse(event, pos, n) {
        if (event.type !== LiteGraph.pointerevents_method + "down") return false;
        editText(label, w.value, multiline).then((v) => {
          if (v === null) return;
          w.value = v;
          saveGraph();
          canvas.setDirty(true, true);
        });
        return true;
      },
    };
    node.addCustomWidget(w);
    return w;
  }

  // ---------------------------------------------------------------- 节点注册
  function registerNodes() {
    for (const def of INFO.nodes) {
      DEFS[def.type] = def;
      const typePath = `${def.category}/${def.type}`;
      if (LiteGraph.registered_node_types[typePath]) continue;

      function Node() {
        const d = DEFS[def.type];
        this.hmType = d.type;
        for (const p of d.inputs) this.addInput(p.name, p.type, { label: p.label + (p.optional ? "（可选）" : "") });
        for (const p of d.outputs) this.addOutput(p.name, p.type, { label: p.label });
        for (const w of d.widgets) addWidgetFromDef(this, w);
        addExtraButtons(this, d);
        this.color = "#2b2b2b";
        this.bgcolor = d.color || "#353535";
        this.size = this.computeSize();
        this.size[0] = Math.max(this.size[0], d.output_node ? 300 : 260);
        if (d.type === "PreviewImage") this.size[1] = Math.max(this.size[1], 280);
      }
      Node.title = def.title;
      Node.desc = def.description;
      Node.prototype.onDrawForeground = drawNodeResults;
      LiteGraph.registerNodeType(typePath, Node);
    }
  }

  function addWidgetFromDef(node, w) {
    const opts = { key: w.name };
    let widget;
    if (w.kind === "textarea" || w.kind === "text") {
      widget = textWidget(node, w.name, w.label, w.default, w.kind === "textarea");
    } else if (w.kind === "combo") {
      widget = node.addWidget("combo", w.name, w.default, () => onWidgetChange(node, w.name), { ...opts, values: w.options });
    } else {
      const isInt = w.kind === "int";
      widget = node.addWidget("number", w.name, w.default, () => onWidgetChange(node, w.name), {
        ...opts, min: w.min, max: w.max, step: (w.step || 1) * 10, precision: isInt ? 0 : 2,
      });
    }
    widget.label = w.label;
    return widget;
  }

  function addExtraButtons(node, def) {
    if (def.type === "VoicePreset") {
      node.addWidget("button", "▶ 试听参考音色", null, () => {
        const name = widgetValue(node, "voice");
        playUrl(`/api/voices/audio?name=${encodeURIComponent(name)}`);
      });
      node.addWidget("button", "＋ 上传音色", null, () => uploadVoice(node));
    }
    if (def.type === "PreviewAudio") {
      node.addWidget("button", "▶ 播放", null, () => {
        const a = node.results?.audio?.[0];
        a ? playUrl(a) : toast("还没有音频，先点「运行」");
      });
    }
  }

  function widget(node, key) { return (node.widgets || []).find((w) => w.options?.key === key); }
  function widgetValue(node, key) { return widget(node, key)?.value; }

  function onWidgetChange(node, key) {
    if (node.hmType === "EmotionPreset" && key === "emotion") {
      const a = INFO.emotion_alphas[widgetValue(node, "emotion")];
      const s = widget(node, "strength");
      if (s && a != null) s.value = a;
    }
    saveGraph();
  }

  // 预览图片 / 文本结果画在节点里
  const imgCache = new Map();
  function loadImg(url) {
    if (!imgCache.has(url)) {
      const im = new Image();
      im.onload = () => canvas.setDirty(true, true);
      im.src = url;
      imgCache.set(url, im);
    }
    return imgCache.get(url);
  }

  function drawNodeResults(ctx) {
    const r = this.results;
    if (!r) return;
    let top = this.widgets?.length
      ? (this.widgets[this.widgets.length - 1].last_y || 0) + LiteGraph.NODE_WIDGET_HEIGHT + 10
      : Math.max(this.inputs?.length || 0, this.outputs?.length || 0) * LiteGraph.NODE_SLOT_HEIGHT + 10;
    ctx.save();
    if (r.text) {
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#aaa";
      const lines = wrapLines(ctx, r.text, this.size[0] - 24, 4);
      lines.forEach((l, i) => ctx.fillText(l, 12, top + 12 + i * 15));
      top += lines.length * 15 + 8;
    }
    if (r.images?.length) {
      const imgs = r.images.map(loadImg).filter((im) => im.complete && im.naturalWidth);
      const n = imgs.length;
      if (n) {
        const cols = Math.ceil(Math.sqrt(n));
        const rows = Math.ceil(n / cols);
        const W = this.size[0] - 16, H = this.size[1] - top - 8;
        const cw = W / cols, ch = H / rows;
        imgs.forEach((im, i) => {
          const s = Math.min(cw / im.naturalWidth, ch / im.naturalHeight);
          const w = im.naturalWidth * s, h = im.naturalHeight * s;
          const x = 8 + (i % cols) * cw + (cw - w) / 2, y = top + Math.floor(i / cols) * ch + (ch - h) / 2;
          ctx.drawImage(im, x, y, w, h);
        });
      }
    }
    if (r.audio?.length) {
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#8c8";
      const lines = wrapLines(ctx, "✓ " + decodeURIComponent(r.audio[0].split("/").pop()), this.size[0] - 24, 3);
      lines.forEach((l, i) => ctx.fillText(l, 12, top + 12 + i * 15));
      top += lines.length * 15 + 8;
    }
    ctx.restore();
    // 文字结果放不下时把节点撑高（图片节点按用户拖的大小显示）
    if (!r.images?.length && this.size[1] < top) {
      this.size[1] = top;
      canvas.setDirty(true, true);
    }
  }

  // ---------------------------------------------------------------- 预设模板
  function add(type, pos, values = {}) {
    const def = DEFS[type];
    const node = LiteGraph.createNode(`${def.category}/${type}`);
    node.pos = pos;
    graph.add(node);
    for (const [k, v] of Object.entries(values)) {
      const w = widget(node, k);
      if (w) w.value = v;
    }
    return node;
  }

  const TEMPLATES = {
    image() {
      const text = add("TextInput", [40, 120], { text: "一只戴着墨镜的橘猫，在夕阳下的海边冲浪" });
      const style = add("StylePreset", [360, 120], { style: firstStyle("电影大片") });
      const img = add("QwenImage", [680, 80]);
      const prev = add("PreviewImage", [1010, 60]);
      prev.size = [420, 460];
      text.connect(0, style, 0);
      style.connect(0, img, 0);
      style.connect(1, img, 1);
      style.connect(2, img, 2);
      img.connect(0, prev, 0);
    },
    tts() {
      const text = add("TextInput", [40, 60], { text: "大家好，欢迎来到我的频道。今天给大家分享一个超级实用的小技巧。" });
      const voice = add("VoicePreset", [40, 260]);
      const emo = add("EmotionPreset", [40, 480], { emotion: "开心" });
      onWidgetChange(emo, "emotion");
      const tts = add("IndexTTS2", [400, 200]);
      const prev = add("PreviewAudio", [720, 200]);
      text.connect(0, tts, 0);
      voice.connect(0, tts, 1);
      emo.connect(0, tts, 2);
      tts.connect(0, prev, 0);
    },
    both() {
      const theme = add("TextInput", [40, 60], { text: "清晨的江南古镇，小桥流水，薄雾" });
      const style = add("StylePreset", [360, 60], { style: firstStyle("国风水墨") });
      const img = add("QwenImage", [680, 30]);
      const prevImg = add("PreviewImage", [1010, 20]);
      prevImg.size = [380, 380];
      theme.connect(0, style, 0);
      style.connect(0, img, 0);
      style.connect(1, img, 1);
      style.connect(2, img, 2);
      img.connect(0, prevImg, 0);

      const line = add("TextInput", [40, 420], { text: "清晨的薄雾还没散去，古镇的石板路上已经响起了第一声吆喝。" });
      const voice = add("VoicePreset", [40, 610]);
      const emo = add("EmotionPreset", [40, 820], { emotion: "平静舒缓" });
      onWidgetChange(emo, "emotion");
      const tts = add("IndexTTS2", [400, 560]);
      const prevA = add("PreviewAudio", [720, 560]);
      line.connect(0, tts, 0);
      voice.connect(0, tts, 1);
      emo.connect(0, tts, 2);
      tts.connect(0, prevA, 0);
    },
  };

  function firstStyle(name) {
    const s = INFO.styles.find((x) => x.name === name) || INFO.styles[0];
    return s?.name;
  }

  function loadTemplate(name) {
    graph.clear();
    TEMPLATES[name]();
    document.querySelectorAll(".tpl").forEach((b) => b.classList.toggle("active", b.dataset.tpl === name));
    canvas.ds.offset = [20, 20];
    canvas.ds.scale = 1;
    fitView();
    store.set(STORE_KEY + ":tpl", name);
    saveGraph();
  }

  function fitView() {
    if (!graph._nodes.length) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of graph._nodes) {
      x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1] - LiteGraph.NODE_TITLE_HEIGHT);
      x1 = Math.max(x1, n.pos[0] + n.size[0]); y1 = Math.max(y1, n.pos[1] + n.size[1]);
    }
    const W = canvasEl.width / (window.devicePixelRatio || 1), H = canvasEl.height / (window.devicePixelRatio || 1);
    const s = Math.min(1.1, Math.max(0.35, Math.min((W - 40) / (x1 - x0), (H - 40) / (y1 - y0))));
    canvas.ds.scale = s;
    canvas.ds.offset = [(W / s - (x1 - x0)) / 2 - x0, (H / s - (y1 - y0)) / 2 - y0];
    canvas.setDirty(true, true);
  }

  function saveGraph() {
    store.set(STORE_KEY, JSON.stringify(graph.serialize()));
  }

  // ---------------------------------------------------------------- 运行
  function toApiGraph() {
    const out = {};
    for (const node of graph._nodes) {
      const def = DEFS[node.hmType];
      if (!def) continue;
      const widgets = {};
      for (const w of node.widgets || []) if (w.options?.key) widgets[w.options.key] = w.value;
      const inputs = {};
      (node.inputs || []).forEach((inp, i) => {
        if (inp.link == null) return;
        const link = graph.links[inp.link];
        if (link) inputs[def.inputs[i].name] = [String(link.origin_id), link.origin_slot];
      });
      out[String(node.id)] = { type: def.type, widgets, inputs };
    }
    return out;
  }

  async function run() {
    if (running) return toast("上一个任务还在跑…");
    const g = toApiGraph();
    if (!Object.keys(g).length) return toast("画布是空的，先选一个预设");
    execState.error = null;
    execState.done.clear();
    graph._nodes.forEach((n) => { if (DEFS[n.hmType]?.output_node || n.hmType === "StylePreset") n.results = null; });
    $("#run").disabled = true;
    $("#run").textContent = "⏳ 运行中…";
    try {
      const r = await api("/api/run", { method: "POST", body: JSON.stringify({ graph: g }) });
      running = { jobId: r.job_id, started: Date.now() };
      poll();
    } catch (e) {
      finish();
      toast(e.message);
    }
  }

  async function poll() {
    if (!running) return;
    let job;
    try { job = await api(`/api/jobs/${running.jobId}`); } catch (e) { setTimeout(poll, 1500); return; }
    execState.current = job.current;
    for (const [nid, res] of Object.entries(job.results || {})) {
      const node = graph.getNodeById(Number(nid));
      if (node) node.results = res;
    }
    canvas.setDirty(true, true);
    if (job.status === "done" || job.status === "error") {
      execState.error = job.error_node;
      showJobResult(job);
      finish();
      if (job.status === "error") toast("出错了：" + job.error, 6000);
      return;
    }
    setTimeout(poll, 700);
  }

  function finish() {
    running = null;
    execState.current = null;
    $("#run").disabled = false;
    $("#run").textContent = "▶ 运行";
    $("#progress div").style.width = "0";
    canvas.setDirty(true, true);
  }

  function showJobResult(job) {
    const box = $("#results");
    box.querySelector(".empty")?.remove();
    const el = document.createElement("div");
    el.className = "result";
    const secs = job.finished ? ((job.finished - job.created)).toFixed(1) : "";
    el.innerHTML = `<div class="meta">${new Date().toLocaleTimeString()} · ${job.status === "done" ? "完成" : "失败"} · ${secs}s</div>`;
    if (job.error) {
      const p = document.createElement("div");
      p.className = "err";
      p.textContent = job.error;
      el.appendChild(p);
    }
    for (const res of Object.values(job.results || {})) {
      for (const url of res.images || []) el.appendChild(mediaImg(url));
      for (const url of res.audio || []) el.appendChild(mediaAudio(url));
      if (res.seeds) {
        const m = document.createElement("div");
        m.className = "meta";
        m.textContent = `尺寸 ${res.size} · 种子 ${res.seeds.join(", ")}`;
        el.appendChild(m);
      }
    }
    box.prepend(el);
  }

  function mediaImg(url) {
    const wrap = document.createElement("div");
    const img = document.createElement("img");
    img.src = url;
    img.alt = "生成的图片";
    img.addEventListener("click", () => { $("#lightbox img").src = url; $("#lightbox").hidden = false; });
    const a = document.createElement("a");
    a.href = url; a.download = ""; a.textContent = "下载 " + decodeURIComponent(url.split("/").pop());
    wrap.append(img, a);
    return wrap;
  }

  function mediaAudio(url) {
    const wrap = document.createElement("div");
    const au = document.createElement("audio");
    au.controls = true; au.src = url; au.preload = "metadata";
    const a = document.createElement("a");
    a.href = url; a.download = ""; a.textContent = "下载 " + decodeURIComponent(url.split("/").pop());
    wrap.append(au, a);
    return wrap;
  }

  async function loadHistory() {
    const items = await api("/api/history");
    const box = $("#results");
    box.innerHTML = items.length ? "" : '<p class="empty">还没有生成过东西。</p>';
    for (const it of items) {
      const el = document.createElement("div");
      el.className = "result";
      el.appendChild(it.kind === "image" ? mediaImg(it.url) : mediaAudio(it.url));
      box.appendChild(el);
    }
  }

  // ---------------------------------------------------------------- 状态栏
  const STATE_TEXT = { ready: "就绪", loading: "加载模型中", starting: "启动中", stopped: "未启动（用到时自动启动）",
    error: "出错", unconfigured: "未配置" };
  async function refreshStatus() {
    try {
      const s = await api("/api/status");
      const pills = [];
      const label = { image: "千问 Image 2.1", tts: "IndexTTS2" };
      for (const key of ["image", "tts"]) {
        const backend = s[key + "_backend"];
        const w = s.workers[key];
        if (!w) { pills.push(pill(label[key], "ready", `后端：${backend}`)); continue; }
        const model = key === "image" && w.model_id ? `（${w.model_id}）` : "";
        pills.push(pill(label[key], w.state, (STATE_TEXT[w.state] || w.state) + model + (w.error ? "：" + w.error : "")));
      }
      $("#status").innerHTML = pills.join("");
      if (s.progress && running) {
        $("#progress div").style.width = `${(100 * s.progress.step) / s.progress.total}%`;
      }
    } catch { /* 后端重启中 */ }
    setTimeout(refreshStatus, running ? 1000 : 4000);
  }
  function pill(name, state, title) {
    const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    return `<span class="pill" title="${esc(title)}"><span class="dot ${esc(state)}"></span>${esc(name)}</span>`;
  }

  // ---------------------------------------------------------------- 模型管理
  const fmtGB = (b) => (b / 1024 ** 3).toFixed(b > 10 * 1024 ** 3 ? 0 : 1) + " GB";

  async function openModels() {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<div class="models-head">下载源
        <select id="msource"></select>
        <span id="mdir"></span></div>
      <div class="model-list" id="mlist">加载中…</div>
      <div class="tip">下载完成后在「千问 Image 2.1」节点的「模型」下拉框里选择；切换模型会自动重新加载（需要一点时间）。
        国内网络建议选「HF 国内镜像」或「ModelScope 魔搭」。</div>`;
    let timer = null, firstRender = true;
    const render = async () => {
      let data;
      try { data = await api("/api/models"); } catch (e) { wrap.querySelector("#mlist").textContent = e.message; return; }
      const sel = wrap.querySelector("#msource");
      if (firstRender) {
        sel.innerHTML = Object.entries(data.sources).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
        sel.value = store.get("hm-source") || data.default_source;
        sel.onchange = () => store.set("hm-source", sel.value);
        wrap.querySelector("#mdir").textContent = "保存到 " + data.models_dir;
        firstRender = false;
      }
      const list = wrap.querySelector("#mlist");
      list.innerHTML = "";
      let anyRunning = false;
      for (const m of data.entries) {
        const t = data.tasks[m.id];
        const row = document.createElement("div");
        row.className = "model-row";
        const top = document.createElement("div");
        top.className = "top";
        const name = document.createElement("div");
        name.className = "name";
        name.textContent = m.name + (m.id === data.default_model ? "（默认）" : "");
        top.appendChild(name);
        if (t && t.state === "running") {
          anyRunning = true;
          const tag = document.createElement("span");
          tag.className = "tag";
          tag.textContent = t.total ? `下载中 ${fmtGB(t.downloaded)} / ${fmtGB(t.total)}` : `下载中 ${fmtGB(t.downloaded)}`;
          top.appendChild(tag);
        } else if (m.installed) {
          const tag = document.createElement("span");
          tag.className = "tag ok";
          tag.textContent = "✓ 可用";
          top.appendChild(tag);
        } else if (m.downloadable) {
          const btn = document.createElement("button");
          btn.className = "small";
          btn.textContent = t && t.state === "error" ? "重试" : "下载";
          btn.onclick = async () => {
            try {
              await api("/api/models/download", { method: "POST", body: JSON.stringify({ id: m.id, source: sel.value }) });
              render();
            } catch (e) { toast(e.message, 6000); }
          };
          top.appendChild(btn);
        }
        row.appendChild(top);
        const desc = document.createElement("div");
        desc.className = "desc";
        desc.textContent = m.desc + (m.base ? `（需要先下载：${data.entries.find((x) => x.id === m.base)?.name || m.base}）` : "");
        row.appendChild(desc);
        if (t && t.state === "running" && t.total) {
          const bar = document.createElement("div");
          bar.className = "bar";
          bar.innerHTML = `<div style="width:${Math.min(100, (100 * t.downloaded) / t.total).toFixed(1)}%"></div>`;
          row.appendChild(bar);
        }
        if (t && t.state === "error") {
          const err = document.createElement("div");
          err.className = "err";
          err.textContent = t.error;
          row.appendChild(err);
        }
        list.appendChild(row);
      }
      if (!anyRunning && timer) { clearInterval(timer); timer = null; await refreshDefs(); }
      if (anyRunning && !timer) timer = setInterval(render, 1500);
    };
    render();
    await modal("模型管理", wrap, "关闭", false);
    if (timer) clearInterval(timer);
    await refreshDefs();
    canvas.setDirty(true, true);
  }

  // ---------------------------------------------------------------- 弹窗 / 工具
  function modal(title, bodyEl, okText = "确定", showCancel = true) {
    return new Promise((resolve) => {
      $("#modal-cancel").hidden = !showCancel;
      $("#modal-title").textContent = title;
      const body = $("#modal-body");
      body.innerHTML = "";
      body.appendChild(bodyEl);
      $("#modal-ok").textContent = okText;
      $("#modal").hidden = false;
      const close = (v) => {
        $("#modal").hidden = true;
        $("#modal-ok").onclick = $("#modal-cancel").onclick = null;
        document.removeEventListener("keydown", onKey, true);
        resolve(v);
      };
      const onKey = (e) => {
        if (e.key === "Escape") { e.stopPropagation(); close(false); }
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.stopPropagation(); close(true); }
      };
      document.addEventListener("keydown", onKey, true);
      $("#modal-ok").onclick = () => close(true);
      $("#modal-cancel").onclick = () => close(false);
      setTimeout(() => bodyEl.querySelector("textarea,input")?.focus(), 30);
    });
  }

  async function editText(label, value, multiline) {
    const wrap = document.createElement("div");
    const input = document.createElement(multiline ? "textarea" : "input");
    if (!multiline) input.type = "text";
    input.value = value || "";
    const tip = document.createElement("div");
    tip.className = "tip";
    tip.textContent = "Ctrl + Enter 确定，Esc 取消";
    wrap.append(input, tip);
    return (await modal(label, wrap)) ? input.value : null;
  }

  async function uploadVoice(node) {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<label for="vname">音色名字（以后在下拉框里选它）</label>
      <input id="vname" type="text" placeholder="例如：我的声音、温柔女声">
      <label for="vfile">参考音频（3~10 秒、干净无背景音的人声，wav/mp3）</label>
      <input id="vfile" type="file" accept="audio/*">
      <div class="tip">保存到 voices 目录，也可以直接把音频文件拷进去。</div>`;
    if (!(await modal("上传音色", wrap, "上传"))) return;
    const file = wrap.querySelector("#vfile").files[0];
    let name = wrap.querySelector("#vname").value.trim();
    if (!file) return toast("没有选择音频文件");
    if (!name) name = file.name.replace(/\.[^.]+$/, "");
    const ext = (file.name.match(/\.[^.]+$/) || [".wav"])[0];
    try {
      const r = await api(`/api/voices/upload?name=${encodeURIComponent(name)}&ext=${encodeURIComponent(ext)}`,
        { method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" } });
      await refreshDefs();
      const w = widget(node, "voice");
      if (w) w.value = r.name;
      saveGraph();
      canvas.setDirty(true, true);
      toast(`音色「${r.name}」已添加`);
    } catch (e) { toast(e.message); }
  }

  async function refreshDefs() {
    INFO = await api("/api/nodes");
    for (const def of INFO.nodes) DEFS[def.type] = def;
    // 更新已有节点的下拉选项
    for (const node of graph._nodes) {
      const def = DEFS[node.hmType];
      if (!def) continue;
      for (const wd of def.widgets) {
        const w = widget(node, wd.name);
        if (w && wd.kind === "combo") w.options.values = wd.options;
      }
    }
  }

  let player = null;
  function playUrl(url) {
    if (player) player.pause();
    player = new Audio(url);
    player.play().catch(() => toast("播放失败：音频不存在或格式不支持"));
  }

  let toastTimer = null;
  function toast(msg, ms = 3000) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), ms);
  }

  async function api(url, opts = {}) {
    const headers = opts.body && !(opts.body instanceof Blob) ? { "Content-Type": "application/json" } : {};
    const r = await fetch(url, { ...opts, headers: { ...headers, ...(opts.headers || {}) } });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `请求失败 ${r.status}`);
    return data;
  }

  // ---------------------------------------------------------------- 启动
  async function init() {
    resize();
    INFO = await api("/api/nodes");
    registerNodes();
    const saved = store.get(STORE_KEY);
    let restored = false;
    if (saved) {
      try { graph.configure(JSON.parse(saved)); restored = graph._nodes.length > 0; } catch { graph.clear(); }
    }
    if (restored) {
      const tpl = store.get(STORE_KEY + ":tpl");
      document.querySelectorAll(".tpl").forEach((b) => b.classList.toggle("active", b.dataset.tpl === tpl));
      await refreshDefs();
      fitView();
    } else {
      loadTemplate("image");
    }
    graph.onAfterChange = saveGraph;
    canvasEl.addEventListener("mouseup", () => setTimeout(saveGraph, 0));

    document.querySelectorAll(".tpl").forEach((b) =>
      b.addEventListener("click", () => loadTemplate(b.dataset.tpl)));
    $("#run").addEventListener("click", run);
    $("#open-models").addEventListener("click", openModels);
    $("#toggle-side").addEventListener("click", () => { $("#side").classList.toggle("hidden"); resize(); });
    $("#refresh-history").addEventListener("click", loadHistory);
    $("#lightbox").addEventListener("click", () => ($("#lightbox").hidden = true));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && $("#modal").hidden) { e.preventDefault(); run(); }
    });
    refreshStatus();
  }

  init().catch((e) => toast("初始化失败：" + e.message, 10000));
})();
