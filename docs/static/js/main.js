/* From Images to Signals — page-specific tables and the strings they build.
   Numbers are transcribed from the TMLR paper's tables; nothing is computed
   here beyond best/second marking. */
(function () {
  "use strict";

  const { $, ZH, tr, el, registerRender, rankMarks, fmt } = MC;

  /* ------------------------------------------------- runtime strings (zh) */
  Object.assign(ZH, {
    "Dataset": "数据集",
    "Method": "方法",
    "Setting": "设置",
    "Model": "模型",
    "# Wins": "最优次数",
    "LVM-based": "基于 LVM",
    "Baselines": "基线",
    "TSC task (accuracy %, ↑)": "TSC 任务（准确率 %，↑）",
    "TSF task (MSE, ↓)": "TSF 任务（MSE，↓）",
    "TSC task": "TSC 任务",
    "TSF task": "TSF 任务",
    "Fine-tuning strategies": "微调策略",
    "Simpler models": "更简单的模型",
    "(a) Fine-tuning all parameters": "(a) 微调全部参数",
    "(b) Fine-tuning all but CLS & Mask": "(b) 微调除 CLS 与 Mask 外的全部参数",
    "(c) Fine-tuning MLP & norm": "(c) 微调 MLP 与归一化层",
    "(d) Fine-tuning norm": "(d) 只微调归一化层",
    "(e) Freezing all parameters": "(e) 冻结全部参数",
    "(f) Training from scratch": "(f) 从头训练",
    "Copy": "复制",
    "Copied": "已复制"
  });

  /* ============================================ TSF comparison (averaged) */
  /* `lvm` columns get the light wash; `sep` starts a column group. */
  const TSF_MODELS = [
    { name: "MAE", lvm: true },
    { name: "ViT", lvm: true },
    { name: "Time-LLM", sep: true },
    { name: "GPT4TS" },
    { name: "CALF" },
    { name: "DLinear" },
    { name: "PatchTST" },
    { name: "TimesNet" },
    { name: "FEDformer" },
    { name: "Autoformer" }
  ];

  /* [dataset, MSE ×10, MAE ×10] in TSF_MODELS order. */
  const TSF = [
    ["ETTh1",
      [0.409, 0.445, 0.418, 0.418, 0.432, 0.423, 0.413, 0.458, 0.440, 0.496],
      [0.419, 0.449, 0.432, 0.421, 0.431, 0.437, 0.431, 0.450, 0.460, 0.487]],
    ["ETTh2",
      [0.357, 0.389, 0.361, 0.354, 0.351, 0.431, 0.330, 0.414, 0.437, 0.450],
      [0.390, 0.411, 0.396, 0.389, 0.384, 0.447, 0.379, 0.427, 0.449, 0.459]],
    ["ETTm1",
      [0.345, 0.409, 0.356, 0.363, 0.396, 0.357, 0.351, 0.400, 0.448, 0.588],
      [0.374, 0.422, 0.377, 0.378, 0.391, 0.379, 0.381, 0.406, 0.452, 0.517]],
    ["ETTm2",
      [0.268, 0.300, 0.261, 0.254, 0.283, 0.267, 0.255, 0.291, 0.305, 0.327],
      [0.327, 0.337, 0.316, 0.311, 0.323, 0.334, 0.315, 0.333, 0.349, 0.371]],
    ["Weather",
      [0.225, 0.234, 0.244, 0.227, 0.251, 0.249, 0.226, 0.259, 0.309, 0.338],
      [0.258, 0.273, 0.270, 0.255, 0.274, 0.300, 0.264, 0.287, 0.360, 0.382]],
    ["Illness",
      [1.837, 2.179, 2.018, 1.871, 1.700, 2.169, 1.443, 2.139, 2.847, 3.006],
      [0.883, 1.016, 0.894, 0.852, 0.869, 1.041, 0.798, 0.931, 1.144, 1.161]],
    ["Traffic",
      [0.386, 0.430, 0.422, 0.421, 0.444, 0.434, 0.391, 0.620, 0.610, 0.628],
      [0.256, 0.343, 0.281, 0.274, 0.284, 0.295, 0.264, 0.336, 0.376, 0.379]],
    ["Electricity",
      [0.159, 0.173, 0.165, 0.170, 0.176, 0.166, 0.162, 0.193, 0.214, 0.227],
      [0.250, 0.266, 0.259, 0.263, 0.266, 0.264, 0.253, 0.295, 0.327, 0.338]]
  ];
  const TSF_WINS = [9, 0, 0, 3, 0, 0, 4, 0, 0, 0];

  function renderTSF() {
    const table = $("#tsf-table");
    if (!table) return;
    const head = table.querySelector("thead");
    const body = table.querySelector("tbody");
    head.innerHTML = "";
    body.innerHTML = "";

    const top = el("tr", "grp");
    top.appendChild(el("th", "name"));
    const lvmTh = el("th", null, tr("LVM-based"));
    lvmTh.colSpan = 4;
    const baseTh = el("th", "sep", tr("Baselines"));
    baseTh.colSpan = 16;
    top.appendChild(lvmTh);
    top.appendChild(baseTh);
    head.appendChild(top);

    const grp = el("tr", "grp");
    grp.appendChild(el("th", "name"));
    TSF_MODELS.forEach((m, i) => {
      const th = el("th", m.sep || i > 0 ? "sep" : null, m.name);
      th.colSpan = 2;
      grp.appendChild(th);
    });
    head.appendChild(grp);

    const hr = document.createElement("tr");
    hr.appendChild(el("th", "name", tr("Dataset")));
    TSF_MODELS.forEach((m, i) => {
      ["MSE", "MAE"].forEach((mt, j) => hr.appendChild(el("th", (m.sep || i > 0) && j === 0 ? "sep" : null, mt)));
    });
    head.appendChild(hr);

    TSF.forEach(([ds, mse, mae]) => {
      const marks = [rankMarks(mse, "lower"), rankMarks(mae, "lower")];
      const row = document.createElement("tr");
      row.appendChild(el("td", "name", ds));
      TSF_MODELS.forEach((m, i) => {
        [mse, mae].forEach((vals, j) => {
          const cls = [marks[j][i], m.lvm ? "ours" : "", (m.sep || i > 0) && j === 0 ? "sep" : ""]
            .filter(Boolean).join(" ");
          row.appendChild(el("td", cls || null, fmt(vals[i])));
        });
      });
      body.appendChild(row);
    });

    const winMarks = rankMarks(TSF_WINS, "higher");
    const wr = el("tr", "wins");
    wr.appendChild(el("td", "name", tr("# Wins")));
    TSF_MODELS.forEach((m, i) => {
      const td = el("td", [winMarks[i], m.lvm ? "ours" : "", m.sep || i > 0 ? "sep" : ""]
        .filter(Boolean).join(" ") || null, String(TSF_WINS[i]));
      td.colSpan = 2;
      wr.appendChild(td);
    });
    body.appendChild(wr);
  }
  registerRender(renderTSF);

  /* ================================================ ablations (RQ4, RQ5) */
  const AB_DATASETS = ["UWave.", "Spoken.", "Handwrit.", "FaceDetect.", "ETTh1", "ETTm1", "Illness", "Weather"];

  /* Four TSC accuracies (%), then four TSF MSEs. */
  const ABLATION = [
    ["Fine-tuning strategies", [
      ["(a) Fine-tuning all parameters",     [88.4, 98.5, 36.4, 67.4, 0.558, 0.399, 1.781, 0.273]],
      ["(b) Fine-tuning all but CLS & Mask", [87.5, 98.2, 35.2, 66.3, 0.530, 0.408, 1.783, 0.275]],
      ["(c) Fine-tuning MLP & norm",         [88.7, 98.4, 35.5, 67.1, 0.532, 0.396, 1.737, 0.264]],
      ["(d) Fine-tuning norm",               [81.6, 98.0, 28.5, 65.2, 0.409, 0.345, 1.837, 0.225]],
      ["(e) Freezing all parameters",        [84.0, 98.5, 27.8, 66.7, 0.452, 0.420, 2.037, 0.308]],
      ["(f) Training from scratch",          [73.4, 97.0, 24.3, 65.0, 0.475, 0.372, 1.723, 0.241]]
    ]],
    ["Simpler models", [
      ["w/o-LVM",  [78.6, 96.4, 22.4, 64.1, 0.423, 0.376, 2.291, 0.255]],
      ["LVM2Attn", [80.1, 96.5, 20.7, 66.2, 0.428, 0.357, 2.108, 0.254]]
    ]]
  ];

  function renderAblation() {
    const table = $("#ablation-table");
    if (!table) return;
    const head = table.querySelector("thead");
    const body = table.querySelector("tbody");
    head.innerHTML = "";
    body.innerHTML = "";

    const grp = el("tr", "grp");
    grp.appendChild(el("th", "name"));
    const tsc = el("th", "sep", tr("TSC task (accuracy %, ↑)"));
    tsc.colSpan = 4;
    const tsf = el("th", "sep", tr("TSF task (MSE, ↓)"));
    tsf.colSpan = 4;
    grp.appendChild(tsc);
    grp.appendChild(tsf);
    head.appendChild(grp);

    const hr = document.createElement("tr");
    hr.appendChild(el("th", "name", tr("Setting")));
    AB_DATASETS.forEach((d, j) => hr.appendChild(el("th", j % 4 === 0 ? "sep" : null, d)));
    head.appendChild(hr);

    /* best and second-best per column, across every row of both panels */
    const all = ABLATION.flatMap(([, rows]) => rows.map((r) => r[1]));
    const marks = AB_DATASETS.map((_, j) =>
      rankMarks(all.map((v) => v[j]), j < 4 ? "higher" : "lower"));

    let k = 0;
    ABLATION.forEach(([section, rows]) => {
      const sr = el("tr", "section-row");
      const td = el("td", null, tr(section));
      td.colSpan = 1 + AB_DATASETS.length;
      sr.appendChild(td);
      body.appendChild(sr);

      rows.forEach(([label, vals]) => {
        const row = document.createElement("tr");
        row.appendChild(el("td", "name left", tr(label)));
        vals.forEach((v, j) => {
          const cls = [marks[j][k], j % 4 === 0 ? "sep" : ""].filter(Boolean).join(" ");
          row.appendChild(el("td", cls || null, fmt(v, j < 4 ? 1 : 3)));
        });
        body.appendChild(row);
        k += 1;
      });
    });
  }
  registerRender(renderAblation);

  /* ============================================ temporal perturbation (RQ6) */
  /* Relative performance drop (%) for w/o-LVM, LVM2Attn, LVM. */
  const PERTURB = [
    ["Sf-All", [
      [78.2, 49.7, 81.7, 19.3, 76.2, 98.4, 116.4, 24.1],
      [86.4, 50.6, 89.9, 22.4, 79.7, 117.1, 109.1, 24.4],
      [80.7, 84.7, 91.5, 29.2, 83.8, 118.4, 162.8, 44.5]]],
    ["Sf-Half", [
      [6.6, 12.4, 74.6, 10.8, 14.4, 28.3, 41.6, 2.4],
      [8.7, 11.6, 83.6, 11.3, 19.5, 44.8, 69.3, 2.4],
      [36.4, 30.2, 86.5, 9.3, 14.5, 48.2, 21.3, 9.6]]],
    ["Ex-Half", [
      [98.8, 82.2, 83.5, 22.8, 13.0, 145.3, 11.0, 34.0],
      [98.9, 82.3, 87.0, 24.6, 9.1, 158.3, 27.9, 35.5],
      [59.4, 89.9, 97.0, 9.2, 14.2, 242.3, 23.0, 67.2]]],
    ["Masking", [
      [-1.0, 3.1, 22.3, -1.2, 47.3, 58.5, 94.1, 33.4],
      [1.0, 3.6, 20.3, 2.7, 46.0, 70.3, 127.8, 33.6],
      [29.0, 41.8, 56.0, 7.4, 47.5, 58.4, 128.9, 49.6]]]
  ];
  const PERTURB_MODELS = ["w/o-LVM", "LVM2Attn", "LVM"];

  function renderPerturb() {
    const table = $("#perturb-table");
    if (!table) return;
    const head = table.querySelector("thead");
    const body = table.querySelector("tbody");
    head.innerHTML = "";
    body.innerHTML = "";

    const grp = el("tr", "grp");
    grp.appendChild(el("th", "name"));
    const tsc = el("th", "sep", tr("TSC task"));
    tsc.colSpan = 4;
    const tsf = el("th", "sep", tr("TSF task"));
    tsf.colSpan = 4;
    grp.appendChild(tsc);
    grp.appendChild(tsf);
    head.appendChild(grp);

    const hr = document.createElement("tr");
    hr.appendChild(el("th", "name", tr("Model")));
    AB_DATASETS.forEach((d, j) => hr.appendChild(el("th", j % 4 === 0 ? "sep" : null, d)));
    head.appendChild(hr);

    PERTURB.forEach(([name, rows]) => {
      const sr = el("tr", "section-row");
      const td = el("td", null, name);
      td.colSpan = 1 + AB_DATASETS.length;
      sr.appendChild(td);
      body.appendChild(sr);

      /* the paper marks only the largest drop under each perturbation */
      const largest = AB_DATASETS.map((_, j) => Math.max(...rows.map((r) => r[j])));
      rows.forEach((vals, i) => {
        const row = document.createElement("tr");
        row.appendChild(el("td", "name left", PERTURB_MODELS[i]));
        vals.forEach((v, j) => {
          const cls = [v === largest[j] ? "best" : "", j % 4 === 0 ? "sep" : ""].filter(Boolean).join(" ");
          row.appendChild(el("td", cls || null, v.toFixed(1) + "%"));
        });
        body.appendChild(row);
      });
    });
  }
  registerRender(renderPerturb);

  MC.start();   /* always last */
})();
