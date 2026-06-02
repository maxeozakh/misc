(() => {
  const model = window.SMOL_RNN_MODEL;

  const $ = (id) => document.getElementById(id);
  const output = $("output");
  const seedInput = $("seed");
  const temperatureInput = $("temperature");
  const temperatureValue = $("temperature-value");
  const charDelayInput = $("char-delay");
  const charDelayValue = $("char-delay-value");
  const useCheckpointStateInput = $("use-checkpoint-state");
  const startButton = $("start");
  const pauseButton = $("pause");
  const resetButton = $("reset");

  let h = null;
  let ix = null;
  let running = false;
  let timer = null;

  function tensor(name) {
    return model.tensors[name];
  }

  function at(matrix, row, col) {
    return matrix.data[row * matrix.cols + col];
  }

  function zeroHidden() {
    return new Float32Array(tensor("Whh").rows);
  }

  function checkpointHidden() {
    return new Float32Array(tensor("hprev").data);
  }

  function matVecPlusBias(matrix, vector, bias) {
    const out = new Float32Array(matrix.rows);
    for (let r = 0; r < matrix.rows; r += 1) {
      let sum = bias.data[r];
      const offset = r * matrix.cols;
      for (let c = 0; c < matrix.cols; c += 1) {
        sum += matrix.data[offset + c] * vector[c];
      }
      out[r] = sum;
    }
    return out;
  }

  function rnnStep(inputIx) {
    const E = tensor("E");
    const Wxh = tensor("Wxh");
    const Whh = tensor("Whh");
    const bh = tensor("bh");
    const Why = tensor("Why");
    const by = tensor("by");

    const nextH = new Float32Array(Whh.rows);
    for (let r = 0; r < Whh.rows; r += 1) {
      let sum = bh.data[r];

      for (let c = 0; c < Wxh.cols; c += 1) {
        sum += at(Wxh, r, c) * at(E, c, inputIx);
      }

      const whhOffset = r * Whh.cols;
      for (let c = 0; c < Whh.cols; c += 1) {
        sum += Whh.data[whhOffset + c] * h[c];
      }

      nextH[r] = Math.tanh(sum);
    }
    h = nextH;

    return matVecPlusBias(Why, h, by);
  }

  function sampleFromLogits(logits, temperature) {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i += 1) {
      const value = logits[i] / temperature;
      if (value > maxLogit) maxLogit = value;
    }

    const probs = new Float32Array(logits.length);
    let total = 0;
    for (let i = 0; i < logits.length; i += 1) {
      const p = Math.exp(logits[i] / temperature - maxLogit);
      probs[i] = p;
      total += p;
    }

    let threshold = Math.random() * total;
    for (let i = 0; i < probs.length; i += 1) {
      threshold -= probs[i];
      if (threshold <= 0) return i;
    }
    return probs.length - 1;
  }

  function buildVocab() {
    const charToIx = new Map();
    model.chars.forEach((ch, index) => {
      charToIx.set(ch, index);
    });
    return charToIx;
  }

  const charToIx = buildVocab();

  function prepareSeed(seed) {
    const cleanSeed = seed || "KING RICHARD:";
    const missing = [...new Set(cleanSeed)].filter((ch) => !charToIx.has(ch));
    if (missing.length > 0) {
      throw new Error(`Seed has chars not in vocab: ${JSON.stringify(missing)}`);
    }

    h = useCheckpointStateInput.checked ? checkpointHidden() : zeroHidden();
    ix = charToIx.get(cleanSeed[0]);

    for (const ch of cleanSeed.slice(1)) {
      rnnStep(ix);
      ix = charToIx.get(ch);
    }

    output.classList.remove("error");
    output.textContent = cleanSeed;
    scrollOutput();
  }

  function generateOne() {
    const temperature = Number(temperatureInput.value);
    const logits = rnnStep(ix);
    ix = sampleFromLogits(logits, temperature);
    output.textContent += model.chars[ix];
    scrollOutput();
  }

  function tick() {
    if (!running) return;
    generateOne();
    timer = window.setTimeout(tick, Number(charDelayInput.value));
  }

  function start() {
    if (running) return;
    if (h === null || ix === null) {
      try {
        prepareSeed(seedInput.value);
      } catch (error) {
        output.classList.add("error");
        output.textContent = error.message;
        return;
      }
    }
    running = true;
    startButton.disabled = true;
    pauseButton.disabled = false;
    tick();
  }

  function pause() {
    running = false;
    startButton.disabled = false;
    pauseButton.disabled = true;
    if (timer !== null) window.clearTimeout(timer);
  }

  function reset() {
    pause();
    h = null;
    ix = null;
    output.classList.remove("error");
    output.textContent = "";
  }

  function scrollOutput() {
    output.scrollTop = output.scrollHeight;
  }

  function syncControls() {
    temperatureValue.value = Number(temperatureInput.value).toFixed(2);
    charDelayValue.value = `${charDelayInput.value}ms`;
  }

  function hydrateStats() {
    $("iteration").textContent = model.iteration.toLocaleString();
    $("loss-ce").textContent = model.lossCe.toFixed(4);
    $("vocab-size").textContent = model.chars.length.toString();
  }

  document.querySelectorAll("[data-seed]").forEach((button) => {
    button.addEventListener("click", () => {
      seedInput.value = button.dataset.seed;
      reset();
    });
  });

  startButton.addEventListener("click", start);
  pauseButton.addEventListener("click", pause);
  resetButton.addEventListener("click", reset);
  temperatureInput.addEventListener("input", syncControls);
  charDelayInput.addEventListener("input", syncControls);
  seedInput.addEventListener("input", reset);
  useCheckpointStateInput.addEventListener("change", reset);

  hydrateStats();
  syncControls();
  start();
})();
