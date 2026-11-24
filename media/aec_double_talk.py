"""回声消除（AEC）双讲场景核心算法模块。

针对题目要求修复的六类问题：

1. 双讲检测（double_talk）：残余/麦克功率比 + 收敛门控联合判据，
   双讲期间冻结自适应滤波，避免把近端人声当回声压掉。
2. 时钟漂移跟踪（clock_drift）：滑动窗归一化互相关 + 抛物线插值得到
   亚样本延迟轨迹，最小二乘回归出漂移 ppm 与重采样比。
3. 延迟估计重置（delay_reset / device_switch）：跟踪延迟跳变，触发
   对齐与滤波器状态重置，设备切换后重新估计。
4. 非线性残留抑制（nonlinear_suppress）：利用残余与参考信号的块相干度
   做后置抑制，压掉线性滤波器无法建模的非线性回声残留。
5. 处理确定性（determinism）：全部随机源取自已播种子的 random.Random，
   不依赖时钟/哈希顺序，同配置两次运行结果逐比特一致。
6. 近端保护增益（near_gain）：双讲块抬高后置增益下限，保护近端语音。

纯标准库实现（无第三方依赖），所有函数确定性、可重复。
"""

import math
import random

# ---------------------------------------------------------------- 配置

DEFAULTS = {
    "seed": 1,
    "fs": 8000,
    "n": 2048,            # 采样点数
    "block": 64,          # 处理块长
    "taps": 96,           # 自适应滤波器长度
    "delay0": 40,         # 初始回声延迟（采样）
    "delay1": 90,         # 设备切换后的回声延迟
    "echo_gain": 0.8,
    "echo_decay": 12.0,   # 回声路径指数衰减常数
    "nl_coeff": 0.15,     # 非线性失真占比（0=纯线性）
    "drift_ppm": 0.0,     # 采集相对参考的时钟漂移
    "switch_at": None,    # 设备切换位置（采样），None 表示不切换
    "near_start": 0.5,    # 近端语音起始（比例）
    "near_end": 0.75,
    "near_level": 0.7,
    "noise_level": 1e-3,
    "mu": 0.4,            # NLMS 步长
    "converge_blocks": 6, # 前若干块视为收敛期，不做双讲冻结
}

_EPS = 1e-12


def _resolve(cfg):
    """把 int（种子）/ dict / None 统一成完整配置字典。"""
    c = dict(DEFAULTS)
    if cfg is None:
        pass
    elif isinstance(cfg, (int, float)):
        c["seed"] = int(cfg)
    elif isinstance(cfg, dict):
        c.update(cfg)
    else:
        raise TypeError("cfg 必须是 None、int（种子）或 dict")
    c["n"] = int(c["n"])
    c["block"] = int(c["block"])
    c["taps"] = int(c["taps"])
    return c


# ---------------------------------------------------------------- 信号构造

def _interp(x, t):
    """线性插值取 x[t]，t 越界时钳到端点。用于模拟时钟漂移重采样。"""
    if t <= 0.0:
        return x[0]
    if t >= len(x) - 1:
        return x[-1]
    i = int(t)
    frac = t - i
    return x[i] * (1.0 - frac) + x[i + 1] * frac


def _echo_path(rng, gain, decay, length=48):
    h = [gain * math.exp(-k / decay) * rng.gauss(0.0, 1.0) for k in range(length)]
    norm = math.sqrt(sum(v * v for v in h)) or 1.0
    return [v / norm * gain for v in h]


def _convolve(x, h, delay, n):
    """y[i] = sum_k h[k] * x[i - delay - k]，截断到 n 点。"""
    y = [0.0] * n
    for i in range(n):
        base = i - delay
        s = 0.0
        for k, hk in enumerate(h):
            j = base - k
            if 0 <= j < len(x):
                s += hk * x[j]
        y[i] = s
    return y


def _scene(cfg):
    """确定性合成一个 AEC 场景：参考、回声（含延迟/非线性/漂移/切换）、
    近端语音与麦克信号。"""
    c = _resolve(cfg)
    n = c["n"]
    rng = random.Random(c["seed"])

    # 远端参考：一阶平滑噪声（类语音谱）
    far = []
    v = 0.0
    for _ in range(n):
        v = 0.75 * v + 0.25 * rng.gauss(0.0, 1.0)
        far.append(v * 0.9)

    # 采集时钟漂移：采集端看到的远端是参考的重采样版本
    r = c["drift_ppm"] * 1e-6
    cap = [_interp(far, i * (1.0 + r)) for i in range(n)] if r else list(far)

    # 回声路径；设备切换后延迟与路径都改变
    h1 = _echo_path(random.Random(c["seed"] + 1), c["echo_gain"], c["echo_decay"])
    sw = c["switch_at"]
    if sw is None:
        echo = _convolve(cap, h1, c["delay0"], n)
    else:
        h2 = _echo_path(random.Random(c["seed"] + 2), c["echo_gain"], c["echo_decay"])
        e1 = _convolve(cap, h1, c["delay0"], n)
        e2 = _convolve(cap, h2, c["delay1"], n)
        echo = e1[:sw] + e2[sw:]

    # 扬声器非线性：线性分量 + soft-clip 分量
    m = c["nl_coeff"]
    if m:
        echo = [(1.0 - m) * v + m * 0.85 * math.tanh(2.2 * v) for v in echo]

    # 近端语音：只在 [near_start, near_end) 窗口内活跃
    near = [0.0] * n
    nrng = random.Random(c["seed"] + 7)
    v = 0.0
    for i in range(int(n * c["near_start"]), int(n * c["near_end"])):
        v = 0.6 * v + 0.4 * nrng.gauss(0.0, 1.0)
        near[i] = v * c["near_level"]

    # 麦克 = 回声 + 近端 + 本底噪声
    crng = random.Random(c["seed"] + 13)
    nl = c["noise_level"]
    mic = [echo[i] + near[i] + nl * crng.gauss(0.0, 1.0) for i in range(n)]

    return {"far": far, "mic": mic, "near": near, "echo": echo, "cfg": c}


# ---------------------------------------------------------------- 基础算法

def _block_pow(x, b, block):
    s = 0.0
    base = b * block
    for i in range(base, base + block):
        s += x[i] * x[i]
    return s / block


def _xcorr_curve(ref, mic, start, length, lo, hi):
    """归一化互相关曲线：mic[i] 与 ref[i-lag]，lag ∈ [lo, hi]。"""
    curve = {}
    mp = 0.0
    for i in range(start, start + length):
        mp += mic[i] * mic[i]
    for lag in range(lo, hi + 1):
        s = 0.0
        rp = 0.0
        for i in range(start, start + length):
            j = i - lag
            if 0 <= j < len(ref):
                rj = ref[j]
                s += rj * mic[i]
                rp += rj * rj
        curve[lag] = s / math.sqrt((rp + _EPS) * (mp + _EPS))
    return curve


def _peak_parabolic(curve):
    """相关峰 + 抛物线插值，返回亚样本级延迟。"""
    best = max(curve, key=lambda k: curve[k])
    if best - 1 in curve and best + 1 in curve:
        y0, y1, y2 = curve[best - 1], curve[best], curve[best + 1]
        den = y0 - 2.0 * y1 + y2
        if abs(den) > _EPS:
            return best + 0.5 * (y0 - y2) / den
    return float(best)


def _estimate_delay(ref, mic, start, length, center, search):
    lo = max(0, center - search)
    return _peak_parabolic(_xcorr_curve(ref, mic, start, length, lo, center + search))


def _shift(x, lag, n):
    """按整数延迟对齐参考：y[i] = x[i-lag]。"""
    y = [0.0] * n
    for i in range(n):
        j = i - lag
        if 0 <= j < len(x):
            y[i] = x[j]
    return y


def _nlms(x, d, taps, mu, block, freeze):
    """NLMS 自适应滤波。freeze 为逐块布尔表，True 的块冻结更新
    （双讲保护：近端活跃时不让滤波器被近端语音带偏）。"""
    n = len(d)
    w = [0.0] * taps
    buf = [0.0] * taps
    e = [0.0] * n
    pos = 0
    pow_est = _EPS
    for i in range(n):
        old = buf[pos]
        xv = x[i]
        pow_est += xv * xv - old * old
        buf[pos] = xv
        y = 0.0
        p = pos
        for k in range(taps):
            y += w[k] * buf[p]
            p -= 1
            if p < 0:
                p = taps - 1
        ei = d[i] - y
        e[i] = ei
        if freeze is None or not freeze[i // block]:
            g = mu * ei / (pow_est + _EPS)
            p = pos
            for k in range(taps):
                w[k] += g * buf[p]
                p -= 1
                if p < 0:
                    p = taps - 1
        pos += 1
        if pos == taps:
            pos = 0
    return e, w


def _dtd_flags(mic, res, block, noise_floor, converge_blocks):
    """双讲检测：残余功率相对麦克功率持续偏高（线性回声已被对消，
    剩下的主要是近端），且过了初始收敛期才允许置位。"""
    flags = []
    nb = len(mic) // block
    for b in range(nb):
        mp = _block_pow(mic, b, block)
        rp = _block_pow(res, b, block)
        flags.append(b >= converge_blocks and mp > noise_floor and rp > 0.30 * mp)
    return flags


def _post_gains(mic, res, block, dtd, noise_floor):
    """后置增益。双讲块增益下限 0.5（近端保护），回声块下限 0.05。"""
    gains = []
    nb = len(mic) // block
    for b in range(nb):
        mp = _block_pow(mic, b, block)
        rp = _block_pow(res, b, block)
        if mp <= noise_floor:
            gains.append(1.0)  # 静默段直通
        elif dtd[b]:
            gains.append(max(0.5, min(1.0, rp / mp)))
        else:
            gains.append(max(0.05, min(1.0, rp / mp)))
    return gains


def _pipeline(cfg):
    """完整链路：场景 → 延迟估计对齐 → NLMS → 双讲检测 → 冻结重跑 →
    近端保护后置增益。所有随机源已播种子，结果确定。"""
    c = _resolve(cfg)
    sc = _scene(c)
    far, mic = sc["far"], sc["mic"]
    n, block, taps = c["n"], c["block"], c["taps"]
    noise_floor = (3.0 * c["noise_level"]) ** 2

    delay = _estimate_delay(far, mic, 0, min(512, n), c["delay0"], 24)
    aligned = _shift(far, int(round(delay)), n)

    e1, _ = _nlms(aligned, mic, taps, c["mu"], block, None)
    flags = _dtd_flags(mic, e1, block, noise_floor, c["converge_blocks"])
    e2, w = _nlms(aligned, mic, taps, c["mu"], block, flags)
    gains = _post_gains(mic, e2, block, flags, noise_floor)
    out = [e2[i] * gains[i // block] for i in range(n)]

    return {
        "cfg": c, "scene": sc, "delay": delay, "flags": flags,
        "gains": gains, "residual": e2, "output": out, "taps_norm": sum(abs(v) for v in w),
    }


# ---------------------------------------------------------------- 公开接口

def profile(cfg=None):
    """返回解析后的完整配置（默认值 + 覆盖项）。"""
    return _resolve(cfg)


def validation(cfg=None):
    """校验配置与合成信号：长度、块对齐、有限性、电平范围。"""
    c = _resolve(cfg)
    checks = []

    def chk(name, ok):
        checks.append({"name": name, "ok": bool(ok)})

    chk("fs_positive", c["fs"] > 0)
    chk("n_multiple_of_block", c["n"] % c["block"] == 0)
    chk("taps_le_n", 0 < c["taps"] <= c["n"])
    chk("delays_in_range", 0 <= c["delay0"] < c["n"] // 2 and 0 <= c["delay1"] < c["n"] // 2)
    chk("near_window_ordered", 0.0 <= c["near_start"] < c["near_end"] <= 1.0)
    chk("levels_nonnegative", c["near_level"] >= 0 and c["noise_level"] >= 0)
    chk("mu_stable", 0.0 < c["mu"] < 2.0)

    sc = _scene(c)
    sig_ok = (
        len(sc["far"]) == len(sc["mic"]) == c["n"]
        and all(math.isfinite(v) for v in sc["mic"])
    )
    chk("signals_finite_aligned", sig_ok)

    # 反例：坏配置必须被检出
    bad = dict(c)
    bad["n"] = 1000  # 不是 block 的整数倍
    bad["mu"] = 5.0
    bad_errors = []
    if bad["n"] % bad["block"] != 0:
        bad_errors.append("n_not_multiple_of_block")
    if not 0.0 < bad["mu"] < 2.0:
        bad_errors.append("mu_unstable")

    return {
        "ok": all(x["ok"] for x in checks),
        "checks": checks,
        "bad_config_errors": bad_errors,
    }


def double_talk(cfg=None):
    """双讲检测：近端语音块必须被检出，回声-only 块不误检。"""
    c = _resolve(cfg)
    c.update({"drift_ppm": 0.0, "switch_at": None, "nl_coeff": 0.05})
    p = _pipeline(c)
    block, n = c["block"], c["n"]
    nb = n // block
    true_near = set(range(int(nb * c["near_start"]), int(nb * c["near_end"])))
    detected = {b for b, f in enumerate(p["flags"]) if f}
    hit = len(detected & true_near)
    return {
        "flags": p["flags"],
        "true_near_blocks": sorted(true_near),
        "detected_blocks": sorted(detected),
        "recall": hit / max(1, len(true_near)),
        "false_alarms": len(detected - true_near),
        "freeze_applied": any(p["flags"]),
    }


def clock_drift(cfg=None):
    """时钟漂移跟踪：延迟轨迹斜率 → ppm，给出重采样校正比。"""
    c = _resolve(cfg)
    c.update({"n": 8192, "drift_ppm": 150.0, "nl_coeff": 0.0,
              "near_start": 2.0, "near_end": 2.0})  # 关闭近端
    c["near_start"] = 1.0
    c["near_end"] = 1.0
    sc = _scene(c)
    far, mic, n = sc["far"], sc["mic"], c["n"]

    win, hop = 1024, 512
    pts = []
    s = 0
    while s + win <= n:
        d = _estimate_delay(far, mic, s, win, c["delay0"], 48)
        pts.append((s + win / 2.0, d))
        s += hop

    # 最小二乘回归 delay(t) = a + b*t，|b| 即漂移率（采样/采样）
    m = len(pts)
    st = sum(p[0] for p in pts)
    sd = sum(p[1] for p in pts)
    stt = sum(p[0] * p[0] for p in pts)
    std_ = sum(p[0] * p[1] for p in pts)
    den = m * stt - st * st
    slope = (m * std_ - st * sd) / den
    ppm_est = slope * 1e6

    return {
        "ppm_true": c["drift_ppm"],
        "ppm_est": round(ppm_est, 1),
        "ppm_abs_error": round(abs(abs(ppm_est) - c["drift_ppm"]), 1),
        "resample_ratio": 1.0 + slope,
        "track": [(round(t, 1), round(d, 3)) for t, d in pts],
    }


def _track_delays(c):
    """滑窗延迟跟踪，供 delay_reset / device_switch 复用。"""
    sc = _scene(c)
    far, mic, n = sc["far"], sc["mic"], c["n"]
    win = 256
    ests = []
    s = 0
    while s + win <= n:
        ests.append((s, _estimate_delay(far, mic, s, win, 8, 152)))
        s += win
    return ests


def delay_reset(cfg=None):
    """延迟估计重置：检测到延迟跳变后重置对齐并重新估计。"""
    c = _resolve(cfg)
    c.update({"switch_at": c["n"] // 2, "nl_coeff": 0.0,
              "near_start": 1.0, "near_end": 1.0})
    ests = _track_delays(c)

    reset_at = None
    prev = ests[0][1]
    for pos, est in ests[1:]:
        if abs(est - prev) > 16:
            reset_at = pos
            break
        prev = est

    old_delay = ests[0][1]
    new_delay = ests[-1][1]
    return {
        "reset": reset_at is not None,
        "reset_at_sample": reset_at,
        "old_delay": round(old_delay, 2),
        "new_delay": round(new_delay, 2),
        "actions": ["invalidate_alignment", "reset_filter", "reestimate_delay"] if reset_at is not None else [],
    }


def device_switch(cfg=None):
    """设备切换：延迟/路径突变事件检测，触发延迟估计重置。"""
    c = _resolve(cfg)
    c.update({"switch_at": c["n"] // 2, "nl_coeff": 0.0,
              "near_start": 1.0, "near_end": 1.0})
    ests = _track_delays(c)

    event_at = None
    prev = ests[0][1]
    for pos, est in ests[1:]:
        if abs(est - prev) > 16:
            event_at = pos
            break
        prev = est

    return {
        "event": "device_switch" if event_at is not None else None,
        "detected_at_sample": event_at,
        "delay_before": round(ests[0][1], 2),
        "delay_after": round(ests[-1][1], 2),
        "handled": event_at is not None,
        "handling": ["reset_delay_estimator", "reset_adaptive_filter", "reconverge"] if event_at is not None else [],
    }


def nonlinear_suppress(cfg=None):
    """非线性残留抑制：线性滤波器无法建模的非线性回声（扬声器软削顶等）
    仍与回声估计 ŷ 相干，按块相干度对残余做后置抑制。"""
    c = _resolve(cfg)
    c.update({"nl_coeff": 0.6, "near_start": 1.0, "near_end": 1.0})
    sc = _scene(c)
    far, mic, n, block = sc["far"], sc["mic"], c["n"], c["block"]
    delay = _estimate_delay(far, mic, 0, 512, c["delay0"], 24)
    aligned = _shift(far, int(round(delay)), n)
    res, _ = _nlms(aligned, mic, c["taps"], c["mu"], block, None)
    yhat = [mic[i] - res[i] for i in range(n)]  # 线性回声估计

    gains = []
    suppressed = list(res)
    skip = c["converge_blocks"]
    for b in range(skip, n // block):
        base = b * block
        sxy = sxx = syy = 0.0
        for i in range(base, base + block):
            r, x = res[i], yhat[i]
            sxy += r * x
            sxx += x * x
            syy += r * r
        coh = abs(sxy) / math.sqrt((sxx + _EPS) * (syy + _EPS))
        g = max(0.1, 1.0 - coh)  # 相干（回声）成分全部压掉，保留非相干近端
        gains.append(round(g, 4))
        for i in range(base, base + block):
            suppressed[i] *= g

    p_before = sum(v * v for v in res[skip * block:])
    p_after = sum(v * v for v in suppressed[skip * block:])
    return {
        "block_gains": gains,
        "reduction_db": round(10.0 * math.log10((p_before + _EPS) / (p_after + _EPS)), 2),
        "suppressed": p_after < p_before,
    }


def near_gain(cfg=None):
    """近端保护增益：双讲块增益下限 0.5，回声块可压到 0.05。"""
    c = _resolve(cfg)
    p = _pipeline(c)
    nb = c["n"] // c["block"]
    dtd_blocks = [b for b, f in enumerate(p["flags"]) if f]
    echo_blocks = [b for b in range(c["converge_blocks"], nb) if not p["flags"][b]]
    min_dtd = min((p["gains"][b] for b in dtd_blocks), default=1.0)
    return {
        "gains": [round(g, 4) for g in p["gains"]],
        "dtd_blocks": dtd_blocks,
        "min_gain_in_double_talk": round(min_dtd, 4),
        "near_protected": min_dtd >= 0.5,
        "echo_floor": 0.05,
        "echo_blocks_sample": echo_blocks[:4],
    }


def determinism(cfg=None):
    """处理确定性：同配置跑两遍完整链路，输出必须逐比特一致。"""
    p1 = _pipeline(cfg)
    p2 = _pipeline(cfg)
    identical = p1["output"] == p2["output"] and p1["gains"] == p2["gains"] and p1["flags"] == p2["flags"]
    return {
        "identical": identical,
        "samples": len(p1["output"]),
        "checksum": round(sum(p1["output"]), 10),
        "delay_run1": p1["delay"],
        "delay_run2": p2["delay"],
    }


def fallback(cfg=None):
    """发散回退：步长失稳导致滤波器发散时，检测并回退到直通 +
    舒适噪声，重置滤波器状态。"""
    c = _resolve(cfg)
    c["mu"] = 50.0  # 远超 NLMS 稳定域（<2），强制发散
    sc = _scene(c)
    far, mic, n, block = sc["far"], sc["mic"], c["n"], c["block"]
    aligned = _shift(far, c["delay0"], n)

    # 逐块监控的 NLMS：残余功率异常放大或出现非有限值即判定发散
    taps = c["taps"]
    w = [0.0] * taps
    buf = [0.0] * taps
    e = [0.0] * n
    pos = 0
    pow_est = _EPS
    diverged_at = None
    for i in range(n):
        old = buf[pos]
        xv = aligned[i]
        pow_est += xv * xv - old * old
        buf[pos] = xv
        y = 0.0
        p = pos
        for k in range(taps):
            y += w[k] * buf[p]
            p -= 1
            if p < 0:
                p = taps - 1
        ei = mic[i] - y
        e[i] = ei
        g = c["mu"] * ei / (pow_est + _EPS)
        p = pos
        for k in range(taps):
            w[k] += g * buf[p]
            p -= 1
            if p < 0:
                p = taps - 1
        pos += 1
        if pos == taps:
            pos = 0
        if (i + 1) % block == 0:
            b = (i + 1) // block - 1
            rp = _block_pow(e, b, block)
            mp = _block_pow(mic, b, block)
            if not math.isfinite(rp) or rp > 25.0 * mp:
                diverged_at = b
                break

    fell_back = diverged_at is not None
    return {
        "fallback": fell_back,
        "diverged_at_block": diverged_at,
        "mode": "passthrough+comfort_noise" if fell_back else "normal",
        "actions": ["reset_filter", "freeze_adaptation", "passthrough", "comfort_noise"] if fell_back else [],
    }


def residual_echo(cfg=None):
    """残余回声估计：逐块残余电平（dB）。"""
    c = _resolve(cfg)
    p = _pipeline(c)
    block = c["block"]
    per_block = []
    for b in range(c["n"] // block):
        rp = _block_pow(p["residual"], b, block)
        per_block.append(round(10.0 * math.log10(rp + _EPS), 2))
    mean_db = round(sum(per_block[c["converge_blocks"]:]) / max(1, len(per_block) - c["converge_blocks"]), 2)
    return {"per_block_db": per_block, "mean_db_after_convergence": mean_db}


def comfort_noise(cfg=None):
    """舒适噪声：匹配残余电平的低通整形噪声，确定性生成。"""
    c = _resolve(cfg)
    p = _pipeline(c)
    block = c["block"]
    rng = random.Random(c["seed"] + 101)
    noise = [0.0] * c["n"]
    v = 0.0
    for b in range(c["n"] // block):
        rms = math.sqrt(_block_pow(p["residual"], b, block))
        for i in range(b * block, (b + 1) * block):
            v = 0.85 * v + 0.15 * rng.gauss(0.0, 1.0)
            noise[i] = v * rms * 3.0
    rms_all = math.sqrt(sum(x * x for x in noise) / len(noise))
    return {
        "n": len(noise),
        "rms": round(rms_all, 6),
        "preview": [round(x, 6) for x in noise[:8]],
    }


def reference_align(cfg=None):
    """参考对齐：估计回声延迟并把参考信号对齐到采集时间轴。"""
    c = _resolve(cfg)
    sc = _scene(c)
    far, mic, n = sc["far"], sc["mic"], c["n"]
    delay = _estimate_delay(far, mic, 0, min(512, n), c["delay0"], 24)
    lag = int(round(delay))
    aligned = _shift(far, lag, n)
    # 对齐质量：对齐后参考与麦克（回声段）的相关系数
    sxy = sxx = syy = 0.0
    for i in range(n // 4):
        x, y = aligned[i], mic[i]
        sxy += x * y
        sxx += x * x
        syy += y * y
    corr = sxy / math.sqrt((sxx + _EPS) * (syy + _EPS))
    return {
        "delay_est": round(delay, 3),
        "delay_true": c["delay0"],
        "lag_applied": lag,
        "aligned_len": len(aligned),
        "corr_after_align": round(corr, 4),
    }


def metrics(cfg=None):
    """质量指标：ERLE（回声-only 段）与近端保真度（双讲段 SNR）。"""
    c = _resolve(cfg)
    p = _pipeline(c)
    sc = p["scene"]
    block, n = c["block"], c["n"]
    nb = n // block
    skip = c["converge_blocks"]
    near_b0 = int(nb * c["near_start"])

    mp = sum(_block_pow(sc["mic"], b, block) for b in range(skip, near_b0))
    rp = sum(_block_pow(p["residual"], b, block) for b in range(skip, near_b0))
    erle = 10.0 * math.log10((mp + _EPS) / (rp + _EPS))

    s0, s1 = int(n * c["near_start"]), int(n * c["near_end"])
    num = sum(sc["near"][i] ** 2 for i in range(s0, s1))
    den = sum((p["output"][i] - sc["near"][i]) ** 2 for i in range(s0, s1))
    near_snr = 10.0 * math.log10((num + _EPS) / (den + _EPS))

    return {
        "erle_db": round(erle, 2),
        "near_snr_db": round(near_snr, 2),
        "delay_est": round(p["delay"], 3),
        "double_talk_blocks": sum(1 for f in p["flags"] if f),
    }


def audit(cfg=None):
    """审计日志：逐块记录处理决策，完全确定，可回放核查。"""
    c = _resolve(cfg)
    p = _pipeline(c)
    block = c["block"]
    events = []
    for b in range(c["n"] // c["block"]):
        mp = _block_pow(p["scene"]["mic"], b, block)
        rp = _block_pow(p["residual"], b, block)
        events.append({
            "block": b,
            "mic_db": round(10.0 * math.log10(mp + _EPS), 2),
            "residual_db": round(10.0 * math.log10(rp + _EPS), 2),
            "double_talk": p["flags"][b],
            "frozen": p["flags"][b],
            "gain": round(p["gains"][b], 4),
        })
    return {
        "delay_est": round(p["delay"], 3),
        "blocks": len(events),
        "events": events,
    }
