"""Build pre-architecture research report from saved results; no model execution."""
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
                     'axes.unicode_minus': False, 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
BLUE, TEAL, RED, GOLD = '#4378B5', '#178A83', '#CD645E', '#D6A343'
NAVY = '#172E46'
sources = {}


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    sources[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


names = ['clean_effective_r16', 'stability_control_b64', 'stability_replay_b64',
         'bounded_lora_control_b64', 'bounded_lora_r005_b64',
         'memory_frequency32_b64', 'memory_raw32_b64', 'memory_preserve32_b64',
         'memory_balanced100_b64', 'memory_balanced50_b64', 'balanced50_five_tasks']
results, protocols, histories = {}, {}, {}
for name in names:
    results[name] = read(f'outputs/{name}/continual_metrics.json')
    protocols[name] = read(f'outputs/{name}/run_manifest.json')
    histories[name] = read(f'outputs/{name}/history.json')
    assert not protocols[name]['protocol'].get('smoke_test', False), name
    assert protocols[name]['protocol']['model'].get('architecture', 'legacy') == 'legacy', name

TASKS = ['lowlight', 'rainh', 'dehaze']
LABELS = ['LSRW', 'RainH', 'Haze4K']
aliases = ['长程频域基线（背景）', '回放实验匹配对照', '回放＋蒸馏＋范围约束',
           '范数约束匹配对照', '逐层增量上限0.05', '频域统计32', '空间统计32',
           '逐批保留旧方向', '汇总式100%历史配额', '汇总式50%历史配额', '加入去噪后（四任务）']
alias = dict(zip(names, aliases))


def final(name):
    r = results[name]
    return r['matrix'][str(r['latest']['stage'])]


def psnr(name):
    return [final(name)[t]['psnr'] for t in TASKS]


def avg(name):
    return results[name]['latest']['acc_psnr']


def delta(a, b):
    return avg(b) - avg(a)


rows = []
for name in names:
    stage = results[name]['latest']['stage']
    for task, metrics in final(name).items():
        rows.append(dict(run=name, label=alias[name], stage=stage, task=task, **metrics,
                         acc_psnr=avg(name),
                         forgetting_psnr=results[name]['latest']['forgetting_psnr']))
        for dataset, subgroup in metrics.get('datasets', {}).items():
            rows.append(dict(run=name, label=alias[name], stage=stage, task=f'{task}/{dataset}',
                             **subgroup, acc_psnr=avg(name),
                             forgetting_psnr=results[name]['latest']['forgetting_psnr']))
with (OUT/'experiment_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
    fields = ['run', 'label', 'stage', 'task', 'psnr', 'ssim', 'lpips', 'num', 'acc_psnr', 'forgetting_psnr']
    writer = csv.DictWriter(f, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)

audit = read('outputs/output_stability_20261104_full/stability.json')
coordinate = read('outputs/gpm_coordinate_audit_20261104/audit.json')
transition = read('outputs/memory_transition_20261105/audit.json')
preserved = read('outputs/memory_transition_preserved_20261105/audit.json')
split = read('outputs/clean_splits_v1/manifest.json')
balance50 = read('outputs/memory_balanced50_b64/task_1_balanced_memory.json')
balance100 = read('outputs/memory_balanced100_b64/task_1_balanced_memory.json')
assert balance50['layers'].keys() == balance100['layers'].keys()
assert all(v['covariance_sha256'] == balance100['layers'][k]['covariance_sha256']
           for k, v in balance50['layers'].items())
assert protocols['memory_raw32_b64']['protocol']['memory_source']['source_sha256'] == \
    protocols['memory_frequency32_b64']['protocol']['memory_source']['source_sha256']
assert protocols['bounded_lora_control_b64']['protocol']['update_bound']['source_sha256'] == \
    protocols['bounded_lora_r005_b64']['protocol']['update_bound']['source_sha256']
assert protocols['stability_control_b64']['protocol']['stability']['source_sha256'] == \
    protocols['stability_replay_b64']['protocol']['stability']['source_sha256']


def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(FIG/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)
    return FIG/f'{name}.png'


def grouped(ax, values, labels, groups, colors, ylabel, top=None):
    x = np.arange(len(groups)); width = .78 / len(values)
    for i, (v, label, color) in enumerate(zip(values, labels, colors)):
        bars = ax.bar(x + (i-(len(values)-1)/2)*width, v, width, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
    ax.set_xticks(x, groups); ax.set_ylabel(ylabel)
    ax.set_ylim(0, top or max(max(v) for v in values)*1.22)
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc='upper right')


fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
grouped(ax[0], [[audit['tasks'][t]['means']['before_psnr'] for t in TASKS[:2]],
                [audit['tasks'][t]['means']['after_psnr'] for t in TASKS[:2]]],
        ['Haze4K前', 'Haze4K后'], LABELS[:2], [BLUE, RED], '验证 PSNR / dB', 27)
grouped(ax[1], [[audit['tasks'][t]['layer_mean_coverage']*100 for t in TASKS[:2]]],
        ['真实卷积输入覆盖率'], LABELS[:2], [TEAL], '层均覆盖率 / %', 110)
ax[0].set_title('全量旧任务验证：输出严重漂移')
ax[1].set_title('抽样覆盖率：正交目标并不完整')
chart_audit = save(fig, '01_output_stability')

fig, ax = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
pairs = [('stability_control_b64','stability_replay_b64','额外旧数据：组合诊断'),
         ('bounded_lora_control_b64','bounded_lora_r005_b64','无回放：范数上限'),
         ('memory_frequency32_b64','memory_raw32_b64','无回放：统计来源')]
for a, (control, treatment, title) in zip(ax, pairs):
    grouped(a, [psnr(control), psnr(treatment)], ['匹配对照', '实验组'], LABELS,
            [BLUE, TEAL], '测试 PSNR / dB', 28)
    a.set_title(f'{title}\n平均变化 {delta(control,treatment):+.3f} dB', fontsize=13)
chart_pairs = save(fig, '02_matched_ablation_psnr')

fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
for i, metric in enumerate(['ssim', 'lpips']):
    values = [[np.mean([final(n)[t][metric] for t in TASKS]) for n in pair[:2]] for pair in pairs]
    grouped(ax[i], [[v[0] for v in values], [v[1] for v in values]],
            ['匹配对照', '实验组'], ['回放组合', '范数上限', '空间统计'],
            [BLUE, TEAL], '三任务宏平均 ' + metric.upper(), .8)
    ax[i].set_title('SSIM 越高越好' if metric == 'ssim' else 'LPIPS 越低越好')
chart_quality = save(fig, '03_ssim_lpips')

fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
for i, kernel in enumerate(['kernel3', 'kernel1']):
    values = [[coordinate['scores'][t+'_full']['groups'][kernel][key]*100 for t in TASKS[:2]]
              for key in ['freq_fit_raw_coverage', 'raw_fit_raw_coverage']]
    grouped(ax[i], values, ['新拟合同秩频域基', '新拟合同秩空间基'], LABELS[:2],
            [BLUE, TEAL], '真实空间输入能量覆盖率 / %', 115)
    ax[i].set_title('3×3 层：空间邻域差异明显' if i == 0 else '1×1 层：差异较小')
chart_coordinate = save(fig, '04_statistical_mismatch')

fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
grouped(ax[0], [[transition['summary']['coverage_before']*100, preserved['summary']['coverage_before']*100],
                [transition['summary']['coverage_after']*100, preserved['summary']['coverage_after']*100]],
        ['边界更新前','边界更新后'], ['原有有损压缩','保留旧子空间'], [BLUE, TEAL], '固定旧特征层均覆盖率 / %', 112)
grouped(ax[1], [psnr('memory_balanced100_b64'), psnr('memory_balanced50_b64')],
        ['100%历史配额','50%历史配额'], LABELS, [BLUE, TEAL], '测试 PSNR / dB', 28)
ax[0].set_title('局部审计：保留方向可避免覆盖率下降')
ax[1].set_title('严格配额对照：50%改善整体而非所有任务')
chart_balance = save(fig, '05_memory_tradeoff')

fig, ax = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
for a, task, label in zip(ax, TASKS, LABELS):
    for name, color, display in [('memory_balanced100_b64',BLUE,'100%历史配额'),
                                  ('memory_balanced50_b64',TEAL,'50%历史配额')]:
        points = [(row['epoch'], row['validation']['tasks'][task]['psnr'])
                  for row in histories[name] if 'validation' in row]
        a.plot(*zip(*points), 'o-', color=color, label=display)
    a.set_title(label);a.set_xlabel('Haze4K epoch');a.set_ylabel('验证 PSNR / dB')
    a.set_xticks([5,10,15,20]);a.grid(alpha=.2);a.legend(frameon=False, fontsize=10)
chart_curves = save(fig, '06_balanced_validation_curves')

before = final('memory_balanced50_b64'); after = final('balanced50_five_tasks')
old_drop = np.mean([after[t]['psnr']-before[t]['psnr'] for t in TASKS])
fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
grouped(ax[0], [psnr('memory_balanced50_b64'),psnr('balanced50_five_tasks')],
        ['去噪前','去噪后'],LABELS,[BLUE,RED],'旧任务测试 PSNR / dB',28)
denoise = after['denoise']['datasets']
grouped(ax[1], [[denoise[t]['psnr'] for t in ['DIV2K','BSD68']]],
        ['AWGN25 去噪'],['DIV2K (100张)','BSD68 (68张)'],[TEAL],'测试 PSNR / dB',32)
ax[0].set_title(f'扩展至第四任务：旧任务平均下降 {-old_drop:.3f} dB')
ax[1].set_title('新任务已学到；但仍有明显稳定性代价')
chart_denoise = save(fig, '07_four_task_extension')

fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
families = ['回放组合（诊断）','范数上限0.05','空间统计','50%历史配额']
comparisons = [(a,b) for a,b,_ in pairs] + [('memory_balanced100_b64','memory_balanced50_b64')]
gains = [delta(a,b) for a,b in comparisons]
with (OUT/'paired_effects.csv').open('w', newline='', encoding='utf-8-sig') as f:
    fields=['experiment','control','treatment','delta_mean_psnr','delta_lowlight_psnr',
            'delta_rainh_psnr','delta_dehaze_psnr','delta_mean_ssim','delta_mean_lpips']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for label,(a,b) in zip(families,comparisons):
        row=dict(experiment=label,control=a,treatment=b,delta_mean_psnr=delta(a,b))
        row.update({f'delta_{t}_psnr':final(b)[t]['psnr']-final(a)[t]['psnr'] for t in TASKS})
        row.update({f'delta_mean_{m}':np.mean([final(b)[t][m]-final(a)[t][m] for t in TASKS])
                    for m in ['ssim','lpips']})
        writer.writerow(row)
bars=ax.barh(families,gains,color=[GOLD,BLUE,TEAL,TEAL]);ax.bar_label(bars,fmt='+%.3f dB',padding=6)
ax.set_xlim(0,max(gains)*1.3);ax.invert_yaxis();ax.set_xlabel('各自匹配对照内的三任务宏平均 PSNR 差值 / dB')
ax.set_title('不同起点/预算/选模：只能解释各对照内差值，不能拼接成累计增益')
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
chart_gains=save(fig,'08_within_pair_gains')

probe=read('outputs/forgetting_probe_20261028/summary.json')
basis=read('outputs/basis_study_20261029/summary.json')
gpm=read('outputs/gpm_study_20261029/summary.json')
fig, ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for a,data,arms,legend,title in [
    (ax[0],probe,['baseline','protected'],['原保护','完整更新保护'],'保护范围探索'),
    (ax[1],basis,['current_basis','effective_r8','effective_r16'],['原基','有效基r8','有效基r16'],'有效基与私有基础秩探索'),
    (ax[2],gpm,['current32','raw64','raw128'],['原统计32','空间统计64','空间统计128'],'更大容量联合探索')]:
    grouped(a,[[data['arms'][arm]['metrics'][t]['psnr'] for t in TASKS[:2]] for arm in arms],
            legend,LABELS[:2],[BLUE,TEAL,GOLD],'测试 PSNR / dB',27)
    a.set_title(title)
chart_early=save(fig,'09_early_exploration')

slides = []


def slide(title, subtitle, bullets, chart=None, foot=''):
    slides.append(dict(title=title,subtitle=subtitle,bullets=bullets,chart=chart,foot=foot))


slide('正交 LoRA 图像复原：阶段研究汇报', '更换网络结构前的实验总结｜正式结果截至 2026-11-05｜汇编 2026-11-07', [
    '研究目标：不使用旧数据回放，以有限子空间记忆保护连续图像复原能力。',
    '主线：完整更新正交 → 输出稳定性诊断 → 空间统计 → 历史方向配额 → 第四任务检验。',
    '阶段结论：空间匹配与平衡配额均有收益，但有限记忆仍未解决长期遗忘。',
    '汇报范围：旧 NAFNet 风格骨干；不包含 stable_v1 的结构改动及其训练结果。'],
    foot='数据来自本地正式 JSON/日志；排除 smoke/capacity 测试的质量指标。')
slide('01｜先明确比较协议，而不是拼接一张排行榜', '数据清洗与可复现实验是本阶段的重要工作', [
    '干净配对：LSRW 训练/验证/测试=5036/564/50；RainH=1128/126/100；Haze4K=2700/300/1000。',
    '三任务顺序固定；训练留出集选模；正式测试在任务边界执行。PSNR/SSIM按[0,1]截断，LPIPS沿用原始输出协议。',
    '短程主对照：batch64、patch128、新任务20轮、AdamW、lr=2e-4、L1+0.1 LPIPS、严格FP32。',
    '回放对照按已见任务验证均值选模；其余短程对照只按新任务验证PSNR选模。历史长程基线为batch16、400轮。',
    '平均PSNR为任务宏平均；遗忘按旧任务历史阶段最佳测试PSNR减当前PSNR、取非负值后，对旧任务平均。',
    '单种子短程结果仅支持当前配置下的观察；不同实验家族不直接排名，也不能宣称统计显著。'])
slide('02｜问题确认：正交有效，但旧任务输出并不稳定', '历史 clean-effective-r16：验证全量配对审计；不是新增训练结果', [
    'LSRW 验证PSNR下降7.087 dB；RainH下降9.257 dB。下降超过1 dB的图像分别为483/564、123/126。',
    '固定参数未变；增量与A@B一致；GPM更新前向误差为0。证据不支持将退化归于冻结失败或重复合并。',
    '投影泄漏约1e-7仍可能伴随真实输入未覆盖分量响应；抽样覆盖率不是全网络总体能量比例。'],
    chart_audit,'来源：output_stability_20261104_full/stability.json；覆盖率每任务8张验证图、每层每图最多128 patches。')
slide('03｜匹配消融：哪些路径有效，哪些效果有限？', '每个子图只在自身匹配对照内解释；前三任务最终测试', [
    f'回放组合平均 +{delta(*comparisons[0]):.3f} dB：额外旧数据与计算有效，但不是最终无回放路线。',
    f'范数上限0.05平均 +{delta(*comparisons[1]):.3f} dB：改善较小，Haze4K略下降，不能视为解决遗忘。',
    f'空间统计平均 +{delta(*comparisons[2]):.3f} dB：RainH明显改善，LSRW并未同步改善。'],
    chart_pairs,'回放为监督回放＋蒸馏＋范围约束的组合；两个旧任务各64张训练图，不可拆分归因。')
slide('04｜不只看PSNR：同时检查SSIM与LPIPS', '同一组正式测试数据；均为三个任务各指标的宏平均', [
    '空间统计的PSNR收益没有转化为全部感知指标提升，应承认指标之间的权衡。',
    '原始输出仍有极端值；截断后的PSNR/SSIM不能代替原始输出稳定性诊断。'],
    chart_quality,'LPIPS越低越好；不能把含异常值的原始LPIPS与截断后LPIPS混作同协议。')
slide('05｜机制证据：频域patch并不等于真实卷积输入', '维度和排列一致不代表统计主子空间相同；重点差异在3×3邻域', [
    '固定模型与相同秩，在分离的TRAIN拟合组/VALIDATION评分组上比较两类记忆基。',
    '空间基在3×3层覆盖率高约9.47/13.28个百分点；1×1层差异小于0.4个百分点。',
    '这属于有限抽样机制证据；局部反事实投影并非全网络训练效果，也不能外推等比例PSNR收益。'],
    chart_coordinate,'来源：gpm_coordinate_audit_20261104/audit.json；每任务8个拟合TRAIN组、16个VALIDATION组。')
slide('06｜保留更早方向，不等于更好地保护所有历史任务', '固定32维记忆：从“全部保留”转向“历史池与当前残差平衡”', [
    '局部审计：压缩使旧特征覆盖率81.25%→67.32%；保留旧空间为81.25%→81.29%，不是准确率或方向数量比例。',
    f'严格配额对照：50%相对100%三任务平均 +{delta(*comparisons[3]):.3f} dB；遗忘3.358→1.976 dB。',
    '代价：LSRW下降1.984 dB；收益：RainH提升4.750 dB。50%不是每个历史任务各占50%，也不是全任务均优。'],
    chart_balance,'左：固定RainH末模型下LSRW留出特征的边界审计。右：同RainH权重、同协方差，只改历史配额。')
slide('07｜验证曲线：历史保护存在稳定性—可塑性取舍', '同一RainH边界，Haze4K训练第5/10/15/20轮的验证结果', [
    '100%历史配额偏向维持更早LSRW特征，RainH在后续训练中明显退化。',
    '50%配额改善RainH保持，但放弃部分LSRW优势；Haze4K学习曲线也随保护空间改变。',
    '两组选模均只看Haze4K验证PSNR；旧任务曲线用于诊断，不参与优化。'],
    chart_curves,'来源：memory_balanced50_b64/history.json 与 memory_balanced100_b64/history.json；验证不是测试。')
slide('08｜第四任务检验：局部改进还不具备长期保证', 'LSRW → RainH → Haze4K → AWGN25去噪；未完成GOPRO正式训练', [
    f'加入去噪后，旧三任务平均17.742→14.983 dB，下降{-old_drop:.3f} dB；RainH下降5.431 dB。',
    '去噪DIV2K=25.136 dB、BSD68=21.753 dB；新任务能学习，但旧任务保持再次恶化。',
    '四任务平均17.179包含去噪，不能直接与三任务平均17.742比较来衡量遗忘。'],
    chart_denoise,'去噪采用512窗口/64重叠推理及面积加权patch LPIPS；旧三任务评估方式保持不变。')
slide('09｜阶段结论：三个证据支持的判断', '只报告各自匹配对照内的收益，不构造累计增益', [
    '① 对保存子空间正交 ≠ 对旧任务函数保持不变；统计不匹配、有限容量与特征漂移共同构成风险。',
    '② 空间统计与平衡配额是值得保留的无回放机制；更新范数上限单独作用较弱。',
    '③ 当前结果还不能宣称消除遗忘、五任务有效或达到最佳整体复原性能。'],
    chart_gains,'回放诊断不计为无回放创新；各家族起点、训练预算和选模规则不同。')
slide('10｜汇报可用的结论与后续计划', '正交LoRA仍是核心创新方向；结构修改的研究留在下一阶段单独汇报', [
    '推荐表述：完成了无回放正交LoRA的机制排查与匹配消融，确认空间统计匹配和平衡记忆分配能改善阶段性保持。',
    '避免表述：已解决灾难性遗忘；50%配额是全局最优；所有实验增益可相加；回放提升证明正交机制有效。',
    '后续验证：记录旧任务逐层/逐轮漂移；检查异常输出；在匹配骨干、预算、选模下复验，并补多种子。',
    '旧空间配额是历史池级启发式，不具备逐任务公平保证；继续扩展任务前须检验保护方向的长期覆盖。',
    '结果可追溯：随附CSV、全部来源SHA256、PNG/SVG图表、可重建脚本。训练产物未修改。'])
slide('附录｜早期探索如何引出后续研究', '2026-10-28/29旧配对协议探针：只作为研究过程，不与干净配对主结果混排', [
    '完整更新保护改善LSRW保持，但牺牲RainH学习；促使后续同时检查稳定性与可塑性。',
    '有效基r16提高RainH适应能力，但仍有旧任务代价；后续主实验沿用投影后有效基与私有基础秩16。',
    '更大容量联合探索没有明显收益；32/64/128组同时涉及统计来源等差异，不能单独归因为容量。'],
    chart_early,'来源：forgetting_probe_20261028、basis_study_20261029、gpm_study_20261029；旧配对探针，不作正式基准。')

# Render a PDF deck and editable PowerPoint text, with identical chart assets.
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)


def textbox(slide_obj, text, x, y, w, h, size=16, color=NAVY, bold=False):
    frame=slide_obj.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame
    frame.word_wrap=True
    for i,line in enumerate(text.split('\n')):
        p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        p.text=line;p.font.name='Noto Sans CJK SC';p.font.size=Pt(size)
        p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color.lstrip('#'))
    return frame


def display_wrap(text, width):
    lines=[];current='';units=0
    for char in text:
        size=2 if unicodedata.east_asian_width(char) in ('W','F') else 1
        if units+size>width:
            lines.append(current);current='';units=0
        current+=char;units+=size
    return '\n'.join(lines+[current])


with PdfPages(OUT/'研究汇报_更换模型前.pdf') as pdf:
    for i,s in enumerate(slides,1):
        f=plt.figure(figsize=(16,9));f.text(.045,.92,s['title'],size=25,color=NAVY,weight='bold')
        f.text(.045,.865,s['subtitle'],size=13,color='#607286')
        ppt=prs.slides.add_slide(prs.slide_layouts[6])
        ppt.notes_slide.notes_text_frame.text='\n'.join([s['subtitle'],*s['bullets'],s['foot']])
        textbox(ppt,s['title'],.55,.35,12.2,.6,25,bold=True)
        textbox(ppt,s['subtitle'],.55,1.02,12.2,.5,13,color='#607286')
        if s['chart']:
            a=f.add_axes([.045,.285,.91,.525]);a.imshow(plt.imread(s['chart']));a.axis('off')
            ppt.shapes.add_picture(str(s['chart']),Inches(.6),Inches(1.52),width=Inches(12.1),height=Inches(3.85))
            y=.235;py=5.55;size=12.0
        else:
            y=.76;py=1.85;size=16 if len(s['bullets'])>5 else 17
        for j,b in enumerate(s['bullets']):
            if not s['chart']:
                y=.76-j*.65/len(s['bullets'])
                py=1.85+j*4.85/len(s['bullets'])
            wrapped=display_wrap('• '+b, width=182 if s['chart'] else 122)
            f.text(.055,y,wrapped,size=size,color=NAVY,va='top',linespacing=1.6)
            lines=wrapped.count('\n')+1
            if s['chart']:
                textbox(ppt,'• '+b,.68,py,12.0,.42 if lines==1 else .62,12)
                y-=.053*lines;py+=.37*lines
            else:
                textbox(ppt,wrapped,.68,py,12.0,4.85/len(s['bullets'])-.08,size)
        f.text(.045,.028,s['foot'] or '研究范围截至网络结构更换前；不包含 stable_v1 结果。',size=8.5,color='#607286')
        f.text(.96,.028,f'{i:02}/{len(slides):02}',ha='right',size=9,color='#607286')
        textbox(ppt,s['foot'] or '研究范围截至网络结构更换前；不包含stable_v1结果。',.55,7.12,11.8,.24,9,color='#607286')
        textbox(ppt,f'{i:02}/{len(slides):02}',12.3,7.1,.6,.25,9,color='#607286')
        pdf.savefig(f);f.savefig(OUT/f'preview_{i:02}.png',dpi=100);plt.close(f)
prs.save(OUT/'研究汇报_更换模型前.pptx')

headers='| 实验 | LSRW PSNR | RainH PSNR | Haze4K PSNR | 宏平均PSNR | 遗忘 |\n|---|---:|---:|---:|---:|---:|'
table=[headers]
for n in names[:-1]:
    table.append('| '+alias[n]+' | '+' | '.join(f'{x:.4f}' for x in psnr(n))+
                 f" | {avg(n):.4f} | {results[n]['latest']['forgetting_psnr']:.4f} |")
markdown=['# 更换模型前的正交LoRA研究总结','',
          '范围：2026-10-29至2026-11-05的干净配对实验与后续诊断；不包含stable_v1结构改动。',
          '汇编日期：2026-11-07。适合约10–15分钟组会汇报。', '',
          '## 核心结论','',
          f'空间统计相对频域匹配对照的三任务宏平均PSNR提高{gains[2]:.4f} dB；',
          f'50%历史配额相对100%汇总配额提高{gains[3]:.4f} dB，但并非每个任务都改善。',
          f'第四任务去噪加入后旧三任务平均下降{-old_drop:.4f} dB，因此尚未解决长期遗忘。',
          '上述收益属于不同匹配对照，不能相加，也不能将额外旧数据回放组当作无回放方法。','',
          '## 汇报讲稿与图表','']
for i,s in enumerate(slides,1):
    markdown.extend([f"### {i}. {s['title']}",'',s['subtitle'],''])
    markdown.extend(['- '+b for b in s['bullets']]);markdown.append('')
    if s['chart']:markdown.extend([f"![{s['title']}](figures/{s['chart'].name})",''])
    if s['foot']:markdown.extend(['口径/来源：'+s['foot'],''])
markdown.extend(['## 正式三任务结果总表（不是统一排行榜）','','\n'.join(table),'',
    '注意：长程基线与短程分支不等预算；回放组按已见任务平均验证选模，范数/统计/配额组只按新任务选模。',
    '逐批preserve与汇总式100%不是同一更新算法，二者只可作为背景参考；配额的严格对照为50%与100%汇总两组。','',
    '## 四任务扩展','',
    f'去噪后旧任务宏平均：{np.mean(psnr("balanced50_five_tasks")):.4f} dB；相对去噪前变化{old_drop:.4f} dB。',
    'GOPRO数据划分和入口已准备，但截止本报告没有正式GOPRO训练结果，不把“具备五任务代码”写成“完成五任务实验”。','',
    '## 早期探索为何不放入主对比表','',
    '2026-10-28/29的完整更新保护、GPM64/128与有效秩探针属于干净配对协议确立前的探索。',
    '这些探针启发了完整更新保护和有效基/私有基础秩16，但不与干净配对正式结果混排。',
    '主表也排除smoke、capacity与仅离线拟合基的PSNR“推断”。','',
    '## 可复现与来源','',
    '`experiment_metrics.csv`保留各正式实验全部PSNR/SSIM/LPIPS；`sources.json`记录来源路径与SHA256。',
    '重建：在仓库根目录运行 `python reports/pre_model_20261107/build_report.py`（系统Python包含matplotlib和python-pptx）。',
    '报告只读取JSON，不加载模型、不访问GPU、不修改训练文件。',''])
(OUT/'研究总结.md').write_text('\n'.join(markdown),encoding='utf-8')
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items()), 'Source changed during reporting'
(OUT/'sources.json').write_text(json.dumps(dict(scope='legacy architecture only, formal results through 2026-11-05',
    report_date='2026-11-07',sources_sha256=sources,
    verified=dict(no_smoke=True,no_stable_v1=True,quota_covariance_hashes_equal=True,
                  paired_source_hashes_equal=True)),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(slides=len(slides),figures=9,sources=len(sources),output=str(OUT)),ensure_ascii=False))
